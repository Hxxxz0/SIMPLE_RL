"""RSL-RL PPO construction using the same model and algorithm scale as SMP."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
from torch.utils.data import DataLoader

from simple.grasp_rl.bc import BcDataset
from simple.grasp_rl.curriculum import TrainingCurriculum, load_curriculum
from simple.grasp_rl.hard_targets import (
    HardTargetManifest,
    load_hard_target_manifest,
)
from simple.grasp_rl.policy import load_actor
from simple.grasp_rl.ppo_integrity import PpoIntegrityAuditor
from simple.grasp_rl.rewards import DEFAULT_TASK_REWARD_PROFILE
from simple.grasp_rl.schema import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_DIM,
    REFERENCE_ACTOR_OBS_V2_DIM,
)
from simple.grasp_rl.vec_env import DistributedGraspVecEnv
from simple.grasp_rl.task_spec import (
    DEFAULT_TASK,
    TaskSpecV2,
    checkpoint_task_metadata,
    get_task_spec,
    validate_task_metadata,
)


def _load_ancestor_critic(
    runner: OnPolicyRunner,
    actor_checkpoint: dict,
    device: str,
) -> str | None:
    """Restore the nearest PPO critic through a chain of BC initializations."""
    source = actor_checkpoint.get("config", {}).get("initialize_checkpoint")
    visited: set[str] = set()
    while source:
        source = str(Path(source).resolve())
        if source in visited:
            raise RuntimeError(f"Cyclic initialize_checkpoint chain at {source}")
        visited.add(source)
        checkpoint = torch.load(source, map_location=device, weights_only=False)
        if "critic_state_dict" in checkpoint:
            runner.alg._raw_critic.load_state_dict(checkpoint["critic_state_dict"])
            return source
        source = checkpoint.get("config", {}).get("initialize_checkpoint")
    return None


def _install_bc_anchor(
    runner: OnPolicyRunner,
    processed_dir: str,
    sources: tuple[str, ...],
    batch_size: int,
    weight: float,
    manipulation_dimension_weight: float,
    max_grad_norm: float,
    device: str,
) -> None:
    """Add real-trajectory BC gradients to every PPO optimizer step.

    Applying a scaled BC loss in a separate Adam step does not implement the
    intended relative weighting: Adam is approximately invariant to a pure
    gradient rescaling.  Injecting the BC gradient before the existing PPO
    step makes ``weight`` an actual trade-off between the two objectives and
    also avoids advancing Adam's state a second time per minibatch.
    """
    if weight <= 0.0:
        raise ValueError("BC anchor weight must be positive")
    actor = runner.alg.get_policy()
    if getattr(actor, "is_recurrent", False):
        raise ValueError(
            "Flat shuffled BC anchoring is invalid for a recurrent actor; "
            "use recurrent sequence BC for the warm start"
        )
    actor_observation_dim = int(actor.mlp[0].weight.shape[1])
    loader = DataLoader(
        BcDataset(
            processed_dir,
            "train",
            sources,
            reference_conditioning=(
                actor_observation_dim in (REFERENCE_ACTOR_OBS_DIM, REFERENCE_ACTOR_OBS_V2_DIM)
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    iterator = iter(loader)
    base_update = runner.alg.update
    optimizer = runner.alg.optimizer
    base_step = optimizer.step
    if manipulation_dimension_weight < 1.0:
        raise ValueError("manipulation_dimension_weight must be at least 1")
    dimension_weights = torch.ones(ACTION_DIM, device=device)
    # The intervention audit shows that these are the necessary and
    # sufficient action dimensions for grasp success: replacing only the
    # right arm and right hand with replay actions changes success from 20%
    # to 100%.  Preserve them more strongly than locomotion/body commands.
    dimension_weights[7:14] = manipulation_dimension_weight
    dimension_weights[21:28] = manipulation_dimension_weight

    anchor_losses: list[float] = []

    def anchored_step(*args, **kwargs):
        nonlocal iterator
        try:
            observations, actions, sample_weights = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            observations, actions, sample_weights = next(iterator)
        observations = observations.to(device)
        actions = actions.to(device)
        sample_weights = sample_weights.to(device)
        prediction = actor(
            TensorDict(
                {"actor": observations},
                batch_size=[len(observations)],
                device=device,
            ),
            stochastic_output=False,
        )
        element_loss = F.smooth_l1_loss(prediction, actions, reduction="none")
        per_sample = (
            element_loss * dimension_weights
        ).sum(-1) / dimension_weights.sum()
        anchor_loss = (per_sample * sample_weights).mean()
        # PPO gradients are already populated here. Add BC to the same
        # gradients so Adam sees the requested objective weighting.
        (weight * anchor_loss).backward()
        # Respect the same trust region used by PPO.  A looser second clip
        # would let the auxiliary loss bypass PPO's configured protection.
        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        anchor_losses.append(float(anchor_loss.detach()))
        return base_step(*args, **kwargs)

    optimizer.step = anchored_step

    def anchored_update():
        anchor_losses.clear()
        metrics = base_update()
        metrics["bc_anchor"] = float(sum(anchor_losses) / len(anchor_losses))
        return metrics

    runner.alg.update = anchored_update


def _install_teacher_anchor(
    runner: OnPolicyRunner,
    checkpoint: str,
    weight: float,
    device: str,
    task: str,
    action_transform: str | Path,
) -> None:
    """Add stateless teacher supervision on each current PPO minibatch."""
    if weight <= 0.0:
        raise ValueError("teacher anchor weight must be positive")
    teacher = load_actor(
        checkpoint,
        device,
        expected_task=task,
        action_transform=action_transform,
    )
    actor = runner.alg.get_policy()
    optimizer = runner.alg.optimizer
    base_step = optimizer.step
    base_update = runner.alg.update
    base_forward = actor.forward
    captured_observations: torch.Tensor | None = None
    anchor_losses: list[float] = []
    dimension_weights = torch.ones(ACTION_DIM, device=device)
    # Whole-body imitation can match the large lifting motion while averaging
    # away millimeter-sensitive grasp closure. Emphasize the right hand/arm
    # for the on-policy teacher objective; the task reward still decides
    # whether those commands produce bilateral contact and stable lift.
    dimension_weights[7:14] = 10.0
    dimension_weights[21:28] = 5.0

    def capturing_forward(observations, *args, **kwargs):
        nonlocal captured_observations
        result = base_forward(observations, *args, **kwargs)
        if kwargs.get("stochastic_output", False):
            captured_observations = observations["actor"].detach()
        return result

    actor.forward = capturing_forward

    def anchored_step(*args, **kwargs):
        if captured_observations is None:
            raise RuntimeError("PPO actor observations were not captured")
        teacher_observation_dim = int(
            getattr(teacher, "grasp_observation_dim", ACTOR_OBS_DIM)
        )
        teacher_observations = captured_observations[:, :teacher_observation_dim]
        observation_dict = TensorDict(
            {"actor": captured_observations},
            batch_size=[len(captured_observations)],
            device=device,
        )
        teacher_observation_dict = TensorDict(
            {"actor": teacher_observations},
            batch_size=[len(teacher_observations)],
            device=device,
        )
        with torch.no_grad():
            targets = teacher(teacher_observation_dict, stochastic_output=False)
        prediction = actor(observation_dict, stochastic_output=False)
        element_loss = F.smooth_l1_loss(prediction, targets, reduction="none")
        anchor_loss = (
            element_loss * dimension_weights
        ).sum(-1).mean() / dimension_weights.sum()
        (weight * anchor_loss).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        anchor_losses.append(float(anchor_loss.detach()))
        return base_step(*args, **kwargs)

    optimizer.step = anchored_step

    def anchored_update():
        anchor_losses.clear()
        metrics = base_update()
        metrics["teacher_anchor"] = float(
            sum(anchor_losses) / len(anchor_losses)
        )
        return metrics

    runner.alg.update = anchored_update


@dataclass
class PpoTrainConfig:
    task: str = DEFAULT_TASK
    reward_audit: str | None = None
    num_envs: int = 56
    iterations: int = 1500
    seed: int = 42
    device: str = "cuda:0"
    reward_variant: str = "task_only"
    ws: float = 4.0
    task_reward_weight: float = 0.02
    smp_reward_weight: float = 0.01
    task_reward_profile: str = DEFAULT_TASK_REWARD_PROFILE
    save_interval: int = 500
    resume: str | None = None
    warm_start: str | None = None
    actor_warm_start: str | None = None
    worker_devices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    rsi_dataset: str | None = None
    rsi_processed: str | None = None
    rsi_prefix: tuple[int, int] = (75, 115)
    rsi_phase: tuple[float, float] | None = None
    rsi_stage: str | None = None
    rsi_episodes: tuple[int, ...] | None = None
    rsi_probability: float = 0.0
    rsi_scene_hold_episodes: int = 32
    rsi_randomize_target: bool = False
    target_position_jitter_xy: tuple[float, float] | None = (0.025, 0.03)
    target_position_offset_center_xy: tuple[float, float] = (0.0, 0.0)
    target_yaw_jitter: float = 0.15
    action_std: float = 0.30
    manipulation_action_std: float | None = None
    observation_noise: bool = True
    learning_rate: float = 1e-3
    actor_learning_rate_scale: float = 1.0
    learning_schedule: str = "adaptive"
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    exploration_hold_steps: int = 1
    freeze_actor_normalizer: bool = False
    bc_anchor_weight: float = 0.0
    bc_anchor_processed: str | None = None
    bc_anchor_sources: tuple[str, ...] = ("bc",)
    bc_anchor_batch_size: int = 1024
    bc_anchor_manipulation_weight: float = 10.0
    teacher_anchor_checkpoint: str | None = None
    teacher_anchor_weight: float = 0.0
    reference_processed: str | None = None
    reference_source: str = "bc"
    reference_splits: tuple[str, ...] = ("train", "val", "test")
    reference_reward_weight: float = 0.0
    reference_rank_max: int = 0
    reference_base_episode_probability: float = 0.0
    reference_action_noise_std: float = 0.0
    reference_action_noise_hold_steps: int = 25
    recurrent_actor: bool = False
    plan_conditioned_actor: bool = False
    rnn_hidden_dim: int = 256
    max_grad_norm: float = 0.1
    curriculum_config: str | None = None
    hard_target_manifest: str | None = None


def rsl_config(config: PpoTrainConfig) -> dict:
    actor_config = {
        "class_name": (
            "simple.grasp_rl.models.ClippedRNNModel"
            if config.recurrent_actor
            else "simple.grasp_rl.models.PlanConditionedMLPModel"
            if config.plan_conditioned_actor
            else "simple.grasp_rl.models.ClippedMLPModel"
        ),
        "hidden_dims": (512, 256, 128),
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": (
                "simple.grasp_rl.distribution."
                "TemporallyCorrelatedGaussianDistribution"
            ),
            "init_std": config.action_std,
            "std_type": "scalar",
            "learn_std": False,
            "hold_steps": config.exploration_hold_steps,
        },
    }
    if config.recurrent_actor:
        actor_config.update(
            {
                "rnn_type": "gru",
                "rnn_hidden_dim": config.rnn_hidden_dim,
                "rnn_num_layers": 1,
            }
        )
    return {
        "seed": config.seed,
        "num_steps_per_env": config.num_steps_per_env,
        "save_interval": config.save_interval,
        "logger": "tensorboard",
        "run_name": (
            f"{config.task}-grasp-{config.reward_variant}-{config.task_reward_profile}"
            f"-seed{config.seed}"
        ),
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "actor": actor_config,
        "critic": {
            "class_name": "simple.grasp_rl.models.ClippedMLPModel",
            "hidden_dims": (512, 256, 128),
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "num_learning_epochs": config.num_learning_epochs,
            "num_mini_batches": config.num_mini_batches,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "learning_rate": config.learning_rate,
            "max_grad_norm": config.max_grad_norm,
            "use_clipped_value_loss": True,
            "schedule": config.learning_schedule,
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "multi_gpu": None,
        "check_for_nan": True,
    }


def train_ppo(
    action_transform: str | Path,
    output_dir: str | Path,
    diffusion_checkpoint: str | Path | None,
    config: PpoTrainConfig | None = None,
) -> Path:
    config = config or PpoTrainConfig()
    task_spec = get_task_spec(config.task)
    config.task = task_spec.name
    if task_spec.name != DEFAULT_TASK:
        if config.reward_audit is None:
            raise ValueError(
                f"{task_spec.name} training requires --reward-audit from expert replay"
            )
        audit = json.loads(Path(config.reward_audit).read_text())
        if audit.get("task") != task_spec.name:
            raise ValueError(
                f"Reward audit is for {audit.get('task')!r}, not {task_spec.name!r}"
            )
        audit_passed = (
            bool(audit.get("passed", False))
            if isinstance(task_spec, TaskSpecV2)
            else bool(
                audit.get("acceptance", {})
                .get(config.task_reward_profile, {})
                .get("passed", False)
            )
        )
        if not audit_passed:
            raise ValueError(
                f"Reward audit did not pass for {task_spec.name}; refusing PPO"
            )
    checkpoint_modes = [config.resume, config.warm_start, config.actor_warm_start]
    if sum(value is not None for value in checkpoint_modes) > 1:
        raise ValueError(
            "resume, warm_start, and actor_warm_start are mutually exclusive"
        )
    torch.manual_seed(config.seed)
    # Match the runner architecture to a warm-start checkpoint automatically.
    # This prevents a recurrent BC policy from being silently instantiated as
    # an MLP merely because a CLI switch was omitted.
    architecture_checkpoint = (
        config.actor_warm_start or config.warm_start or config.resume
    )
    if architecture_checkpoint is not None:
        architecture_data = torch.load(
            architecture_checkpoint, map_location="cpu", weights_only=False
        )
        validate_task_metadata(
            architecture_data,
            task_spec,
            checkpoint=architecture_checkpoint,
            action_transform=action_transform,
        )
        state = architecture_data.get("actor_state_dict", {})
        checkpoint_recurrent = "rnn.rnn.weight_ih_l0" in state
        checkpoint_plan_conditioned = "_plan_conditioned_actor" in state
        config.recurrent_actor = checkpoint_recurrent
        config.plan_conditioned_actor = checkpoint_plan_conditioned
        if checkpoint_recurrent:
            config.rnn_hidden_dim = int(
                state["rnn.rnn.weight_hh_l0"].shape[1]
            )
    output = Path(output_dir)
    if config.plan_conditioned_actor:
        if config.recurrent_actor:
            raise ValueError(
                "plan_conditioned_actor and recurrent_actor are mutually exclusive"
            )
        if config.reference_processed is None:
            raise ValueError(
                "plan_conditioned_actor requires reference_processed"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    curriculum: TrainingCurriculum | None = (
        load_curriculum(config.curriculum_config)
        if config.curriculum_config is not None
        else None
    )
    hard_targets: HardTargetManifest | None = (
        load_hard_target_manifest(config.hard_target_manifest)
        if config.hard_target_manifest is not None
        else None
    )
    if curriculum is not None and any(
        phase.target_mix["hard"] > 0.0 for phase in curriculum.phases
    ) and hard_targets is None:
        raise ValueError(
            "curriculum target_mix hard>0 requires --hard-target-manifest"
        )

    def training_config_for_update(update: int) -> dict | None:
        if curriculum is None:
            return None
        runtime = curriculum.phase_for_update(update).runtime_dict()
        if hard_targets is not None:
            runtime["hard_target_offsets_xy"] = [
                list(target.target_offset_xy) for target in hard_targets.targets
            ]
        return runtime

    initial_update = 0
    if config.resume is not None and architecture_checkpoint is not None:
        initial_update = int(architecture_data.get("iter", 0))
        latest_integrity = (
            architecture_data.get("ppo_integrity", {}).get("latest_record")
        )
        if latest_integrity is not None:
            initial_update = int(latest_integrity["update"]) + 1
    initial_training_config = training_config_for_update(initial_update)
    if curriculum is not None:
        (output / "curriculum_resolved.json").write_text(
            json.dumps(curriculum.metadata(), indent=2)
        )
    env = DistributedGraspVecEnv(
        config.num_envs,
        action_transform,
        device=config.device,
        diffusion_checkpoint=diffusion_checkpoint,
        reward_variant=config.reward_variant,
        ws=config.ws,
        task_reward_weight=config.task_reward_weight,
        smp_reward_weight=config.smp_reward_weight,
        task_reward_profile=config.task_reward_profile,
        seed=config.seed,
        worker_devices=config.worker_devices,
        rsi_dataset=config.rsi_dataset,
        rsi_processed=config.rsi_processed,
        rsi_prefix=config.rsi_prefix,
        rsi_phase=config.rsi_phase,
        rsi_stage=config.rsi_stage,
        rsi_episodes=config.rsi_episodes,
        rsi_probability=config.rsi_probability,
        rsi_scene_hold_episodes=config.rsi_scene_hold_episodes,
        rsi_randomize_target=config.rsi_randomize_target,
        target_position_jitter_xy=config.target_position_jitter_xy,
        target_position_offset_center_xy=(
            config.target_position_offset_center_xy
        ),
        target_yaw_jitter=config.target_yaw_jitter,
        observation_noise=config.observation_noise,
        reference_processed=config.reference_processed,
        reference_source=config.reference_source,
        reference_splits=config.reference_splits,
        reference_reward_weight=config.reference_reward_weight,
        reference_rank_max=config.reference_rank_max,
        reference_base_episode_probability=(
            config.reference_base_episode_probability
        ),
        reference_action_noise_std=config.reference_action_noise_std,
        reference_action_noise_hold_steps=config.reference_action_noise_hold_steps,
        task=task_spec.name,
        training_config=initial_training_config,
    )
    try:
        runner = OnPolicyRunner(env, rsl_config(config), log_dir=str(output), device=config.device)
        base_algorithm_save = runner.alg.save
        integrity_auditor: PpoIntegrityAuditor | None = None

        def save_with_task_metadata():
            payload = base_algorithm_save()
            payload["task_metadata"] = checkpoint_task_metadata(
                task_spec, action_transform
            )
            if integrity_auditor is not None:
                payload["ppo_integrity"] = integrity_auditor.metadata()
            if curriculum is not None:
                payload["curriculum"] = curriculum.metadata()
            if hard_targets is not None:
                payload["hard_targets"] = {
                    "source_evaluation": hard_targets.source_evaluation,
                    "source_split": hard_targets.source_split,
                    "source_seed": hard_targets.source_seed,
                    "count": len(hard_targets.targets),
                }
            return payload

        runner.alg.save = save_with_task_metadata
        if config.actor_learning_rate_scale != 1.0:
            if config.actor_learning_rate_scale <= 0.0:
                raise ValueError("actor_learning_rate_scale must be positive")
            if config.learning_schedule != "fixed":
                raise ValueError(
                    "actor_learning_rate_scale requires learning_schedule='fixed'"
                )
            # Configure the same optimizer parameter groups *before* loading a
            # resume checkpoint.  PPO checkpoints created with decoupled
            # actor/critic rates contain two Adam groups; loading them into the
            # runner's default one-group optimizer fails, while recreating Adam
            # after loading silently discards all moment estimates.
            runner.alg.optimizer = torch.optim.Adam(
                [
                    {
                        "params": runner.alg.get_policy().parameters(),
                        "lr": config.learning_rate
                        * config.actor_learning_rate_scale,
                    },
                    {
                        "params": runner.alg._raw_critic.parameters(),
                        "lr": config.learning_rate,
                    },
                ]
            )
        if config.resume:
            runner.load(config.resume)
            # RSL-RL stores the just-completed loop index in ``iter`` and
            # normally repeats it after loading. Audited checkpoints have an
            # unambiguous policy version, so continue at the next update.
            runner.current_learning_iteration = initial_update
        elif config.warm_start:
            runner.load(
                config.warm_start,
                load_cfg={
                    "actor": True,
                    "critic": True,
                    "optimizer": False,
                    "iteration": False,
                    "rnd": False,
                },
            )
        elif config.actor_warm_start:
            actor_checkpoint = torch.load(
                config.actor_warm_start,
                map_location=config.device,
                weights_only=False,
            )
            runner.alg.get_policy().load_state_dict(
                actor_checkpoint["actor_state_dict"]
            )
            # A BC checkpoint may record the PPO checkpoint it was initialized
            # from. Reuse that critic while keeping the BC actor, so the first
            # long-horizon PPO update does not start from a random value model.
            _load_ancestor_critic(runner, actor_checkpoint, config.device)
        runner.alg.learning_rate = config.learning_rate
        if config.actor_learning_rate_scale != 1.0:
            runner.alg.optimizer.param_groups[0]["lr"] = (
                config.learning_rate * config.actor_learning_rate_scale
            )
            runner.alg.optimizer.param_groups[1]["lr"] = config.learning_rate
        else:
            for param_group in runner.alg.optimizer.param_groups:
                param_group["lr"] = config.learning_rate
        actor = runner.alg.get_policy()
        if config.freeze_actor_normalizer and hasattr(actor.obs_normalizer, "until"):
            actor.obs_normalizer.until = int(actor.obs_normalizer.count.item())
        with torch.no_grad():
            actor.distribution.std_param.fill_(config.action_std)
            if config.manipulation_action_std is not None:
                if config.manipulation_action_std <= 0.0:
                    raise ValueError("manipulation_action_std must be positive")
                actor.distribution.std_param[7:14].fill_(
                    config.manipulation_action_std
                )
                actor.distribution.std_param[21:28].fill_(
                    config.manipulation_action_std
                )
        if config.bc_anchor_weight > 0.0:
            if config.bc_anchor_processed is None:
                raise ValueError("bc_anchor_processed is required when anchoring")
            _install_bc_anchor(
                runner,
                config.bc_anchor_processed,
                config.bc_anchor_sources,
                config.bc_anchor_batch_size,
                config.bc_anchor_weight,
                config.bc_anchor_manipulation_weight,
                config.max_grad_norm,
                config.device,
            )
        if config.teacher_anchor_weight > 0.0:
            if config.teacher_anchor_checkpoint is None:
                raise ValueError(
                    "teacher_anchor_checkpoint is required when anchoring"
                )
            _install_teacher_anchor(
                runner,
                config.teacher_anchor_checkpoint,
                config.teacher_anchor_weight,
                config.device,
                config.task,
                action_transform,
            )
        if not (config.resume or config.warm_start or config.actor_warm_start):
            final_layers = [module for module in actor.mlp.modules() if isinstance(module, nn.Linear)]
            nn.init.zeros_(final_layers[-1].weight)
            nn.init.zeros_(final_layers[-1].bias)
        # This wrapper is deliberately installed after optional BC/teacher
        # anchors so it observes the exact optimizer steps used by the final
        # objective. It aborts rather than producing an unaudited checkpoint.
        integrity_auditor = PpoIntegrityAuditor(
            runner,
            output / "ppo_integrity.jsonl",
            start_update=runner.current_learning_iteration,
            start_total_transitions=(
                int(
                    architecture_data.get("ppo_integrity", {}).get(
                        "total_transitions", 0
                    )
                )
                if config.resume is not None
                and architecture_checkpoint is not None
                else 0
            ),
        )
        if curriculum is not None:
            active_phase = curriculum.phase_for_update(
                runner.current_learning_iteration
            ).name
            audited_update = runner.alg.update

            def update_with_curriculum():
                nonlocal active_phase
                metrics = audited_update()
                phase = curriculum.phase_for_update(integrity_auditor.update)
                if phase.name != active_phase:
                    runtime = training_config_for_update(integrity_auditor.update)
                    assert runtime is not None
                    env.configure_training(runtime)
                    active_phase = phase.name
                return metrics

            runner.alg.update = update_with_curriculum
        # Keep the exact pre-update policy for warm-start regression checks.
        initial_payload = runner.alg.save()
        initial_payload["iter"] = runner.current_learning_iteration
        initial_payload["infos"] = {"stage": "before_first_update"}
        torch.save(initial_payload, output / "model_initial.pt")
        runner.learn(config.iterations)
        checkpoint = output / f"model_{runner.current_learning_iteration}.pt"
        if not checkpoint.exists():
            runner.save(str(checkpoint))
        return checkpoint
    finally:
        env.close()
