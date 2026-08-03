"""Refresh byte counts and SHA256 values in a tracked mjlab GPU release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple.grasp_rl.mjlab_gpu.release import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()
    root = args.release_dir.resolve()
    manifest_path = root / "release.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        if path.is_symlink():
            raise ValueError(f"release artifacts may not be symlinks: {path}")
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    checkpoint = root / manifest["checkpoint"]
    manifest["checkpoint_sha256"] = sha256_file(checkpoint)
    manifest["artifacts"] = artifacts
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"updated {manifest_path} with {len(artifacts)} artifacts")


if __name__ == "__main__":
    main()
