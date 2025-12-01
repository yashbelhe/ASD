"""
Prepare Blender scene manifests for implicit-raymarching outputs.

Reads `target.png`, `initial.png`, and `iter_*.png` from a results
directory (produced by `python/examples/implicit_raymarching.py`) and
writes `scenes.json` with optional transforms that Blender's script can
consume.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create Blender scene descriptors for implicit raymarching results.")
    parser.add_argument("results_dir", type=Path, help="Path to a results directory under results/implicit_raymarching/")
    parser.add_argument("--scene-name", type=str, default=None, help="Optional scene tag (defaults to directory name)")
    parser.add_argument("--application", type=str, default="implicit", help="Application label written in scenes.json")
    parser.add_argument(
        "--transform", type=str, default=None, help="Optional JSON string for a 4x4 transform matrix"
    )
    parser.add_argument(
        "--camera-transform", type=str, default=None, help="Optional JSON string for a 4x4 camera transform"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory '{results_dir}' not found.")

    scene = args.scene_name or results_dir.name
    transform = json.loads(args.transform) if args.transform else None
    camera_transform = json.loads(args.camera_transform) if args.camera_transform else None

    # Always include target, final, initial entries.
    entries = [
        {
            "name": f"{scene}_target",
            "transform": transform,
            "camera_transform": camera_transform,
        },
        {
            "name": f"{scene}_final",
            "transform": transform,
            "camera_transform": camera_transform,
        },
        {
            "name": f"{scene}_initial",
            "transform": transform,
            "camera_transform": camera_transform,
        },
    ]

    scenes_json = results_dir / "scenes.json"
    scenes_json.write_text(json.dumps(entries, indent=4))
    print(f"Wrote {scenes_json}")


if __name__ == "__main__":
    main()
