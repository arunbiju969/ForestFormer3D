"""
Setup Baseline Full-Tract Resumption
====================================
Clones the verified 38 baseline tiles and checkpoint to 'D:\\LLR DATA\\ff3d_baseline_output'
and runs run_inference_las.py to complete Tiles 39-45 using the original baseline engine.
"""

import os
import sys
import shutil
import json
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FAST_OUTPUT_DIR = r"D:\LLR DATA\ff3d_output"
BASELINE_OUTPUT_DIR = r"D:\LLR DATA\ff3d_baseline_output"
INPUT_LAS = r"D:\LLR DATA\20250407_LLR_L2_normalized.las"

# The 38 tiles completed by the baseline before the stop at tile 39:
BASELINE_38_TILES = [
    "tile_006_x335313_y4982802", "tile_007_x335313_y4982872", "tile_008_x335313_y4982942",
    "tile_009_x335313_y4983012", "tile_010_x335313_y4983082", "tile_014_x335383_y4982662",
    "tile_015_x335383_y4982732", "tile_016_x335383_y4982802", "tile_017_x335383_y4982872",
    "tile_018_x335383_y4982942", "tile_019_x335383_y4983012", "tile_020_x335383_y4983082",
    "tile_023_x335453_y4982592", "tile_024_x335453_y4982662", "tile_025_x335453_y4982732",
    "tile_026_x335453_y4982802", "tile_027_x335453_y4982872", "tile_028_x335453_y4982942",
    "tile_029_x335453_y4983012", "tile_030_x335453_y4983082", "tile_031_x335453_y4983152",
    "tile_033_x335523_y4982522", "tile_034_x335523_y4982592", "tile_035_x335523_y4982662",
    "tile_036_x335523_y4982732", "tile_037_x335523_y4982802", "tile_038_x335523_y4982872",
    "tile_039_x335523_y4982942", "tile_040_x335523_y4983012", "tile_041_x335523_y4983082",
    "tile_042_x335593_y4982522", "tile_043_x335593_y4982592", "tile_044_x335593_y4982662",
    "tile_045_x335593_y4982732", "tile_046_x335593_y4982802", "tile_047_x335593_y4982872",
    "tile_051_x335663_y4982452"
]


def prepare_baseline_directory():
    os.makedirs(BASELINE_OUTPUT_DIR, exist_ok=True)
    baseline_tiles_dir = os.path.join(BASELINE_OUTPUT_DIR, "tiles")
    os.makedirs(baseline_tiles_dir, exist_ok=True)

    src_progress_file = os.path.join(FAST_OUTPUT_DIR, "tract_tiling_progress.json")
    if not os.path.isfile(src_progress_file):
        raise FileNotFoundError(f"Source progress file not found at {src_progress_file}")

    with open(src_progress_file, "r") as f:
        src_progress = json.load(f)

    # Filter to only the 38 tiles completed by baseline prior to Tile 39 (tile_052)
    baseline_progress = {
        "completed_tiles": {},
        "global_tree_offset": 9981
    }

    print(f"[Setup Baseline] Copying 38 verified baseline tile LAS files...")
    for tile_name, tile_info in src_progress["completed_tiles"].items():
        # Exclude tile_052 and onwards
        tile_num = int(tile_name.split("_")[1])
        if tile_num >= 52:
            continue
            
        src_las = tile_info["las_file"]
        dst_las = os.path.join(baseline_tiles_dir, os.path.basename(src_las))
        
        if not os.path.isfile(dst_las) and os.path.isfile(src_las):
            shutil.copy2(src_las, dst_las)
            
        info_copy = dict(tile_info)
        info_copy["las_file"] = dst_las
        baseline_progress["completed_tiles"][tile_name] = info_copy

    dst_progress_file = os.path.join(BASELINE_OUTPUT_DIR, "tract_tiling_progress.json")
    with open(dst_progress_file, "w") as f:
        json.dump(baseline_progress, f, indent=2)

    print(f"[Setup Baseline] Created {dst_progress_file} with {len(baseline_progress['completed_tiles'])} tiles.")
    print(f"[Setup Baseline] Baseline directory ready. Next tile will be Tile 39 (tile_052) with offset 9,981.")


def start_baseline_inference():
    prepare_baseline_directory()
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "run_inference_las.py"),
        "--input", INPUT_LAS,
        "--output-dir", BASELINE_OUTPUT_DIR,
        "--tile-all"
    ]
    print(f"\n[Baseline Runner] Launching baseline engine inference...")
    print(f"[Baseline Runner] Output directory: {BASELINE_OUTPUT_DIR}")
    print(f"[Baseline Runner] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if "--start" in sys.argv:
        start_baseline_inference()
    else:
        prepare_baseline_directory()
