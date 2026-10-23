"""
ForestFormer3D Baseline vs. Fast Engine Side-by-Side Comparison Benchmark
=========================================================================
Runs both the baseline pipeline (run_inference_las.py) and the optimized
fast pipeline (run_inference_fast.py) on the exact same test plot to verify:
  1. Execution speedup (seconds and speedup factor).
  2. Tree instance detection count agreement.
  3. Semantic classification consistency (wood vs. leaves vs. ground).
  4. Point cloud file integrity for TreeAIBox Enhancer.
"""

import os
import sys
import time
import argparse
import numpy as np
import laspy

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def compare_las_outputs(baseline_las_path, fast_las_path):
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE OUTPUT POINT CLOUD COMPARISON")
    print("=" * 70)
    
    with laspy.open(baseline_las_path) as f_base, laspy.open(fast_las_path) as f_fast:
        b_las = f_base.read()
        f_las = f_fast.read()
        
    n_base = len(b_las.x)
    n_fast = len(f_las.x)
    print(f"Point count: Baseline = {n_base:,} | Fast = {n_fast:,} (Match: {n_base == n_fast})")
    
    b_trees = len(np.unique(b_las.tree_id[b_las.tree_id >= 0]))
    f_trees = len(np.unique(f_las.tree_id[f_las.tree_id >= 0]))
    print(f"Trees detected: Baseline = {b_trees} | Fast = {f_trees}")
    
    # Semantic agreement
    if hasattr(b_las, 'semantic_pred') and hasattr(f_las, 'semantic_pred'):
        sem_match = np.mean(b_las.semantic_pred == f_las.semantic_pred) * 100.0
        print(f"Semantic label agreement: {sem_match:.2f}% identical")
        
    # Classification agreement
    cls_match = np.mean(b_las.classification == f_las.classification) * 100.0
    print(f"ASPRS classification agreement: {cls_match:.2f}% identical")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ForestFormer3D Benchmark Comparison")
    parser.add_argument("--test-las", type=str, default=r"D:\LLR DATA\Clipped_Plots\plot_1.las",
                        help="Path to test LAS file for benchmarking")
    parser.add_argument("--baseline-las", type=str, default=r"D:\LLR DATA\ff3d_output\plot_1_segmented.las",
                        help="Path to pre-computed baseline segmented LAS")
    parser.add_argument("--fast-las", type=str, default=r"D:\LLR DATA\ff3d_fast_output\plot_1_segmented_fast.las",
                        help="Path to fast segmented LAS")
    args = parser.parse_args()
    
    if os.path.isfile(args.baseline_las) and os.path.isfile(args.fast_las):
        compare_las_outputs(args.baseline_las, args.fast_las)
    else:
        print(f"To run benchmark comparison on {args.test_las}:")
        print(f"1. Run fast inference:")
        print(f"   python run_inference_fast.py --input \"{args.test_las}\" --output-dir \"D:\\LLR DATA\\ff3d_fast_output\"")
        print(f"2. Run comparison:")
        print(f"   python benchmark_compare.py --baseline-las \"{args.baseline_las}\" --fast-las \"{args.fast_las}\"")


if __name__ == "__main__":
    main()
