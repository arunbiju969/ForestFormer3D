"""
Extract Full Tree Crowns for Divergent Trees
============================================
Extracts the complete point clouds of:
1. Fast-Only Trees (trees detected by Fast Engine where Baseline called them ground)
2. Baseline-Only Trees (trees detected by Baseline Engine where Fast called them ground)
Preserves original coordinates, tree_id, itc, tree_score, and semantic_pred.
"""

import os
import sys
import json
import laspy
import numpy as np

FAST_LAS = r"D:\LLR DATA\ff3d_output\20250407_LLR_L2_normalized_entire_tract_segmented.las"
BASE_LAS = r"D:\LLR DATA\ff3d_baseline_output\20250407_LLR_L2_normalized_entire_tract_segmented.las"

OUT_FAST_ONLY = r"D:\LLR DATA\ff3d_output\trees_fast_engine_only.las"
OUT_BASE_ONLY = r"D:\LLR DATA\ff3d_output\trees_baseline_engine_only.las"
OUT_COMBINED = r"D:\LLR DATA\ff3d_output\divergent_trees_comparison.las"
REPORT_JSON = r"D:\LLR DATA\ff3d_output\divergent_trees_metadata.json"


def main():
    print(f"[Extractor] Reading Tiles 39-45 point data (from offset 65,134,048)...")
    with laspy.open(FAST_LAS) as f1, laspy.open(BASE_LAS) as f2:
        f1.seek(65_134_048)
        f2.seek(65_134_048)
        c1 = f1.read_points(71_075_337 - 65_134_048)
        c2 = f2.read_points(71_075_337 - 65_134_048)
        hdr1 = f1.header
        hdr2 = f2.header

    t1 = np.array(c1.tree_id)
    t2 = np.array(c2.tree_id)
    x1 = np.array(c1.x)
    y1 = np.array(c1.y)
    z1 = np.array(c1.z)

    trees1 = np.unique(t1[t1 >= 0])
    trees2 = np.unique(t2[t2 >= 0])

    # 1. Match Fast -> Baseline
    fast_only_info = []
    for tid in trees1:
        mask = (t1 == tid)
        base_labels, counts = np.unique(t2[mask], return_counts=True)
        tree_mask = (base_labels >= 0)
        if np.any(tree_mask):
            best_idx = np.argmax(counts[tree_mask])
            best_base_count = counts[tree_mask][best_idx]
            overlap_ratio = best_base_count / len(t2[mask])
        else:
            overlap_ratio = 0.0

        if overlap_ratio < 0.2:  # Less than 20% overlap with any Baseline tree
            gnd_count = counts[base_labels == -1][0] if -1 in base_labels else 0
            cx, cy, cz = float(np.mean(x1[mask])), float(np.mean(y1[mask])), float(np.mean(z1[mask]))
            max_h = float(np.max(z1[mask]))
            min_h = float(np.min(z1[mask]))
            mean_sc = float(np.mean(c1.tree_score[mask]))
            fast_only_info.append({
                'tree_id': int(tid),
                'point_count': int(len(t1[mask])),
                'overlap_with_base_trees': f"{overlap_ratio*100:.1f}%",
                'ground_in_baseline': f"{gnd_count / len(t1[mask]) * 100:.1f}%",
                'centroid': [round(cx, 2), round(cy, 2), round(cz, 2)],
                'height_range': [round(min_h, 2), round(max_h, 2)],
                'height_span': round(max_h - min_h, 2),
                'mean_score': round(mean_sc, 3)
            })

    # 2. Match Baseline -> Fast
    base_only_info = []
    for tid in trees2:
        mask = (t2 == tid)
        fast_labels, counts = np.unique(t1[mask], return_counts=True)
        tree_mask = (fast_labels >= 0)
        if np.any(tree_mask):
            best_idx = np.argmax(counts[tree_mask])
            best_fast_count = counts[tree_mask][best_idx]
            overlap_ratio = best_fast_count / len(t1[mask])
        else:
            overlap_ratio = 0.0

        if overlap_ratio < 0.2:  # Less than 20% overlap with any Fast tree
            gnd_count = counts[fast_labels == -1][0] if -1 in fast_labels else 0
            cx, cy, cz = float(np.mean(x1[mask])), float(np.mean(y1[mask])), float(np.mean(z1[mask]))
            max_h = float(np.max(z1[mask]))
            min_h = float(np.min(z1[mask]))
            mean_sc = float(np.mean(c2.tree_score[mask]))
            base_only_info.append({
                'tree_id': int(tid),
                'point_count': int(len(t2[mask])),
                'overlap_with_fast_trees': f"{overlap_ratio*100:.1f}%",
                'ground_in_fast': f"{gnd_count / len(t2[mask]) * 100:.1f}%",
                'centroid': [round(cx, 2), round(cy, 2), round(cz, 2)],
                'height_range': [round(min_h, 2), round(max_h, 2)],
                'height_span': round(max_h - min_h, 2),
                'mean_score': round(mean_sc, 3)
            })

    print(f"\nFound {len(fast_only_info)} Fast-Only trees and {len(base_only_info)} Baseline-Only trees.")

    # Save Metadata JSON
    with open(REPORT_JSON, "w") as f:
        json.dump({
            'fast_only_trees_count': len(fast_only_info),
            'baseline_only_trees_count': len(base_only_info),
            'net_difference': len(fast_only_info) - len(base_only_info),
            'fast_only_trees': fast_only_info,
            'baseline_only_trees': base_only_info
        }, f, indent=2)
    print(f"[Report] Saved metadata to: {REPORT_JSON}")

    # 3. Extract points for Fast-Only trees
    fast_tids_set = set(t['tree_id'] for t in fast_only_info)
    mask_fast_pts = np.isin(t1, list(fast_tids_set))
    pts_fast_only = c1[mask_fast_pts]
    
    hdr_out1 = laspy.LasHeader(point_format=hdr1.point_format.id, version=hdr1.version)
    hdr_out1.scales = hdr1.scales
    hdr_out1.offsets = hdr1.offsets
    for ed in hdr1.point_format.extra_dimensions:
        hdr_out1.add_extra_dim(laspy.ExtraBytesParams(name=ed.name, type=ed.dtype))
    las_f = laspy.LasData(hdr_out1)
    las_f.points = pts_fast_only
    las_f.write(OUT_FAST_ONLY)
    print(f"[Output] Saved {len(pts_fast_only):,} points for {len(fast_only_info)} Fast-Only trees to:\n  -> {OUT_FAST_ONLY}")

    # 4. Extract points for Baseline-Only trees
    base_tids_set = set(t['tree_id'] for t in base_only_info)
    mask_base_pts = np.isin(t2, list(base_tids_set))
    pts_base_only = c2[mask_base_pts]
    
    hdr_out2 = laspy.LasHeader(point_format=hdr2.point_format.id, version=hdr2.version)
    hdr_out2.scales = hdr2.scales
    hdr_out2.offsets = hdr2.offsets
    for ed in hdr2.point_format.extra_dimensions:
        hdr_out2.add_extra_dim(laspy.ExtraBytesParams(name=ed.name, type=ed.dtype))
    las_b = laspy.LasData(hdr_out2)
    las_b.points = pts_base_only
    las_b.write(OUT_BASE_ONLY)
    print(f"[Output] Saved {len(pts_base_only):,} points for {len(base_only_info)} Baseline-Only trees to:\n  -> {OUT_BASE_ONLY}")

    # 5. Create Combined Side-by-Side Comparison LAS
    # With origin_engine dimension: 1 = Fast-Only Tree, 2 = Baseline-Only Tree
    hdr_comb = laspy.LasHeader(point_format=hdr1.point_format.id, version=hdr1.version)
    hdr_comb.scales = hdr1.scales
    hdr_comb.offsets = hdr1.offsets
    for ed in hdr1.point_format.extra_dimensions:
        hdr_comb.add_extra_dim(laspy.ExtraBytesParams(name=ed.name, type=ed.dtype))
    hdr_comb.add_extra_dim(laspy.ExtraBytesParams(name='origin_engine', type=np.int32))

    n_comb = len(pts_fast_only) + len(pts_base_only)
    comb_las = laspy.LasData(hdr_comb)
    
    for dname in hdr1.point_format.dimension_names:
        comb_las[dname] = np.concatenate([pts_fast_only[dname], pts_base_only[dname]])
        
    for ed in hdr1.point_format.extra_dimensions:
        comb_las[ed.name] = np.concatenate([pts_fast_only[ed.name], pts_base_only[ed.name]])
        
    origin_eng = np.concatenate([
        np.full(len(pts_fast_only), 1, dtype=np.int32),
        np.full(len(pts_base_only), 2, dtype=np.int32)
    ])
    comb_las.origin_engine = origin_eng
    
    # Classification: Class 4 for Fast-Only trees, Class 5 for Baseline-Only trees
    cls_comb = np.concatenate([
        np.full(len(pts_fast_only), 4, dtype=np.uint8),
        np.full(len(pts_base_only), 5, dtype=np.uint8)
    ])
    comb_las.classification = cls_comb
    comb_las.write(OUT_COMBINED)
    print(f"[Output] Saved {n_comb:,} combined points to:\n  -> {OUT_COMBINED}")


if __name__ == "__main__":
    main()
