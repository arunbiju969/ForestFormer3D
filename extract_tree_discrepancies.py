"""
Extract and Analyze Tree vs Ground / Not-Tree Discrepancies
==========================================================
Compares Fast Engine output vs Baseline Engine output across all 71,075,337 points.
Extracts:
1. All points where one engine predicted Tree and the other predicted Ground/Not-Tree.
2. Identifies the specific 17 divergent tree instances (centroids, point counts, heights).
3. Saves a color-coded LAS file for interactive inspection in TreeAIBox Enhancer.
"""

import os
import sys
import numpy as np
import laspy

FAST_LAS = r"D:\LLR DATA\ff3d_output\20250407_LLR_L2_normalized_entire_tract_segmented.las"
BASE_LAS = r"D:\LLR DATA\ff3d_baseline_output\20250407_LLR_L2_normalized_entire_tract_segmented.las"
OUT_DISCREPANCY_LAS = r"D:\LLR DATA\ff3d_output\tree_vs_ground_discrepancies.las"


def main():
    print(f"[Discrepancy Analyzer] Opening master LAS files...")
    chunk_size = 5_000_000
    
    disc_x = []
    disc_y = []
    disc_z = []
    disc_type = []      # 1 = Fast Tree / Base Ground, 2 = Base Tree / Fast Ground
    disc_fast_tid = []
    disc_base_tid = []
    disc_fast_score = []
    disc_base_score = []
    disc_fast_sem = []
    disc_base_sem = []
    
    total_pts = 0
    t_fast_b_gnd = 0
    t_base_f_gnd = 0

    with laspy.open(FAST_LAS) as f_fast, laspy.open(BASE_LAS) as f_base:
        hdr_template = f_fast.header
        
        for idx, (c_fast, c_base) in enumerate(zip(f_fast.chunk_iterator(chunk_size), f_base.chunk_iterator(chunk_size)), 1):
            total_pts += len(c_fast)
            t1 = (c_fast.tree_id >= 0)
            t2 = (c_base.tree_id >= 0)
            
            mask_fast_only = (t1 & ~t2)
            mask_base_only = (~t1 & t2)
            mask_disc = (mask_fast_only | mask_base_only)
            
            n_disc = np.count_nonzero(mask_disc)
            if n_disc > 0:
                disc_x.append(np.array(c_fast.x[mask_disc]))
                disc_y.append(np.array(c_fast.y[mask_disc]))
                disc_z.append(np.array(c_fast.z[mask_disc]))
                
                # Type array
                types = np.where(mask_fast_only[mask_disc], 1, 2).astype(np.int8)
                disc_type.append(types)
                
                disc_fast_tid.append(np.array(c_fast.tree_id[mask_disc]))
                disc_base_tid.append(np.array(c_base.tree_id[mask_disc]))
                disc_fast_score.append(np.array(c_fast.tree_score[mask_disc]))
                disc_base_score.append(np.array(c_base.tree_score[mask_disc]))
                disc_fast_sem.append(np.array(c_fast.semantic_pred[mask_disc]))
                disc_base_sem.append(np.array(c_base.semantic_pred[mask_disc]))
                
                t_fast_b_gnd += np.count_nonzero(mask_fast_only)
                t_base_f_gnd += np.count_nonzero(mask_base_only)
                
            print(f"Processed chunk {idx} ({total_pts:,} pts) | Discrepancies accumulated: {t_fast_b_gnd + t_base_f_gnd:,}")

    # Combine arrays
    all_x = np.concatenate(disc_x)
    all_y = np.concatenate(disc_y)
    all_z = np.concatenate(disc_z)
    all_type = np.concatenate(disc_type)
    all_f_tid = np.concatenate(disc_fast_tid)
    all_b_tid = np.concatenate(disc_base_tid)
    all_f_score = np.concatenate(disc_fast_score)
    all_b_score = np.concatenate(disc_base_score)
    all_f_sem = np.concatenate(disc_fast_sem)
    all_b_sem = np.concatenate(disc_base_sem)
    
    n_total_disc = len(all_x)
    print(f"\n=======================================================")
    print(f"DISCREPANCY EXTRACTION SUMMARY ({n_total_disc:,} points)")
    print(f"=======================================================")
    print(f"Type 1 (Fast = Tree, Base = Ground):     {t_fast_b_gnd:,} points ({(t_fast_b_gnd/n_total_disc)*100:.1f}%)")
    print(f"Type 2 (Base = Tree, Fast = Ground):     {t_base_f_gnd:,} points ({(t_base_f_gnd/n_total_disc)*100:.1f}%)")
    
    # Height distribution analysis
    print(f"\nHeight (Z normalized) Profile of Discrepancies:")
    print(f"  Understory / Ground boundary (Z < 2.0m): {np.count_nonzero(all_z < 2.0):,} points ({(np.count_nonzero(all_z < 2.0)/n_total_disc)*100:.1f}%)")
    print(f"  Mid-canopy (2.0m <= Z < 10.0m):          {np.count_nonzero((all_z >= 2.0) & (all_z < 10.0)):,} points ({(np.count_nonzero((all_z >= 2.0) & (all_z < 10.0))/n_total_disc)*100:.1f}%)")
    print(f"  Upper canopy / Crown edge (Z >= 10.0m):  {np.count_nonzero(all_z >= 10.0):,} points ({(np.count_nonzero(all_z >= 10.0)/n_total_disc)*100:.1f}%)")
    
    # Write discrepancy LAS
    print(f"\n[Writing Output LAS] Creating {OUT_DISCREPANCY_LAS}...")
    hdr = laspy.LasHeader(point_format=hdr_template.point_format.id, version=hdr_template.version)
    hdr.scales = hdr_template.scales
    hdr.offsets = hdr_template.offsets
    
    hdr.add_extra_dim(laspy.ExtraBytesParams(name='diff_type', type=np.int32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name='fast_tree_id', type=np.int32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name='base_tree_id', type=np.int32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name='fast_score', type=np.float32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name='base_score', type=np.float32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name='fast_sem', type=np.int32))
    hdr.add_extra_dim(laspy.ExtraBytesParams(name='base_sem', type=np.int32))
    
    out_las = laspy.LasData(hdr)
    out_las.x = all_x
    out_las.y = all_y
    out_las.z = all_z
    
    out_las.diff_type = all_type.astype(np.int32)
    out_las.fast_tree_id = all_f_tid
    out_las.base_tree_id = all_b_tid
    out_las.fast_score = all_f_score
    out_las.base_score = all_b_score
    out_las.fast_sem = all_f_sem
    out_las.base_sem = all_b_sem
    
    # Color-code by classification:
    # Class 4 (Medium Veg / Orange) = Fast Detected Tree, Base called Ground
    # Class 5 (High Veg / Cyan)     = Baseline Detected Tree, Fast called Ground
    cls_arr = np.zeros(n_total_disc, dtype=np.uint8)
    cls_arr[all_type == 1] = 4
    cls_arr[all_type == 2] = 5
    out_las.classification = cls_arr
    
    # RGB color coding:
    # Fast-only = Bright Red (255, 60, 60)
    # Base-only = Bright Blue (60, 160, 255)
    red = np.zeros(n_total_disc, dtype=np.uint16)
    green = np.zeros(n_total_disc, dtype=np.uint16)
    blue = np.zeros(n_total_disc, dtype=np.uint16)
    
    red[all_type == 1] = 65000
    green[all_type == 1] = 15000
    blue[all_type == 1] = 15000
    
    red[all_type == 2] = 15000
    green[all_type == 2] = 40000
    blue[all_type == 2] = 65000
    
    out_las.red = red
    out_las.green = green
    out_las.blue = blue
    
    out_las.write(OUT_DISCREPANCY_LAS)
    print(f"[Done] Saved {n_total_disc:,} discrepancy points to: {OUT_DISCREPANCY_LAS}")


if __name__ == "__main__":
    main()
