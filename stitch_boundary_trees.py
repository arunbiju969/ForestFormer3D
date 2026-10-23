"""
Boundary Tree Stitcher & Seam Reconnector
=========================================
Resolves tile boundary tree splitting in whole-tract segmented point clouds:
1. Scans all internal X and Y grid seams (70m grid intervals).
2. Uses spatial KD-Trees to detect adjacent tree halves touching across the seam (< 35cm gap).
3. Verifies vertical crown alignment (height consistency).
4. Employs Union-Find (Disjoint Set) to unify split halves into single contiguous tree crowns.
5. Writes the final stitched LAS file and reports exact merged tree count.
"""

import os
import sys
import time
import argparse
import laspy
import numpy as np
from scipy.spatial import cKDTree


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Keep smaller ID as canonical root for consistency
            if root_i < root_j:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j


def detect_seam_pairs(pts_x, pts_y, pts_z, pts_tid, seam_val, is_x_seam, max_gap=0.35, max_z_diff=3.0):
    """Detects matching tree halves across a linear seam plane."""
    if is_x_seam:
        coord_split = pts_x
        tangent_coord = pts_y
    else:
        coord_split = pts_y
        tangent_coord = pts_x

    side_a_mask = (coord_split < seam_val)
    side_b_mask = (coord_split >= seam_val)

    if not np.any(side_a_mask) or not np.any(side_b_mask):
        return []

    coords_a = np.column_stack([pts_x[side_a_mask], pts_y[side_a_mask], pts_z[side_a_mask]])
    coords_b = np.column_stack([pts_x[side_b_mask], pts_y[side_b_mask], pts_z[side_b_mask]])
    tids_a = pts_tid[side_a_mask]
    tids_b = pts_tid[side_b_mask]

    # Build KDTree on Side B
    kdtree_b = cKDTree(coords_b)
    
    # Query all points on Side A against Side B
    dists, indices = kdtree_b.query(coords_a, distance_upper_bound=max_gap, workers=-1)
    
    valid_hits = (dists < max_gap)
    if not np.any(valid_hits):
        return []

    matched_a = tids_a[valid_hits]
    matched_b = tids_b[indices[valid_hits]]
    hit_dists = dists[valid_hits]

    # Precompute max heights for validation
    unique_a = np.unique(matched_a)
    unique_b = np.unique(matched_b)
    z_max_a = {tid: float(np.max(pts_z[pts_tid == tid])) for tid in unique_a}
    z_max_b = {tid: float(np.max(pts_z[pts_tid == tid])) for tid in unique_b}

    matched_pairs = {}
    for tid_a, tid_b, d in zip(matched_a, matched_b, hit_dists):
        if tid_a == tid_b:
            continue
        pair = (min(tid_a, tid_b), max(tid_a, tid_b))
        
        # Height alignment check
        if abs(z_max_a[tid_a] - z_max_b[tid_b]) <= max_z_diff:
            if pair not in matched_pairs or d < matched_pairs[pair]:
                matched_pairs[pair] = d

    return [(k[0], k[1], v) for k, v in matched_pairs.items()]


def stitch_tract(input_las, output_las, tile_size=70.0, max_gap=0.35):
    t0 = time.time()
    print(f"\n=======================================================")
    print(f"BOUNDARY TREE STITCHER & SEAM RECONNECTOR")
    print(f"Input:  {input_las}")
    print(f"Output: {output_las}")
    print(f"=======================================================")

    with laspy.open(input_las) as f:
        hdr = f.header
        min_x = hdr.x_min
        max_x = hdr.x_max
        min_y = hdr.y_min
        max_y = hdr.y_max
        total_pts = hdr.point_count

    # Internal seams
    xs = np.arange(min_x, max_x, tile_size)[1:]
    ys = np.arange(min_y, max_y, tile_size)[1:]
    print(f"[Grid Seams] Found {len(xs)} internal X-seams and {len(ys)} internal Y-seams.")

    # 1. Collect points in boundary corridors
    band = 1.0  # +/- 1.0m strip along each seam
    print(f"[Corridor Extraction] Extracting tree points along seam corridors (+/- {band}m)...")
    
    corridor_x = []
    corridor_y = []
    corridor_z = []
    corridor_tid = []

    with laspy.open(input_las) as f:
        for chunk in f.chunk_iterator(10_000_000):
            t_mask = (chunk.tree_id >= 0)
            if not np.any(t_mask):
                continue
                
            cx = chunk.x[t_mask]
            cy = chunk.y[t_mask]
            cz = chunk.z[t_mask]
            ctid = chunk.tree_id[t_mask]

            # Check distance to any X seam or Y seam
            dx_min = np.min(np.abs(cx[:, None] - xs[None, :]), axis=1) if len(xs) > 0 else np.full(len(cx), 999.0)
            dy_min = np.min(np.abs(cy[:, None] - ys[None, :]), axis=1) if len(ys) > 0 else np.full(len(cy), 999.0)

            seam_pts_mask = (dx_min <= band) | (dy_min <= band)
            if np.any(seam_pts_mask):
                corridor_x.append(cx[seam_pts_mask])
                corridor_y.append(cy[seam_pts_mask])
                corridor_z.append(cz[seam_pts_mask])
                corridor_tid.append(ctid[seam_pts_mask])

    bx = np.concatenate(corridor_x)
    by = np.concatenate(corridor_y)
    bz = np.concatenate(corridor_z)
    btid = np.concatenate(corridor_tid)
    print(f"[Corridor Extraction] Loaded {len(bx):,} boundary corridor points in {time.time() - t0:.1f}s.")

    # 2. Detect matching tree halves across every seam
    uf = UnionFind()
    total_raw_pairs = 0

    print(f"\n[Seam Analysis] Scanning X-seams for severed crowns...")
    for seam in xs:
        mask = (np.abs(bx - seam) <= band)
        if np.any(mask):
            pairs = detect_seam_pairs(bx[mask], by[mask], bz[mask], btid[mask], seam, is_x_seam=True, max_gap=max_gap)
            for t1, t2, d in pairs:
                uf.union(t1, t2)
            total_raw_pairs += len(pairs)
            if pairs:
                print(f"  X-Seam {seam:.1f}: matched {len(pairs)} severed tree pairs (min gap: {min(p[2] for p in pairs)*100:.1f} cm)")

    print(f"\n[Seam Analysis] Scanning Y-seams for severed crowns...")
    for seam in ys:
        mask = (np.abs(by - seam) <= band)
        if np.any(mask):
            pairs = detect_seam_pairs(bx[mask], by[mask], bz[mask], btid[mask], seam, is_x_seam=False, max_gap=max_gap)
            for t1, t2, d in pairs:
                uf.union(t1, t2)
            total_raw_pairs += len(pairs)
            if pairs:
                print(f"  Y-Seam {seam:.1f}: matched {len(pairs)} severed tree pairs (min gap: {min(p[2] for p in pairs)*100:.1f} cm)")

    # Build remapping dictionary
    remap = {}
    for child in list(uf.parent.keys()):
        root = uf.find(child)
        if child != root:
            remap[child] = root

    print(f"\n=======================================================")
    print(f"SEAM RECONNECTION RESULTS")
    print(f"=======================================================")
    print(f"Total Seam Contact Matches:     {total_raw_pairs} boundary connections")
    print(f"Total Split Tree Halves Merged: {len(remap)} trees reconnected into single crowns")

    # 3. Stream write updated LAS with unified tree IDs
    print(f"\n[Writing Output] Streaming stitched tract to: {output_las}...")
    hdr_out = laspy.LasHeader(point_format=hdr.point_format.id, version=hdr.version)
    hdr_out.scales = hdr.scales
    hdr_out.offsets = hdr.offsets
    for ed in hdr.point_format.extra_dimensions:
        hdr_out.add_extra_dim(laspy.ExtraBytesParams(name=ed.name, type=ed.dtype))

    # Fast lookup array for remapping
    max_id = max(max(remap.keys(), default=0), max(remap.values(), default=0)) + 1
    lookup_table = np.arange(max_id, dtype=np.int32)
    for old_id, new_id in remap.items():
        lookup_table[old_id] = new_id

    total_updated_pts = 0
    t_write = time.time()
    
    with laspy.open(output_las, mode='w', header=hdr_out) as writer:
        with laspy.open(input_las) as reader:
            for chunk_idx, chunk in enumerate(reader.chunk_iterator(10_000_000), 1):
                t_mask = (chunk.tree_id >= 0) & (chunk.tree_id < max_id)
                if np.any(t_mask):
                    orig_ids = chunk.tree_id[t_mask]
                    new_ids = lookup_table[orig_ids]
                    
                    changed = (new_ids != orig_ids)
                    total_updated_pts += np.count_nonzero(changed)
                    
                    chunk.tree_id[t_mask] = new_ids
                    # Synchronize itc (itc = tree_id + 1)
                    chunk.itc[t_mask] = new_ids + 1

                writer.write_points(chunk)
                print(f"  Wrote chunk {chunk_idx} ({total_pts:,} total) | Boundary points reconnected so far: {total_updated_pts:,}")

    elapsed = time.time() - t0
    print(f"\n[Complete] Successfully stitched {total_updated_pts:,} points across {len(remap)} severed trees in {elapsed:.1f}s!")
    return output_las, len(remap), total_updated_pts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch boundary-split trees across tile seams")
    parser.add_argument("--input", "-i", type=str, default=r"D:\LLR DATA\ff3d_output\20250407_LLR_L2_normalized_entire_tract_segmented.las")
    parser.add_argument("--output", "-o", type=str, default=r"D:\LLR DATA\ff3d_output\20250407_LLR_L2_normalized_entire_tract_stitched.las")
    parser.add_argument("--max-gap", type=float, default=0.35, help="Maximum gap across seam in meters (default: 0.35m)")
    args = parser.parse_args()

    stitch_tract(args.input, args.output, max_gap=args.max_gap)
