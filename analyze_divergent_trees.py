import laspy
import numpy as np

p_fast = r"D:\LLR DATA\ff3d_output\20250407_LLR_L2_normalized_entire_tract_segmented.las"
p_base = r"D:\LLR DATA\ff3d_baseline_output\20250407_LLR_L2_normalized_entire_tract_segmented.las"

with laspy.open(p_fast) as f1, laspy.open(p_base) as f2:
    f1.seek(65_134_048)
    f2.seek(65_134_048)
    c1 = f1.read_points(71_075_337 - 65_134_048)
    c2 = f2.read_points(71_075_337 - 65_134_048)

t1_ids = np.array(c1.tree_id)
t2_ids = np.array(c2.tree_id)
x1 = np.array(c1.x)
y1 = np.array(c1.y)
z1 = np.array(c1.z)

trees1 = np.unique(t1_ids[t1_ids >= 0])
trees2 = np.unique(t2_ids[t2_ids >= 0])

print(f"Total trees in Tiles 39-45: Fast={len(trees1)}, Baseline={len(trees2)} (Net Diff: {len(trees1) - len(trees2)})")

unmatched_fast = []
for tid in trees1:
    mask = (t1_ids == tid)
    matched_base = t2_ids[mask]
    gnd_pts = np.count_nonzero(matched_base == -1)
    overlap_ratio = (len(matched_base) - gnd_pts) / len(matched_base)
    if overlap_ratio < 0.5:
        cx, cy, cz = float(np.mean(x1[mask])), float(np.mean(y1[mask])), float(np.mean(z1[mask]))
        max_h = float(np.max(z1[mask]))
        unmatched_fast.append({
            'tree_id': int(tid),
            'pts': int(len(matched_base)),
            'gnd_ratio': round(gnd_pts / len(matched_base) * 100, 1),
            'centroid': (round(cx, 1), round(cy, 1), round(cz, 1)),
            'max_height': round(max_h, 1)
        })

print(f"\nFast Trees mostly unassigned/ground in Baseline: {len(unmatched_fast)}")
for item in unmatched_fast:
    print(f"  Fast Tree {item['tree_id']}: {item['pts']} pts | Max Height: {item['max_height']}m | Centroid: {item['centroid']} | Baseline was {item['gnd_ratio']}% Ground")

unmatched_base = []
for tid in trees2:
    mask = (t2_ids == tid)
    matched_fast = t1_ids[mask]
    gnd_pts = np.count_nonzero(matched_fast == -1)
    overlap_ratio = (len(matched_fast) - gnd_pts) / len(matched_fast)
    if overlap_ratio < 0.5:
        cx, cy, cz = float(np.mean(x1[mask])), float(np.mean(y1[mask])), float(np.mean(z1[mask]))
        max_h = float(np.max(z1[mask]))
        unmatched_base.append({
            'tree_id': int(tid),
            'pts': int(len(matched_fast)),
            'gnd_ratio': round(gnd_pts / len(matched_fast) * 100, 1),
            'centroid': (round(cx, 1), round(cy, 1), round(cz, 1)),
            'max_height': round(max_h, 1)
        })

print(f"\nBaseline Trees mostly unassigned/ground in Fast: {len(unmatched_base)}")
for item in unmatched_base:
    print(f"  Baseline Tree {item['tree_id']}: {item['pts']} pts | Max Height: {item['max_height']}m | Centroid: {item['centroid']} | Fast was {item['gnd_ratio']}% Ground")
