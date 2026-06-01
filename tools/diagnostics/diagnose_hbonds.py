#!/usr/bin/env python3
"""Show per-atom H-bond counts for crystal-close vs ETr poses in key failing targets.
This reveals whether H-bond saturation (per-atom cap) could help.
"""
import sys, math
sys.path.insert(0, '/home/cycheng/LigandScope/scripts')
import rank_poses as rp
from pathlib import Path

BASE = '/home/cycheng/Vina-GPU-2.1/metal_validation'

# Focus on e_hb_sc-dominated failures and the UZE false-minimum issue
TARGETS = [
    ('2C6N','LPR','ZN'),   # gap=4.22, hb_sc drives; P10 is false minimum
    ('1BNN','AL1','ZN'),   # gap=2.04, hb_sc drives
    ('1A42','BZU','ZN'),   # gap=3.0, need to check
    ('1UZE','EAL','ZN'),   # gap=0.70, hb_sc drives; P8/P9 is false minimum
    ('2OVX','4MR','ZN'),   # gap=4.5, hb_sc drives
    ('1OQ5','CEL','ZN'),   # gap=3.3
]

def cdist(a1, a2):
    ha = [a for a in a1 if a.element not in ('H','HD')]
    hb = [a for a in a2 if a.element not in ('H','HD')]
    if not ha or not hb: return 999.0
    cx,cy,cz=sum(a.x for a in ha)/len(ha),sum(a.y for a in ha)/len(ha),sum(a.z for a in ha)/len(ha)
    dx,dy,dz=sum(b.x for b in hb)/len(hb)-cx,sum(b.y for b in hb)/len(hb)-cy,sum(b.z for b in hb)/len(hb)-cz
    return math.sqrt(dx*dx+dy*dy+dz*dz)


def hbond_per_atom_detail(pose, prot_atoms, prot_rings):
    """Compute H-bonds per ligand atom (rough estimate via re-scoring)."""
    # We need to look inside the HB calculation. Since we can't easily
    # extract per-atom HB, we'll use a proxy: run with different atom subsets.
    # Instead, let's just compute total e_hb_sc and count unique HB pairs.

    # Parse: run compute_pose_energy and get total
    pe = rp.compute_pose_energy(pose, prot_atoms, prot_rings)
    return pe


for pdb, lig, metal in TARGETS:
    rec = Path(f'{BASE}/pdbqt_receptor/{pdb}_receptor.pdbqt')
    pf = Path(f'{BASE}/results_metal/{pdb}/sd32/{lig}_out.pdbqt')
    if not pf.exists(): pf = Path(f'{BASE}/results_metal/{pdb}/{lig}_out.pdbqt')
    if not pf.exists(): pf = Path(f'{BASE}/results/{pdb}/{lig}_out.pdbqt')
    cf = Path(f'{BASE}/pdbqt_ligand/{pdb}_{lig}.pdbqt')
    if not rec.exists() or not pf.exists(): continue

    poses = rp.split_vina_poses(pf)
    if not poses: continue
    l0 = poses[0].atoms
    pa = rp.parse_protein(rec, rp.ligand_adaptive_radius(l0), rp._center_of_mass(l0), lig_atoms=l0)
    pr = rp.detect_rings(pa)
    results = [rp.compute_pose_energy(p, pa, pr) for p in poses]

    if cf.exists():
        from rank_poses import _parse_atom_line
        catoms = []
        with open(cf) as fh:
            for ln in fh:
                if ln.startswith(('ATOM','HETATM')):
                    a = _parse_atom_line(ln)
                    if a: catoms.append(a)
        dists = [cdist(p.atoms, catoms) for p in poses]
    else:
        dists = [0.0] + [999.0]*(len(poses)-1)

    bi = min(range(len(dists)), key=lambda i: dists[i])
    best_pose = poses[bi]
    best_res = results[bi]
    best_dist = dists[bi]
    etr1_res = min(results, key=lambda r: r.e_total)
    etr1_pose = poses[etr1_res.pose_index - 1]
    etr1_dist = dists[etr1_res.pose_index - 1]

    print(f"\n{'='*65}")
    print(f"  {pdb} ({lig}/{metal})")
    print(f"  Crystal-close: P{best_pose.index}({best_dist:.3f}Å) "
          f"e_hb_sc={best_res.e_hb_sc:.3f} e_total={best_res.e_total:.3f}")
    print(f"  ETr=1:         P{etr1_pose.index}({etr1_dist:.3f}Å) "
          f"e_hb_sc={etr1_res.e_hb_sc:.3f} e_total={etr1_res.e_total:.3f}")
    print(f"  Gap: {etr1_res.e_total - best_res.e_total:+.2f} kcal/mol")

    # Show ALL poses sorted by distance — e_hb_sc and e_total
    print(f"\n  All poses (by centroid distance):")
    print(f"  {'P':3} {'Dist':7} {'e_hb_sc':9} {'e_vdw':7} {'e_metal':8} {'e_total':8}")
    pose_data = sorted(zip(poses, results, dists), key=lambda x: x[2])
    for p, r, d in pose_data[:10]:
        mark = '<-- crystal' if p.index == best_pose.index else ('*** ETr' if p.index == etr1_pose.index else '')
        print(f"  P{p.index:<2} {d:7.3f}  {r.e_hb_sc:9.3f} {r.e_vdw:7.3f} {r.e_metal:8.3f} {r.e_total:8.3f}  {mark}")

print("\n\nDone.")
