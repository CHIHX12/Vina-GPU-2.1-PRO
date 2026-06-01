#!/usr/bin/env python3
"""Break e_metal into: Gaussian well, bidentate bonus, geometry bonus.
Also show for 1GKC specifically: all per-metal per-donor Gaussian scores.
"""
import sys, math
sys.path.insert(0, '/home/cycheng/LigandScope/scripts')
import rank_poses as rp
from pathlib import Path

BASE = '/home/cycheng/Vina-GPU-2.1/metal_validation'

# Most interesting cases for metal decomposition
TARGETS = [
    ('1GKC','NFH','ZN'),   # gap=1.64 metal drives
    ('1YDB','AZM','ZN'),   # gap=0.79 vdw+metal
    ('1BNN','AL1','ZN'),   # gap=2.04 hb_sc drives
    ('2W0D','CGS','ZN'),   # gap=2.42 all terms
    ('2C6N','LPR','ZN'),   # gap=4.22 hb_sc drives
]

def centroid_dist_atoms(atoms_a, atoms_b):
    ha = [a for a in atoms_a if a.element not in ('H','HD')]
    hb = [a for a in atoms_b if a.element not in ('H','HD')]
    if not ha or not hb: return 999.0
    cx = sum(a.x for a in ha)/len(ha); cy = sum(a.y for a in ha)/len(ha); cz = sum(a.z for a in ha)/len(ha)
    dx = sum(b.x for b in hb)/len(hb)-cx; dy = sum(b.y for b in hb)/len(hb)-cy; dz = sum(b.z for b in hb)/len(hb)-cz
    return math.sqrt(dx*dx+dy*dy+dz*dz)


def get_metal_components(pose, prot_atoms):
    """Return (e_gauss_primary, e_bidentate, e_geom) separately."""
    donors = rp._get_lig_donors(pose.atoms) if hasattr(rp, '_get_lig_donors') else \
             [a for a in pose.atoms if a.element.upper() in rp.METAL_DONOR_ELEMENTS and a.element not in ('H','HD')]

    # Use the internal functions via monkey inspection
    # The total e_metal = _fast_metal_total + _metal_bidentate_bonus + _metal_geom_energy
    # We need to call these directly
    metals = [a for a in prot_atoms if a.element.upper() in rp.METAL_GAUSSIAN_PARAMS]

    # Per-metal, per-donor Gaussian scores
    detail = {}
    for ma in metals:
        params = rp.METAL_GAUSSIAN_PARAMS.get(ma.element.upper(), {})
        per_donor = []
        for la in donors:
            dx = la.x-ma.x; dy = la.y-ma.y; dz = la.z-ma.z
            r = math.sqrt(dx*dx+dy*dy+dz*dz)
            if r > rp.METAL_COORD_RMAX: continue
            dcat = rp._donor_cat(la.element)
            if dcat not in params: continue
            r0, D = params[dcat]
            e = -D * math.exp(-rp.METAL_GAUSSIAN_K * (r-r0)**2)
            per_donor.append((la, r, e, dcat))
        if per_donor:
            per_donor.sort(key=lambda x: x[2])  # most negative first
            detail[id(ma)] = (ma, per_donor)

    return detail, metals, donors  # detail: {id(ma): (ma, per_donor)}


def run_target(pdb, lig, metal):
    rec = Path(f'{BASE}/pdbqt_receptor/{pdb}_receptor.pdbqt')
    poses_f = Path(f'{BASE}/results_metal/{pdb}/sd32/{lig}_out.pdbqt')
    if not poses_f.exists(): poses_f = Path(f'{BASE}/results_metal/{pdb}/{lig}_out.pdbqt')
    if not poses_f.exists(): poses_f = Path(f'{BASE}/results/{pdb}/{lig}_out.pdbqt')
    crystal_f = Path(f'{BASE}/pdbqt_ligand/{pdb}_{lig}.pdbqt')

    if not rec.exists() or not poses_f.exists(): return
    poses = rp.split_vina_poses(poses_f)
    if not poses: return

    lig0 = poses[0].atoms
    prot_atoms = rp.parse_protein(rec, rp.ligand_adaptive_radius(lig0), rp._center_of_mass(lig0), lig_atoms=lig0)
    prot_rings = rp.detect_rings(prot_atoms)
    results = [rp.compute_pose_energy(p, prot_atoms, prot_rings) for p in poses]

    catoms = []
    if crystal_f.exists():
        from rank_poses import _parse_atom_line
        with open(crystal_f) as fh:
            for line in fh:
                if line.startswith(('ATOM','HETATM')):
                    a = _parse_atom_line(line)
                    if a: catoms.append(a)
    dists = [centroid_dist_atoms(p.atoms, catoms) if catoms else 999.0 for p in poses]
    if not catoms: dists[0] = 0.0

    best_idx = min(range(len(dists)), key=lambda i: dists[i])
    best_pose = poses[best_idx]; best_res = results[best_idx]; best_dist = dists[best_idx]
    etr1_res = min(results, key=lambda r: r.e_total)
    etr1_pose = poses[etr1_res.pose_index-1]; etr1_dist = dists[etr1_res.pose_index-1]

    print(f"\n{'='*70}")
    print(f"  {pdb} ({lig}/{metal})  best=P{best_pose.index}({best_dist:.3f}Å)  ETr=P{etr1_pose.index}({etr1_dist:.3f}Å)")
    print(f"  gap={etr1_res.e_total - best_res.e_total:+.2f} kcal/mol | e_metal best={best_res.e_metal:.3f} ETr={etr1_res.e_metal:.3f}")

    for label, pose, res, dist in [('Best(crystal-close)', best_pose, best_res, best_dist),
                                    ('ETr=1', etr1_pose, etr1_res, etr1_dist)]:
        print(f"\n  [{label}] P{pose.index} ({dist:.3f}Å)  e_metal={res.e_metal:.3f}  e_total={res.e_total:.3f}")
        detail, metals, donors = get_metal_components(pose, prot_atoms)

        for key, (ma, per_donor) in detail.items():
            primary_e = per_donor[0][2] if per_donor else 0.0
            all_e_sum = sum(x[2] for x in per_donor)
            print(f"    Metal {ma.element}:")
            for la, r, e, dcat in per_donor[:6]:
                name = getattr(la, 'name', la.element)
                print(f"      {name:6}({dcat}) r={r:.3f}  e={e:.3f}")
            print(f"      PRIMARY={primary_e:.3f}  SUM_ALL={all_e_sum:.3f}  donors_in_range={len(per_donor)}")

        # Bidentate bonus: call the function
        e_bi = rp._metal_bidentate_bonus(prot_atoms, donors)
        print(f"    e_bidentate={e_bi:.3f}")

        # Geometry bonus
        e_geom = rp._metal_geom_energy(prot_atoms, donors)
        print(f"    e_geom={e_geom:.3f}")

        # Expected primary sum
        primary_sum = sum(v[1][0][2] for v in detail.values() if v[1])
        print(f"    e_gauss_primary_sum={primary_sum:.3f}")


# Check if _get_lig_donors exists in rp
try:
    _ = rp._get_lig_donors
except AttributeError:
    pass

for pdb, lig, metal in TARGETS:
    run_target(pdb, lig, metal)

print("\n\nDone.")
