#!/usr/bin/env python3
"""Test consensus rank (vina_rank + etotal_rank) vs pure e_total for best-pose selection.

The idea: Vina's score favors crystal-close poses because it converged to them.
If we combine Vina rank + e_total rank, crystal-close poses that have good Vina
scores may outperform non-crystal poses.

Also tests Vina-weighted selection: best by e_total + w * vina_score.
"""
import sys, math, copy
sys.path.insert(0, '/home/cycheng/LigandScope/scripts')
import rank_poses as rp
from pathlib import Path

BASE = '/home/cycheng/Vina-GPU-2.1/metal_validation'

TARGETS = [
    ('1A42','BZU','ZN'), ('1BNN','AL1','ZN'), ('1G52','F2B','ZN'),
    ('1GKC','NFH','ZN'), ('1JAQ','01S','ZN'), ('1MMQ','RRS','ZN'),
    ('1O86','LPR','ZN'), ('1OQ5','CEL','ZN'), ('1UZE','EAL','ZN'),
    ('1YDB','AZM','ZN'), ('1ZNC','SO4','ZN'), ('2C6N','LPR','ZN'),
    ('2CBD','SO3','ZN'), ('2G1M','4HG','FE'), ('2OVX','4MR','ZN'),
    ('2W0D','CGS','ZN'), ('3HS4','AZM','ZN'), ('3L2U','ELV','MG'),
    ('3P5A','IT2','ZN'), ('3S3M','DLU','MG'),
]


def centroid_dist_atoms(atoms_a, atoms_b):
    ha = [a for a in atoms_a if a.element not in ('H','HD')]
    hb = [a for a in atoms_b if a.element not in ('H','HD')]
    if not ha or not hb: return 999.0
    cx = sum(a.x for a in ha)/len(ha); cy = sum(a.y for a in ha)/len(ha)
    cz = sum(a.z for a in ha)/len(ha)
    dx = sum(b.x for b in hb)/len(hb) - cx
    dy = sum(b.y for b in hb)/len(hb) - cy
    dz = sum(b.z for b in hb)/len(hb) - cz
    return math.sqrt(dx*dx+dy*dy+dz*dz)


def run_target(pdb, lig, metal, selector='etotal'):
    """selector: 'etotal' | 'consensus' | float (weight for vina_score in e_total + w*vina)"""
    rec = Path(f'{BASE}/pdbqt_receptor/{pdb}_receptor.pdbqt')
    poses_f = Path(f'{BASE}/results_metal/{pdb}/sd32/{lig}_out.pdbqt')
    if not poses_f.exists():
        poses_f = Path(f'{BASE}/results_metal/{pdb}/{lig}_out.pdbqt')
    if not poses_f.exists():
        poses_f = Path(f'{BASE}/results/{pdb}/{lig}_out.pdbqt')
    crystal_f = Path(f'{BASE}/pdbqt_ligand/{pdb}_{lig}.pdbqt')

    if not rec.exists() or not poses_f.exists():
        return None

    poses = rp.split_vina_poses(poses_f)
    if not poses:
        return None

    lig0 = poses[0].atoms
    adapt_r = rp.ligand_adaptive_radius(lig0)
    com = rp._center_of_mass(lig0)
    prot_atoms = rp.parse_protein(rec, adapt_r, com, lig_atoms=lig0)
    prot_rings = rp.detect_rings(prot_atoms)

    results = [rp.compute_pose_energy(p, prot_atoms, prot_rings) for p in poses]
    rp.assign_consensus_ranks(results, poses)

    if crystal_f.exists():
        from rank_poses import _parse_atom_line
        catoms = []
        with open(crystal_f) as fh:
            for line in fh:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    a = _parse_atom_line(line)
                    if a: catoms.append(a)
        dists = [centroid_dist_atoms(p.atoms, catoms) for p in poses]
    else:
        dists = [0.0] + [999.0] * (len(poses) - 1)

    best_idx = min(range(len(dists)), key=lambda i: dists[i])
    best_pose_num = poses[best_idx].index
    best_dist = dists[best_idx]

    # Select ETr=1 based on selector
    if selector == 'etotal':
        etr1 = min(results, key=lambda r: r.e_total)
    elif selector == 'consensus':
        etr1 = min(results, key=lambda r: r.consensus_rank)
    elif isinstance(selector, (int, float)):
        w = float(selector)
        etr1 = min(results, key=lambda r: r.e_total + w * r.vina_score)
    else:
        etr1 = min(results, key=lambda r: r.e_total)

    etr1_dist = dists[etr1.pose_index - 1]

    return {
        'best_pose': best_pose_num,
        'best_dist': best_dist,
        'etr1_pose': etr1.pose_index,
        'etr1_dist': etr1_dist,
        'match': etr1.pose_index == best_pose_num,
        'etr1_pass': etr1_dist < 2.0,
    }


def run_all(label, selector='etotal'):
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  {'PDB':6}  {'Best':7}  {'ETr=1':7}  {'Match':6}  {'ETr PASS':9}  Note")
    print(f"  {'-'*65}")

    n_pass = n_match = n_total = 0
    known_fail = {'1G52', '3S3M'}

    for pdb, lig, metal in TARGETS:
        r = run_target(pdb, lig, metal, selector)
        if r is None:
            print(f"  {pdb:6}  [no data]")
            continue
        n_total += 1
        if r['etr1_pass']: n_pass += 1
        if r['match']: n_match += 1

        note = ""
        if pdb in known_fail:
            note = "[structural fail]"
        elif not r['match']:
            note = f"best=P{r['best_pose']}({r['best_dist']:.3f}Å), ETr=P{r['etr1_pose']}({r['etr1_dist']:.3f}Å)"

        print(f"  {pdb:6}  {r['best_dist']:7.3f}  {r['etr1_dist']:7.3f}  "
              f"{'✓' if r['match'] else '✗':6}  "
              f"{'PASS' if r['etr1_pass'] else 'FAIL':9}  {note}")

    print(f"\n  ETr=1 PASS (<2Å): {n_pass}/{n_total}")
    print(f"  ETr=1 = crystal-closest: {n_match}/{n_total}")
    return n_pass, n_match, n_total


run_all("BASELINE: min(e_total)", selector='etotal')
run_all("CONSENSUS: min(vina_rank + etotal_rank)", selector='consensus')
run_all("WEIGHTED w=0.1: min(e_total + 0.1×vina)", selector=0.1)
run_all("WEIGHTED w=0.2: min(e_total + 0.2×vina)", selector=0.2)
run_all("WEIGHTED w=0.3: min(e_total + 0.3×vina)", selector=0.3)
run_all("WEIGHTED w=0.5: min(e_total + 0.5×vina)", selector=0.5)

print("\n\nDone.")
