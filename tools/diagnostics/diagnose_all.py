#!/usr/bin/env python3
"""Detailed energy decomposition for ALL non-matching targets.

Shows per-term breakdown: crystal-closest pose vs ETr=1 pose.
Prints gap size and which terms drive ETr=1's advantage.
"""
import sys, math
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


def run_target(pdb, lig, metal):
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

    etr1 = min(results, key=lambda r: r.e_total)
    etr1_dist = dists[etr1.pose_index - 1]

    return {
        'best_pose': best_pose_num,
        'best_dist': best_dist,
        'etr1_pose': etr1.pose_index,
        'etr1_dist': etr1_dist,
        'match': etr1.pose_index == best_pose_num,
        'best_pe': results[best_idx],
        'etr1_pe': etr1,
        'n_poses': len(poses),
    }


TERMS = ['e_hb_bb', 'e_hb_sc', 'e_vdw', 'e_elec', 'e_pipi',
         'e_hphob', 'e_chpi', 'e_cpi', 'e_xbond', 'e_metal', 'e_total']

print(f"\n{'='*100}")
print(f"  Energy decomposition: crystal-closest pose vs ETr=1 pose for all non-matching targets")
print(f"{'='*100}")

matching = []
failing_large = []
failing_small = []

for pdb, lig, metal in TARGETS:
    r = run_target(pdb, lig, metal)
    if r is None:
        print(f"  {pdb}: [no data]")
        continue
    if r['match']:
        matching.append((pdb, r))
        continue

    gap = r['best_pe'].e_total - r['etr1_pe'].e_total  # how much more energy best has vs ETr=1
    # gap > 0 means ETr=1 is more negative (better) by 'gap' kcal/mol

    best = r['best_pe']
    etr = r['etr1_pe']

    print(f"\n  {pdb} ({lig}/{metal}) | best=P{r['best_pose']}({r['best_dist']:.3f}Å) | "
          f"ETr=P{r['etr1_pose']}({r['etr1_dist']:.3f}Å) | gap={gap:.2f} kcal/mol")
    print(f"  {'Term':12}  {'Best(P'+str(r['best_pose'])+')':>12}  {'ETr(P'+str(r['etr1_pose'])+')':>12}  {'Δ(ETr-Best)':>14}  {'Driver?':8}")
    print(f"  {'-'*65}")
    for t in TERMS:
        bv = getattr(best, t)
        ev = getattr(etr, t)
        delta = ev - bv
        driver = '← ETr wins' if delta < -0.3 else ('→ Best wins' if delta > 0.3 else '')
        print(f"  {t:12}  {bv:12.3f}  {ev:12.3f}  {delta:14.3f}  {driver}")

    if gap > 3.0:
        failing_large.append((pdb, gap))
    else:
        failing_small.append((pdb, gap))

print(f"\n\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Matching ({len(matching)}): {', '.join(p for p,_ in matching)}")
print(f"  Gap > 3 kcal/mol ({len(failing_large)}): {', '.join(f'{p}({g:.1f})' for p,g in failing_large)}")
print(f"  Gap ≤ 3 kcal/mol ({len(failing_small)}): {', '.join(f'{p}({g:.1f})' for p,g in failing_small)}")
