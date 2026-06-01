#!/usr/bin/env python3
"""Radical parameter sweep — try everything that hasn't been tested yet.

Approaches:
  A) METAL_HB_SC_SCALE 0→1  (reduce SC H-bonds in metal envs)
  B) ZN r0_O shift 2.00→2.10/2.15/2.20  (move Gaussian well centre)
  C) K reduction 10→5/3/1   (broader Gaussian → less distance penalty)
  D) Combinations of best A + best B/C
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
    cx=sum(a.x for a in ha)/len(ha); cy=sum(a.y for a in ha)/len(ha); cz=sum(a.z for a in ha)/len(ha)
    dx=sum(b.x for b in hb)/len(hb)-cx; dy=sum(b.y for b in hb)/len(hb)-cy; dz=sum(b.z for b in hb)/len(hb)-cz
    return math.sqrt(dx*dx+dy*dy+dz*dz)


def run_target(pdb, lig, metal, params):
    rec = Path(f'{BASE}/pdbqt_receptor/{pdb}_receptor.pdbqt')
    poses_f = Path(f'{BASE}/results_metal/{pdb}/sd32/{lig}_out.pdbqt')
    if not poses_f.exists(): poses_f = Path(f'{BASE}/results_metal/{pdb}/{lig}_out.pdbqt')
    if not poses_f.exists(): poses_f = Path(f'{BASE}/results/{pdb}/{lig}_out.pdbqt')
    crystal_f = Path(f'{BASE}/pdbqt_ligand/{pdb}_{lig}.pdbqt')
    if not rec.exists() or not poses_f.exists(): return None

    poses = rp.split_vina_poses(poses_f)
    if not poses: return None

    lig0 = poses[0].atoms
    prot_atoms = rp.parse_protein(rec, rp.ligand_adaptive_radius(lig0), rp._center_of_mass(lig0), lig_atoms=lig0)
    prot_rings = rp.detect_rings(prot_atoms)

    # Apply parameter overrides
    saved = {}
    new_zn_r0_O = params.get('ZN_R0_O', None)
    new_K = params.get('METAL_GAUSSIAN_K', None)

    if new_zn_r0_O is not None:
        saved['ZN_R0_O'] = rp.METAL_GAUSSIAN_PARAMS['ZN']['O']
        old = rp.METAL_GAUSSIAN_PARAMS['ZN']['O']
        rp.METAL_GAUSSIAN_PARAMS['ZN']['O'] = (new_zn_r0_O, old[1])

    if new_K is not None:
        saved['METAL_GAUSSIAN_K'] = rp.METAL_GAUSSIAN_K
        rp.METAL_GAUSSIAN_K = new_K

    results = []
    for pose in poses:
        pe = rp.compute_pose_energy(pose, prot_atoms, prot_rings)

        # SC HB scale (post-compute, works because e_total is @property)
        hb_sc_scale = params.get('METAL_HB_SC_SCALE', 1.0)
        hb_bb_scale = params.get('METAL_HB_BB_SCALE', 1.0)
        if hb_sc_scale != 1.0 or hb_bb_scale != 1.0:
            has_metal = any(a.element.upper() in rp.METAL_GAUSSIAN_PARAMS for a in prot_atoms)
            if has_metal:
                pe.e_hb_sc *= hb_sc_scale
                pe.e_hb_bb *= hb_bb_scale

        results.append(pe)

    # Restore
    if 'ZN_R0_O' in saved:
        rp.METAL_GAUSSIAN_PARAMS['ZN']['O'] = saved['ZN_R0_O']
    if 'METAL_GAUSSIAN_K' in saved:
        rp.METAL_GAUSSIAN_K = saved['METAL_GAUSSIAN_K']

    if crystal_f.exists():
        from rank_poses import _parse_atom_line
        catoms = []
        with open(crystal_f) as fh:
            for line in fh:
                if line.startswith(('ATOM','HETATM')):
                    a = _parse_atom_line(line)
                    if a: catoms.append(a)
        dists = [centroid_dist_atoms(p.atoms, catoms) for p in poses]
    else:
        dists = [0.0] + [999.0]*(len(poses)-1)

    best_idx = min(range(len(dists)), key=lambda i: dists[i])
    best_pose = poses[best_idx].index
    best_dist = dists[best_idx]
    etr1 = min(results, key=lambda r: r.e_total)
    etr1_dist = dists[etr1.pose_index - 1]
    is_best = (etr1.pose_index == best_pose)
    return best_dist, etr1_dist, is_best, etr1.pose_index, best_pose


def run_all(label, params, verbose=False):
    n_pass = n_match = n_total = 0
    rows = []
    known_fail = {'1G52', '3S3M'}
    for pdb, lig, metal in TARGETS:
        r = run_target(pdb, lig, metal, params)
        if r is None: continue
        best_dist, etr1_dist, is_best, etr1_pose, best_pose = r
        n_total += 1
        etr_pass = etr1_dist < 2.0
        if etr_pass: n_pass += 1
        if is_best: n_match += 1
        rows.append((pdb, best_dist, etr1_dist, is_best, etr_pass, etr1_pose, best_pose))

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  PASS={n_pass}/{n_total}  MATCH={n_match}/{n_total}")
    if verbose:
        print(f"  {'PDB':6}  {'Best':7}  {'ETr=1':7}  {'M':1}  {'P':1}  Note")
        for pdb, bd, ed, is_best, etr_pass, ep, bp in rows:
            note = '' if is_best else f'best=P{bp}({bd:.3f}) etr=P{ep}({ed:.3f})'
            if pdb in known_fail: note = '[structural fail]'
            print(f"  {pdb:6}  {bd:7.3f}  {ed:7.3f}  {'OK' if is_best else '--'}  {'P' if etr_pass else 'F'}  {note}")
    return n_pass, n_match


# ── Baseline ──────────────────────────────────────────────────────────────
run_all("BASELINE", {})

# ── A: METAL_HB_SC_SCALE sweep ────────────────────────────────────────────
for s in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    run_all(f"METAL_HB_SC_SCALE={s}", {'METAL_HB_SC_SCALE': s})

# ── B: ZN r0_O shift ──────────────────────────────────────────────────────
for r0 in [2.05, 2.10, 2.15, 2.20, 2.25]:
    run_all(f"ZN_R0_O={r0}", {'ZN_R0_O': r0})

# ── C: Gaussian K reduction ───────────────────────────────────────────────
for K in [7.0, 5.0, 3.0, 2.0, 1.0]:
    run_all(f"METAL_GAUSSIAN_K={K}", {'METAL_GAUSSIAN_K': K})

# ── D: Best SC scale + best r0 combination ────────────────────────────────
for sc in [0.0, 0.1, 0.2]:
    for r0 in [2.10, 2.15, 2.20]:
        run_all(f"SC_SCALE={sc} + ZN_R0_O={r0}", {'METAL_HB_SC_SCALE': sc, 'ZN_R0_O': r0})

# ── E: SC scale + K ───────────────────────────────────────────────────────
for sc in [0.0, 0.1]:
    for K in [5.0, 3.0]:
        run_all(f"SC_SCALE={sc} + K={K}", {'METAL_HB_SC_SCALE': sc, 'METAL_GAUSSIAN_K': K})

# ── F: SC + r0 + K triple combination ─────────────────────────────────────
for sc in [0.0, 0.1]:
    for r0 in [2.10, 2.20]:
        for K in [5.0, 3.0]:
            run_all(f"SC={sc}+r0={r0}+K={K}", {'METAL_HB_SC_SCALE': sc, 'ZN_R0_O': r0, 'METAL_GAUSSIAN_K': K})

print("\n\nDone.")
