#!/usr/bin/env python3
"""Test parameter changes on all 20 metal_validation targets.

Reports how many ETr=1 = crystal-closest with proposed new parameters.
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


def run_target(pdb, lig, metal, params):
    """Compute (best_dist, etr1_dist, etr1_is_best) for one target with given params."""
    rec = Path(f'{BASE}/pdbqt_receptor/{pdb}_receptor.pdbqt')
    # Priority: results_metal/sd32/ (used for TSV) > results_metal/ > results/
    poses_f = Path(f'{BASE}/results_metal/{pdb}/sd32/{lig}_out.pdbqt')
    if not poses_f.exists():
        poses_f = Path(f'{BASE}/results_metal/{pdb}/{lig}_out.pdbqt')
    if not poses_f.exists():
        poses_f = Path(f'{BASE}/results/{pdb}/{lig}_out.pdbqt')
    crystal_f = Path(f'{BASE}/pdbqt_ligand/{pdb}_{lig}.pdbqt')

    if not rec.exists() or not poses_f.exists():
        return None

    # Load poses
    poses = rp.split_vina_poses(poses_f)
    if not poses:
        return None

    # Load protein
    lig0 = poses[0].atoms
    adapt_r = rp.ligand_adaptive_radius(lig0)
    com = rp._center_of_mass(lig0)
    prot_atoms = rp.parse_protein(rec, adapt_r, com, lig_atoms=lig0)
    prot_rings = rp.detect_rings(prot_atoms)

    # Apply parameter overrides
    saved = {}
    for k, v in params.items():
        if k == 'METAL_GAUSSIAN_PARAMS':
            saved['METAL_GAUSSIAN_PARAMS'] = copy.deepcopy(rp.METAL_GAUSSIAN_PARAMS)
            for elem, donors in v.items():
                rp.METAL_GAUSSIAN_PARAMS[elem] = donors
        elif k == 'METAL_VDW_SCALE':
            saved['METAL_VDW_SCALE'] = rp.METAL_VDW_SCALE
            rp.METAL_VDW_SCALE = v
        elif k == 'METAL_HB_BB_SCALE':
            saved['METAL_HB_BB_SCALE'] = getattr(rp, 'METAL_HB_BB_SCALE', None)
            rp.METAL_HB_BB_SCALE = v
        elif k == 'METAL_HB_SC_SCALE':
            saved['METAL_HB_SC_SCALE'] = getattr(rp, 'METAL_HB_SC_SCALE', None)
            rp.METAL_HB_SC_SCALE = v
        elif k == 'VDW_SR_CAP':
            saved['VDW_SR_CAP'] = rp.VDW_SR_CAP
            rp.VDW_SR_CAP = v

    # Compute energies for all poses
    results = []
    for pose in poses:
        pe = rp.compute_pose_energy(pose, prot_atoms, prot_rings)

        # Apply HB scaling if specified (monkey-patch the result)
        hb_bb_scale = params.get('METAL_HB_BB_SCALE', 1.0)
        hb_sc_scale = params.get('METAL_HB_SC_SCALE', 1.0)
        if hb_bb_scale != 1.0 or hb_sc_scale != 1.0:
            has_metal = any(a.element.upper() in rp.METAL_GAUSSIAN_PARAMS
                           for a in prot_atoms)
            if has_metal:
                pe.e_hb_bb *= hb_bb_scale
                pe.e_hb_sc *= hb_sc_scale

        results.append(pe)

    # Restore params
    for k, v in saved.items():
        if k == 'METAL_GAUSSIAN_PARAMS':
            rp.METAL_GAUSSIAN_PARAMS.clear()
            rp.METAL_GAUSSIAN_PARAMS.update(v)
        elif k == 'METAL_VDW_SCALE':
            rp.METAL_VDW_SCALE = v
        elif k == 'VDW_SR_CAP':
            rp.VDW_SR_CAP = v
        elif k in ('METAL_HB_BB_SCALE', 'METAL_HB_SC_SCALE'):
            if v is None:
                try: delattr(rp, k)
                except: pass
            else:
                setattr(rp, k, v)

    rp.assign_consensus_ranks(results, poses)

    # Crystal distances
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
    best_dist = dists[best_idx]
    best_pose = poses[best_idx].index

    etr1_pe = min(results, key=lambda r: r.e_total)
    etr1_dist = dists[etr1_pe.pose_index - 1]
    is_best = (etr1_pe.pose_index == best_pose)

    return best_dist, etr1_dist, is_best, etr1_pe.pose_index, best_pose


def run_all(label, params):
    """Run all 20 targets and summarize."""
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  {'PDB':6}  {'Best':7}  {'ETr=1':7}  {'Match':6}  {'ETr PASS':9}  Note")
    print(f"  {'-'*65}")

    n_pass_etr = 0
    n_match = 0
    n_total = 0
    known_fail = {'1G52', '3S3M'}  # structurally irrecoverable

    for pdb, lig, metal in TARGETS:
        r = run_target(pdb, lig, metal, params)
        if r is None:
            print(f"  {pdb:6}  [no data]")
            continue
        best_dist, etr1_dist, is_best, etr1_pose, best_pose = r
        n_total += 1

        etr_pass = etr1_dist < 2.0
        if etr_pass: n_pass_etr += 1
        if is_best: n_match += 1

        match_flag = "✓" if is_best else "✗"
        note = ""
        if pdb in known_fail:
            note = "[structural fail]"
        elif not is_best:
            note = f"best=P{best_pose}({best_dist:.3f}Å), ETr=P{etr1_pose}({etr1_dist:.3f}Å)"

        print(f"  {pdb:6}  {best_dist:7.3f}  {etr1_dist:7.3f}  {match_flag:6}  "
              f"{'PASS' if etr_pass else 'FAIL':9}  {note}")

    print(f"\n  ETr=1 PASS (<2Å): {n_pass_etr}/{n_total}")
    print(f"  ETr=1 = crystal-closest: {n_match}/{n_total}")
    return n_pass_etr, n_match, n_total


# ============================================================
# Baseline (current parameters)
# ============================================================
run_all("BASELINE (current params)", {})

# ============================================================
# VDW_SR_CAP sweep: reduce repulsive wall to help crystal-close
# poses that have positive e_vdw in LigandScope's model
# ============================================================
run_all("VDW_SR_CAP=1.15 (softer repulsion)", {'VDW_SR_CAP': 1.15})
run_all("VDW_SR_CAP=1.1  (much softer)", {'VDW_SR_CAP': 1.1})
run_all("VDW_SR_CAP=1.05 (near-zero repulsion)", {'VDW_SR_CAP': 1.05})

# ============================================================
# Combine best VDW cap with ZN well depth increase
# (crystal-close poses get better metal score AND less VDW penalty)
# ============================================================
import copy as _copy
new_zn = {'N': (2.05, 8.0), 'O': (2.00, 6.5), 'S': (2.30, 6.0)}
run_all("VDW_SR_CAP=1.1 + ZN D ↑",
        {'VDW_SR_CAP': 1.1,
         'METAL_GAUSSIAN_PARAMS': {'ZN': new_zn}})

# ============================================================
# METAL_VDW_SCALE: currently 0.30, try relaxing it to 0.5-0.7
# (less damping of VDW in metal sites might help some cases)
# ============================================================
run_all("VDW_SR_CAP=1.1 + METAL_VDW_SCALE=0.5",
        {'VDW_SR_CAP': 1.1, 'METAL_VDW_SCALE': 0.5})
run_all("VDW_SR_CAP=1.1 + METAL_VDW_SCALE=0.7",
        {'VDW_SR_CAP': 1.1, 'METAL_VDW_SCALE': 0.7})

print("\n\nDone.")
