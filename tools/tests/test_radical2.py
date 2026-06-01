#!/usr/bin/env python3
"""Second wave of radical tests:
  - Simple score: only e_metal + e_hb_bb + e_vdw
  - Metal heavy: e_metal×2 + rest
  - No hphob in metal
  - Positive VDW hard-cap to 0 (no repulsion penalty)
  - e_hb_sc partial + VDW positive cap
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


def cdist(a1, a2):
    ha = [a for a in a1 if a.element not in ('H','HD')]
    hb = [a for a in a2 if a.element not in ('H','HD')]
    if not ha or not hb: return 999.0
    cx,cy,cz=sum(a.x for a in ha)/len(ha),sum(a.y for a in ha)/len(ha),sum(a.z for a in ha)/len(ha)
    dx,dy,dz=sum(b.x for b in hb)/len(hb)-cx,sum(b.y for b in hb)/len(hb)-cy,sum(b.z for b in hb)/len(hb)-cz
    return math.sqrt(dx*dx+dy*dy+dz*dz)


def custom_score(pe, mode, has_metal):
    """Return a custom score for selection."""
    if mode == 'simple':
        # Only metal + backbone HB + VDW
        return pe.e_metal + pe.e_hb_bb + pe.e_vdw
    elif mode == 'simple_no_vdw_pos':
        # Metal + backbone HB + min(VDW, 0) — no repulsion penalty
        return pe.e_metal + pe.e_hb_bb + min(pe.e_vdw, 0.0)
    elif mode == 'metal_heavy':
        # Metal × 2 + everything else
        return pe.e_total + pe.e_metal  # adds e_metal one more time
    elif mode == 'metal_heavy3':
        return pe.e_total + 2.0 * pe.e_metal  # total w/ e_metal×3
    elif mode == 'no_vdw_pos':
        # Cap positive VDW to 0 (no repulsion penalty) — affects e_total via property
        vdw_adj = min(pe.e_vdw, 0.0) - pe.e_vdw  # negative adjustment
        return pe.e_total + vdw_adj
    elif isinstance(mode, tuple) and mode[0] == 'sc_scale_novdwpos':
        sc_scale = mode[1]
        vdw_adj = min(pe.e_vdw, 0.0) - pe.e_vdw
        if has_metal:
            sc_adj = pe.e_hb_sc * (sc_scale - 1.0)
        else:
            sc_adj = 0.0
        return pe.e_total + vdw_adj + sc_adj
    elif isinstance(mode, tuple) and mode[0] == 'sc_scale_vdwscale':
        sc_scale, vdw_scale = mode[1], mode[2]
        if has_metal:
            sc_adj = pe.e_hb_sc * (sc_scale - 1.0)
            vdw_adj = pe.e_vdw * (vdw_scale - 1.0)
        else:
            sc_adj = vdw_adj = 0.0
        return pe.e_total + sc_adj + vdw_adj
    return pe.e_total


def run_all(label, mode):
    n_pass = n_match = n_total = 0
    details = []
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
        has_metal = any(a.element.upper() in rp.METAL_GAUSSIAN_PARAMS for a in pa)
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
        best_pose = poses[bi].index
        best_dist = dists[bi]

        etr1 = min(results, key=lambda r: custom_score(r, mode, has_metal))
        etr1_dist = dists[etr1.pose_index - 1]
        is_best = (etr1.pose_index == best_pose)
        n_total += 1
        if etr1_dist < 2.0: n_pass += 1
        if is_best: n_match += 1
        details.append((pdb, best_dist, etr1_dist, is_best, etr1_dist<2.0, etr1.pose_index, best_pose))

    # Print summary
    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"  PASS={n_pass}/{n_total}  MATCH={n_match}/{n_total}")
    # Only print non-matching for clean output
    bad = [(p,bd,ed,ep,bp) for p,bd,ed,ib,pass_,ep,bp in details
           if not ib and ed>=2.0]
    if bad:
        print(f"  FAIL cases: " + ", ".join(f"{p}:P{ep}({ed:.2f})" for p,bd,ed,ep,bp in bad))
    return n_pass, n_match


# ── Baseline ──────────────────────────────────────────────────────────────
run_all("BASELINE (e_total)", 'default')

# ── Only metal + BB HB + VDW ──────────────────────────────────────────────
run_all("simple: e_metal + e_hb_bb + e_vdw", 'simple')
run_all("simple_no_vdwpos: e_metal + e_hb_bb + min(vdw,0)", 'simple_no_vdw_pos')

# ── Metal weight multiplied ────────────────────────────────────────────────
run_all("metal_heavy: e_total + e_metal  (metal×2)", 'metal_heavy')
run_all("metal_heavy3: e_total + 2×e_metal  (metal×3)", 'metal_heavy3')

# ── Positive VDW = 0 ──────────────────────────────────────────────────────
run_all("no_vdw_pos: cap vdw at 0 (no repulsion penalty)", 'no_vdw_pos')

# ── SC scale + VDW positive cap ───────────────────────────────────────────
for sc in [0.0, 0.1, 0.2, 0.3, 0.5]:
    run_all(f"SC={sc} + no_vdw_pos", ('sc_scale_novdwpos', sc))

# ── SC scale + VDW scale (reduce VDW weight entirely) ─────────────────────
for sc in [0.0, 0.1, 0.2]:
    for vdws in [0.0, 0.1, 0.2]:
        run_all(f"SC_scale={sc} + VDW_scale={vdws}", ('sc_scale_vdwscale', sc, vdws))

print("\n\nDone.")
