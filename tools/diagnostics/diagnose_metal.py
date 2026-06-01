#!/usr/bin/env python3
"""Deep-dive into metal coordination geometry for failing cases.

For each target, show which ligand donors contribute to e_metal for both
the crystal-closest pose and the ETr=1 pose.  This reveals whether the
scoring difference is due to geometry (distance/angle) or donor count.
"""
import sys, math
sys.path.insert(0, '/home/cycheng/LigandScope/scripts')
import rank_poses as rp
from pathlib import Path

BASE = '/home/cycheng/Vina-GPU-2.1/metal_validation'

TARGETS = [
    ('1GKC','NFH','ZN'),  # gap=1.64, e_metal drives
    ('1YDB','AZM','ZN'),  # gap=0.79, e_vdw drives
    ('1BNN','AL1','ZN'),  # gap=2.04, e_hb_sc drives
    ('2C6N','LPR','ZN'),  # gap=4.22, e_hb_sc drives
    ('3P5A','IT2','ZN'),  # gap=0.50, e_hphob drives
    ('1UZE','EAL','ZN'),  # gap=0.70, e_hb_sc drives
    ('1ZNC','SO4','ZN'),  # gap=0.57, spread
    ('2W0D','CGS','ZN'),  # gap=2.42, all terms ETr wins
]

METAL_COORD_RMAX = 4.0
_METAL_HB_EXCL_R2 = 7.84  # 2.8Å^2


def centroid_dist_atoms(atoms_a, atoms_b):
    ha = [a for a in atoms_a if a.element not in ('H','HD')]
    hb = [a for a in atoms_b if a.element not in ('H','HD')]
    if not ha or not hb: return 999.0
    cx = sum(a.x for a in ha)/len(ha)
    cy = sum(a.y for a in ha)/len(ha)
    cz = sum(a.z for a in ha)/len(ha)
    dx = sum(b.x for b in hb)/len(hb) - cx
    dy = sum(b.y for b in hb)/len(hb) - cy
    dz = sum(b.z for b in hb)/len(hb) - cz
    return math.sqrt(dx*dx+dy*dy+dz*dz)


def gauss_well(r, r0, D, K):
    """Metal Gaussian well energy (negative = attractive)."""
    return -D * math.exp(-K * (r - r0)**2)


def metal_coord_detail(pose_atoms, prot_atoms):
    """Return per-donor metal coordination details."""
    metals = [a for a in prot_atoms if a.element.upper() in rp.METAL_GAUSSIAN_PARAMS]
    donors = [a for a in pose_atoms if a.element.upper() in rp.METAL_DONOR_ELEMENTS
              and a.element not in ('H','HD')]

    details = []
    for m in metals:
        params = rp.METAL_GAUSSIAN_PARAMS.get(m.element.upper(), {})
        for d in donors:
            dx = d.x - m.x; dy = d.y - m.y; dz = d.z - m.z
            r = math.sqrt(dx*dx+dy*dy+dz*dz)
            if r > METAL_COORD_RMAX:
                continue
            elem = d.element.upper()
            if elem not in params:
                continue
            r0, D = params[elem][:2]
            # K from rp module
            K = rp.METAL_GAUSSIAN_K
            e = gauss_well(r, r0, D, K)
            details.append({
                'metal': m.element,
                'metal_res': getattr(m, 'res_name', '?'),
                'donor': d.element,
                'donor_name': getattr(d, 'name', '?'),
                'r': r,
                'r0': r0,
                'D': D,
                'e_gauss': e,
            })

    # Primary coordination: best donor wins
    if not details:
        return details, 0.0

    best = min(details, key=lambda x: x['e_gauss'])
    return details, best['e_gauss']


def run_target(pdb, lig, metal):
    rec = Path(f'{BASE}/pdbqt_receptor/{pdb}_receptor.pdbqt')
    poses_f = Path(f'{BASE}/results_metal/{pdb}/sd32/{lig}_out.pdbqt')
    if not poses_f.exists():
        poses_f = Path(f'{BASE}/results_metal/{pdb}/{lig}_out.pdbqt')
    if not poses_f.exists():
        poses_f = Path(f'{BASE}/results/{pdb}/{lig}_out.pdbqt')
    crystal_f = Path(f'{BASE}/pdbqt_ligand/{pdb}_{lig}.pdbqt')

    if not rec.exists() or not poses_f.exists():
        return

    poses = rp.split_vina_poses(poses_f)
    if not poses: return

    lig0 = poses[0].atoms
    adapt_r = rp.ligand_adaptive_radius(lig0)
    com = rp._center_of_mass(lig0)
    prot_atoms = rp.parse_protein(rec, adapt_r, com, lig_atoms=lig0)
    prot_rings = rp.detect_rings(prot_atoms)

    results = [rp.compute_pose_energy(p, prot_atoms, prot_rings) for p in poses]

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
        dists = [0.0] + [999.0]*(len(poses)-1)

    best_idx = min(range(len(dists)), key=lambda i: dists[i])
    best_pose = poses[best_idx]
    best_res = results[best_idx]
    best_dist = dists[best_idx]

    etr1_res = min(results, key=lambda r: r.e_total)
    etr1_pose = poses[etr1_res.pose_index - 1]
    etr1_dist = dists[etr1_res.pose_index - 1]

    print(f"\n{'='*70}")
    print(f"  {pdb} ({lig}/{metal}) | best=P{best_pose.index}({best_dist:.3f}Å) | ETr=P{etr1_pose.index}({etr1_dist:.3f}Å)")
    print(f"{'='*70}")

    # Metal coordination details for best pose
    metals = [a for a in prot_atoms if a.element.upper() in rp.METAL_GAUSSIAN_PARAMS]
    if not metals:
        print("  No protein metals found in prot_atoms")
        return

    print(f"\n  Protein metals: {[a.element for a in metals]}")

    for label, pose, res, dist in [
        ('Crystal-closest', best_pose, best_res, best_dist),
        ('ETr=1', etr1_pose, etr1_res, etr1_dist),
    ]:
        print(f"\n  [{label}] P{pose.index} ({dist:.3f}Å) | e_metal={res.e_metal:.3f} e_total={res.e_total:.3f}")
        details, _ = metal_coord_detail(pose.atoms, prot_atoms)

        # Show metal-coord exclusion set
        metal_coord_ids = set()
        for ma in metals:
            for la in pose.atoms:
                if la.element.upper() not in rp.METAL_DONOR_ELEMENTS: continue
                dx = la.x-ma.x; dy = la.y-ma.y; dz = la.z-ma.z
                if dx*dx+dy*dy+dz*dz <= _METAL_HB_EXCL_R2:
                    metal_coord_ids.add(id(la))

        if details:
            print(f"  {'Donor':8} {'Elem':5} {'r(Å)':6} {'r0(Å)':6} {'D':6} {'E_gauss':8}  Excl?")
            print(f"  {'-'*55}")
            for d in sorted(details, key=lambda x: x['e_gauss']):
                # Find the atom
                excl = ''
                for la in pose.atoms:
                    if la.element.upper() == d['donor'] and abs(la.x - metals[0].x - d['r'] * (la.x - metals[0].x) / d['r']) < 0.01:
                        pass
                # Simplified: check if any donor atom matching this r is in excl
                # Actually we need to track by position
                print(f"  {d['donor_name']:8} {d['donor']:5} {d['r']:6.3f} {d['r0']:6.3f} {d['D']:6.1f} {d['e_gauss']:8.3f}")
        else:
            print("  No donors within 4Å of metal")

        # Show metal-coordinating atoms that are within 2.8Å
        close_donors = []
        for ma in metals:
            for la in pose.atoms:
                if la.element.upper() not in rp.METAL_DONOR_ELEMENTS: continue
                dx = la.x-ma.x; dy = la.y-ma.y; dz = la.z-ma.z
                r2 = dx*dx+dy*dy+dz*dz
                if r2 <= _METAL_HB_EXCL_R2:
                    close_donors.append((math.sqrt(r2), la.element, getattr(la,'name','?')))
        if close_donors:
            print(f"  Ligand donors within 2.8Å (HB-excluded): " +
                  ", ".join(f"{n}({e},{r:.2f}Å)" for r,e,n in close_donors))


for pdb, lig, metal in TARGETS:
    run_target(pdb, lig, metal)

print("\n\nDone.")
