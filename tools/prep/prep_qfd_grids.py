#!/usr/bin/env python3
"""
prep_qfd_grids.py — Compute QFD (Quantum Field Docking) receptor field grids.

Produces three binary grid files in the current working directory:
  qfd_esp.bin      — electrostatic potential field  [kcal/(mol·e)]
  qfd_desolv.bin   — desolvation susceptibility     [kcal/mol per e²]
  qfd_infomap.bin  — information resonance field    [dimensionless]

Binary format (read by main_procedure_cl.cpp::load_qfd_grid_file):
  int[3]   : m_i, m_j, m_k
  float[12]: m_init[3], m_factor[3], m_dim_fl_minus_1[3], m_factor_inv[3]
  float[m_i*m_j*m_k*8]: trilinear interpolation coefficients (Vina grid format)

Usage:
  python3 prep_qfd_grids.py --receptor receptor.pdbqt \\
      --center_x X --center_y Y --center_z Z \\
      --size_x SX --size_y SY --size_z SZ \\
      [--spacing 0.375] [--output_dir .]

The grid box should match the Vina docking box exactly.
"""

import argparse
import struct
import sys
import os
import math
import numpy as np


# ---------------------------------------------------------------------------
# Atom-type partial charge table (Gasteiger-type defaults for PDBQT atoms)
# Used when the PDBQT charge column is absent or zero for heavy atoms.
# ---------------------------------------------------------------------------
_ELEMENT_CHARGE_FALLBACK = {
    'N':  -0.30, 'NA': -0.35, 'NS': -0.25,
    'O':  -0.35, 'OA': -0.40, 'OS': -0.30,
    'S':  -0.10, 'SA': -0.15,
    'C':   0.05, 'A':  0.05,
    'H':   0.20, 'HD':  0.25,
    'P':   0.40,
    'F':  -0.15, 'Cl': -0.10, 'Br': -0.10, 'I':  -0.05,
    'Zn':  2.0,  'Ca':  2.0,  'Mg':  2.0,  'Fe':  2.0, 'Mn':  2.0,
    'Cu':  2.0,  'Co':  2.0,  'Ni':  2.0,
}


def _parse_pdbqt(path):
    """Parse PDBQT file; return list of (x, y, z, charge, atom_type_str)."""
    atoms = []
    with open(path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ('ATOM', 'HETATM'):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            try:
                charge = float(line[70:76])
            except (ValueError, IndexError):
                charge = 0.0
            atom_type = line[77:].split()[0].strip() if len(line) > 77 else ''
            if charge == 0.0:
                charge = _ELEMENT_CHARGE_FALLBACK.get(atom_type, 0.0)
            atoms.append((x, y, z, charge, atom_type))
    return atoms


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _make_grid_metadata(cx, cy, cz, sx, sy, sz, spacing):
    """
    Build grid metadata matching Vina's grid_cl layout.

    Vina grid: m_i × m_j × m_k voxels centred on (cx,cy,cz) with given spacing.
    The grid covers sx × sy × sz Ångströms total.
    Returns dict with all metadata fields + actual grid dims.
    """
    mi = int(round(sx / spacing)) + 1
    mj = int(round(sy / spacing)) + 1
    mk = int(round(sz / spacing)) + 1

    # Origin: lower corner of grid (same convention as Vina)
    ox = cx - (mi - 1) / 2.0 * spacing
    oy = cy - (mj - 1) / 2.0 * spacing
    oz = cz - (mk - 1) / 2.0 * spacing

    m_init           = [ox, oy, oz]
    m_range          = [sx, sy, sz]
    m_factor         = [1.0 / spacing, 1.0 / spacing, 1.0 / spacing]
    m_dim_fl_minus_1 = [float(mi - 1), float(mj - 1), float(mk - 1)]
    m_factor_inv     = [spacing, spacing, spacing]

    return {
        'm_i': mi, 'm_j': mj, 'm_k': mk,
        'm_init': m_init, 'm_range': m_range,
        'm_factor': m_factor,
        'm_dim_fl_minus_1': m_dim_fl_minus_1,
        'm_factor_inv': m_factor_inv,
    }


def _flat_index(i, j, k, mj, mk):
    return (i * mj + j) * mk + k


def _trilinear_coeffs(values_at_corners):
    """
    Given 8 corner values of a voxel (c000, c100, c010, c110, c001, c101, c011, c111),
    return the 8 trilinear coefficients in Vina's precomputed format:
      f(u,v,w) = sum_n coeff[n] * basis_n(u,v,w)
    Vina stores the grid as 8 floats per voxel: the values at the 8 corners.
    (Vina does the interpolation itself; we just store the 8 corner values.)
    """
    return values_at_corners


# ---------------------------------------------------------------------------
# ESP grid computation
# ---------------------------------------------------------------------------

def compute_esp_grid(atoms, meta, dielectric_scale=4.0):
    """
    Coulomb electrostatic potential with distance-dependent dielectric.
    ε(r) = dielectric_scale * r  (Born-like screening)
    E_esp(r) = sum_i  q_i / (ε(r_i) * r_i)  in kcal/(mol·e)

    Conversion: 1 e² / (4πε₀ · 1 Å) = 332.06 kcal/mol  (AMBER constant)
    """
    COULOMB = 332.06  # kcal·Å / (mol·e²)
    mi, mj, mk = meta['m_i'], meta['m_j'], meta['m_k']
    ox, oy, oz = meta['m_init']
    spacing = meta['m_factor_inv'][0]

    # Build coordinate arrays for grid points
    xs = ox + np.arange(mi) * spacing
    ys = oy + np.arange(mj) * spacing
    zs = oz + np.arange(mk) * spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')  # shape (mi, mj, mk)

    esp = np.zeros((mi, mj, mk), dtype=np.float32)

    for ax, ay, az, q, _ in atoms:
        if q == 0.0:
            continue
        dx = gx - ax
        dy = gy - ay
        dz = gz - az
        r2 = dx*dx + dy*dy + dz*dz
        r  = np.sqrt(r2 + 1e-8)    # avoid /0
        # distance-dependent dielectric: ε(r) = dielectric_scale * r
        esp += (COULOMB * q) / (dielectric_scale * r2 + 1e-8)

    return esp


# ---------------------------------------------------------------------------
# Desolvation susceptibility grid
# ---------------------------------------------------------------------------

def compute_desolv_grid(atoms, meta, burial_sigma=3.5):
    """
    Desolvation grid: fraction of solvent-excluded volume at each point.
    Approximated as a Gaussian accumulation from receptor atoms:
      D(r) = sum_i  exp(-|r - r_i|² / (2σ²))
    Represents "how buried this point is" in the receptor.
    When a ligand atom with charge q sits here, it pays desolvation cost ∝ q² * D(r).
    """
    mi, mj, mk = meta['m_i'], meta['m_j'], meta['m_k']
    ox, oy, oz = meta['m_init']
    spacing = meta['m_factor_inv'][0]

    xs = ox + np.arange(mi) * spacing
    ys = oy + np.arange(mj) * spacing
    zs = oz + np.arange(mk) * spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')

    desolv = np.zeros((mi, mj, mk), dtype=np.float32)
    inv_2sig2 = 1.0 / (2.0 * burial_sigma**2)

    for ax, ay, az, _, _ in atoms:
        dx = gx - ax
        dy = gy - ay
        dz = gz - az
        r2 = dx*dx + dy*dy + dz*dz
        desolv += np.exp(-r2 * inv_2sig2).astype(np.float32)

    return desolv


# ---------------------------------------------------------------------------
# Water displacement penalty grid — Phase 3 of QFD
# ---------------------------------------------------------------------------

def parse_water_positions(path: str) -> list:
    """
    Extract crystallographic water oxygen positions from a PDB or PDBQT file.

    Matches:
      - HETATM lines with residue name HOH, WAT, or H2O
      - ATOM lines with atom name O in a water residue
    Returns list of (x, y, z) tuples.
    """
    waters = []
    with open(path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ('ATOM', 'HETATM'):
                continue
            resname = line[17:20].strip()
            if resname not in ('HOH', 'WAT', 'H2O', 'DOD'):
                continue
            atom_name = line[12:16].strip()
            if not atom_name.startswith('O'):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            waters.append((x, y, z))
    return waters


def compute_water_grid(water_positions: list, meta: dict,
                       sigma: float = 1.5) -> np.ndarray:
    """
    Water displacement penalty grid (QFD Phase 3).

    Places a Gaussian bump at each crystallographic water oxygen:
      W(r) = exp(-r² / (2σ²))
    with σ = 1.5 Å (the radius at which the penalty falls to 1/e).

    Interpretation: a ligand heavy atom sitting atop a crystallographic
    water pays a penalty proportional to W — unless it makes a compensating
    H-bond (not modelled here; the weight QFD_WATER_WEIGHT=0.03 is light
    enough that a genuine H-bond displaces the water beneficially).

    Args:
        water_positions : list of (x, y, z) for crystallographic water oxygens
        meta            : grid metadata from _make_grid_metadata()
        sigma           : Gaussian width in Å [default: 1.5]

    Returns:
        ndarray of shape (mi, mj, mk) in [0, 1]  (normalised so peak = 1)
    """
    mi, mj, mk = meta['m_i'], meta['m_j'], meta['m_k']
    ox, oy, oz = meta['m_init']
    spacing    = meta['m_factor_inv'][0]

    xs = ox + np.arange(mi) * spacing
    ys = oy + np.arange(mj) * spacing
    zs = oz + np.arange(mk) * spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')

    water = np.zeros((mi, mj, mk), dtype=np.float32)
    inv_2sig2 = 1.0 / (2.0 * sigma ** 2)

    for wx, wy, wz in water_positions:
        dx = gx - wx
        dy = gy - wy
        dz = gz - wz
        r2 = dx*dx + dy*dy + dz*dz
        water += np.exp(-r2 * inv_2sig2).astype(np.float32)

    # Normalise to [0, 1] so QFD_WATER_WEIGHT controls absolute scale
    vmax = water.max()
    if vmax > 0:
        water /= vmax

    return water


# ---------------------------------------------------------------------------
# Information resonance (infomap) grid — Phase 4 of QFD
# ---------------------------------------------------------------------------

def compute_infomap_grid(atoms: list, meta: dict, esp: np.ndarray | None = None,
                         l_max: int = 3, cutoff: float = 8.0) -> np.ndarray:
    """
    QFD Information Resonance field (Phase 4).

    Captures two complementary effects:

    1. Spherical-harmonic field orientation anisotropy (|∇ESP|²):
       High where the receptor's electrostatic field changes most rapidly.
       A ligand atom here gains maximum coupling by aligning its partial charge
       with the field gradient direction → angular / orientation sensitivity.

       Physics: This approximates the l=1 (dipole) term of the multipole
       expansion of the receptor's field.  Where |∇ESP|² is large, the ligand
       must orient correctly to gain energy — exactly the directional hydrogen-
       bonds, salt bridges, and π-stacking captured by infomap.

    2. Cooperative charge environment (fallback when ESP not available):
       |q| · exp(-r/λ) summed over receptor atoms → "how many charged neighbours"
       at each grid point.

    The two terms are combined: infomap = α·|∇ESP|_norm² + (1−α)·coop_norm
    with α=0.7 (gradient term dominates).

    Args:
        atoms   : list of (x, y, z, charge, atom_type)
        meta    : grid metadata dict from _make_grid_metadata()
        esp     : precomputed ESP ndarray (mi, mj, mk) — used for gradient term
        l_max   : max spherical harmonic order (currently only l=1 used)
        cutoff  : cooperative term cutoff distance (Å)
    """
    mi, mj, mk = meta['m_i'], meta['m_j'], meta['m_k']
    ox, oy, oz = meta['m_init']
    spacing    = meta['m_factor_inv'][0]

    # ── Term 1: |∇ESP|² — orientation sensitivity ──────────────────────────
    grad_term = np.zeros((mi, mj, mk), dtype=np.float32)
    if esp is not None:
        gx_grad = np.gradient(esp.astype(np.float64), spacing, axis=0)
        gy_grad = np.gradient(esp.astype(np.float64), spacing, axis=1)
        gz_grad = np.gradient(esp.astype(np.float64), spacing, axis=2)
        grad2   = (gx_grad**2 + gy_grad**2 + gz_grad**2).astype(np.float32)
        vmax    = grad2.max()
        if vmax > 0:
            grad_term = grad2 / vmax

    # ── Term 2: cooperative charge environment ──────────────────────────────
    xs = ox + np.arange(mi) * spacing
    ys = oy + np.arange(mj) * spacing
    zs = oz + np.arange(mk) * spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')

    coop = np.zeros((mi, mj, mk), dtype=np.float32)
    decay_len = 2.5  # Å

    for ax, ay, az, q, _ in atoms:
        abs_q = abs(q)
        if abs_q < 0.01:
            continue
        dx  = gx - ax
        dy  = gy - ay
        dz  = gz - az
        r   = np.sqrt(dx*dx + dy*dy + dz*dz + 1e-8)
        coop += (abs_q * np.exp(-r / decay_len) * (r < cutoff)).astype(np.float32)

    vmax = coop.max()
    if vmax > 0:
        coop /= vmax

    # ── Combine ─────────────────────────────────────────────────────────────
    alpha   = 0.7 if esp is not None else 0.0
    infomap = (alpha * grad_term + (1.0 - alpha) * coop).astype(np.float32)

    return infomap


# ---------------------------------------------------------------------------
# Vina trilinear grid packing
# (8 corner values per voxel, row-major i→j→k, same layout as g_evaluate in quasi_newton.cl)
# ---------------------------------------------------------------------------

def _pack_trilinear(values, mi, mj, mk):
    """
    Pack a (mi, mj, mk) ndarray into Vina's m_data layout:
      For each voxel (i,j,k), store 8 corner values:
      [v(i,j,k), v(i+1,j,k), v(i,j+1,k), v(i+1,j+1,k),
       v(i,j,k+1), v(i+1,j,k+1), v(i,j+1,k+1), v(i+1,j+1,k+1)]
    Total: (mi-1)*(mj-1)*(mk-1)*8 floats? No — Vina stores one 8-tuple per voxel using
    the voxel lower-left corner, with dimensions (mi)*(mj)*(mk)*8 where the last
    voxel wraps to the boundary value.

    Looking at g_evaluate in quasi_newton.cl lines 50-90:
      m->m_data[(i*MJ + j)*MK + k .. +7] are the 8 trilinear basis values.
    Convention: index k cycles fastest.
    Each set of 8 covers the (i,j,k) voxel corners at [i,i+1] × [j,j+1] × [k,k+1].
    For the boundary rows (i=mi-1 etc.), clamp to boundary.
    """
    v = values  # shape (mi, mj, mk)
    # Pad by one in each dimension for corner access
    vp = np.pad(v, ((0, 1), (0, 1), (0, 1)), mode='edge').astype(np.float32)
    data = np.empty((mi, mj, mk, 8), dtype=np.float32)
    data[..., 0] = vp[0:mi,   0:mj,   0:mk  ]
    data[..., 1] = vp[1:mi+1, 0:mj,   0:mk  ]
    data[..., 2] = vp[0:mi,   1:mj+1, 0:mk  ]
    data[..., 3] = vp[1:mi+1, 1:mj+1, 0:mk  ]
    data[..., 4] = vp[0:mi,   0:mj,   1:mk+1]
    data[..., 5] = vp[1:mi+1, 0:mj,   1:mk+1]
    data[..., 6] = vp[0:mi,   1:mj+1, 1:mk+1]
    data[..., 7] = vp[1:mi+1, 1:mj+1, 1:mk+1]
    return data.reshape(-1)


# ---------------------------------------------------------------------------
# Binary file I/O
# ---------------------------------------------------------------------------

def _write_qfd_bin(path, meta, grid_values):
    """Write a QFD grid binary file."""
    mi, mj, mk = meta['m_i'], meta['m_j'], meta['m_k']
    packed = _pack_trilinear(grid_values, mi, mj, mk)
    header_int   = struct.pack('<3i', mi, mj, mk)
    header_float = struct.pack('<12f',
        *meta['m_init'],
        *meta['m_factor'],
        *meta['m_dim_fl_minus_1'],
        *meta['m_factor_inv'],
    )
    with open(path, 'wb') as f:
        f.write(header_int)
        f.write(header_float)
        packed.astype('<f4').tofile(f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Wrote {path}  ({mi}×{mj}×{mk} = {mi*mj*mk} voxels, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--receptor', required=True,
                    help='Receptor PDBQT file with partial charges in column 72-76')
    ap.add_argument('--center_x', type=float, required=True, help='Box centre X (Å)')
    ap.add_argument('--center_y', type=float, required=True, help='Box centre Y (Å)')
    ap.add_argument('--center_z', type=float, required=True, help='Box centre Z (Å)')
    ap.add_argument('--size_x',   type=float, required=True, help='Box size X (Å)')
    ap.add_argument('--size_y',   type=float, required=True, help='Box size Y (Å)')
    ap.add_argument('--size_z',   type=float, required=True, help='Box size Z (Å)')
    ap.add_argument('--spacing',  type=float, default=0.375, help='Grid spacing (Å) [default: 0.375]')
    ap.add_argument('--output_dir', default='.', help='Output directory [default: .]')
    ap.add_argument('--dielectric_scale', type=float, default=4.0,
                    help='Distance-dependent dielectric scale ε(r)=k·r [default: 4.0]')
    ap.add_argument('--burial_sigma', type=float, default=3.5,
                    help='Gaussian σ for desolvation burial (Å) [default: 3.5]')
    ap.add_argument('--no_esp',      action='store_true', help='Skip ESP grid')
    ap.add_argument('--no_desolv',   action='store_true', help='Skip desolvation grid')
    ap.add_argument('--no_infomap',  action='store_true', help='Skip infomap grid')
    ap.add_argument('--no_water',    action='store_true', help='Skip water displacement grid')
    ap.add_argument('--water_pdb',   default=None,
                    help='PDB/PDBQT file with crystallographic water positions '
                         '(HOH/WAT residues). If omitted, waters are extracted '
                         'from the receptor file itself.')
    ap.add_argument('--water_sigma', type=float, default=1.5,
                    help='Gaussian σ for water penalty bump (Å) [default: 1.5]')
    args = ap.parse_args()

    print(f"[prep_qfd_grids] Loading receptor: {args.receptor}")
    atoms = _parse_pdbqt(args.receptor)
    print(f"  {len(atoms)} atoms loaded")
    charged = [a for a in atoms if abs(a[3]) > 0.01]
    print(f"  {len(charged)} atoms with |q| > 0.01")

    meta = _make_grid_metadata(
        args.center_x, args.center_y, args.center_z,
        args.size_x, args.size_y, args.size_z,
        args.spacing,
    )
    print(f"  Grid dims: {meta['m_i']}×{meta['m_j']}×{meta['m_k']}")
    print(f"  Grid origin: ({meta['m_init'][0]:.2f}, {meta['m_init'][1]:.2f}, {meta['m_init'][2]:.2f})")

    os.makedirs(args.output_dir, exist_ok=True)

    # ESP is computed first; used by infomap (Phase 4 gradient term)
    esp = None
    if not args.no_esp:
        print("[prep_qfd_grids] Computing ESP grid…")
        esp = compute_esp_grid(atoms, meta, dielectric_scale=args.dielectric_scale)
        _write_qfd_bin(os.path.join(args.output_dir, 'qfd_esp.bin'), meta, esp)

    if not args.no_desolv:
        print("[prep_qfd_grids] Computing desolvation grid…")
        desolv = compute_desolv_grid(atoms, meta, burial_sigma=args.burial_sigma)
        _write_qfd_bin(os.path.join(args.output_dir, 'qfd_desolv.bin'), meta, desolv)

    if not args.no_infomap:
        print("[prep_qfd_grids] Computing information resonance grid (Phase 4)…")
        if esp is not None:
            print("  Using |∇ESP|² orientation-sensitivity term (l=1 spherical harmonic proxy)")
        else:
            print("  ESP not computed — using cooperative charge environment only")
        infomap = compute_infomap_grid(atoms, meta, esp=esp)
        _write_qfd_bin(os.path.join(args.output_dir, 'qfd_infomap.bin'), meta, infomap)

    if not args.no_water:
        water_src = args.water_pdb if args.water_pdb else args.receptor
        print(f"[prep_qfd_grids] Reading water positions from: {water_src}")
        water_pos = parse_water_positions(water_src)
        if not water_pos:
            print("  No HOH/WAT residues found — skipping water grid.")
        else:
            print(f"  {len(water_pos)} crystallographic water(s) found.")
            water_grid = compute_water_grid(water_pos, meta, sigma=args.water_sigma)
            _write_qfd_bin(os.path.join(args.output_dir, 'qfd_water.bin'), meta, water_grid)

    print("[prep_qfd_grids] Done. Run Vina from the same directory to use QFD grids.")
    print("  QFD grids active: ESP=%s  DESOLV=%s  INFOMAP=%s  WATER=%s" % (
        'yes' if not args.no_esp else 'no',
        'yes' if not args.no_desolv else 'no',
        'yes' if not args.no_infomap else 'no',
        'yes' if (not args.no_water and (args.water_pdb or True)) else 'no',
    ))


if __name__ == '__main__':
    main()
