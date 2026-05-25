#!/usr/bin/env python3
"""
Prepare receptor + ligand PDBQT files and docking configs for all validation targets.
Uses AutoDockTools (jp_214 env) for receptor/ligand preparation.
Usage: python3 prepare_all.py
"""

import os, sys, subprocess, glob, json
import re

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PDB_DIR     = os.path.join(BASE_DIR, "pdb")
REC_DIR     = os.path.join(BASE_DIR, "pdbqt_receptor")
LIG_DIR     = os.path.join(BASE_DIR, "pdbqt_ligand")
CFG_DIR     = os.path.join(BASE_DIR, "configs")
WORK_DIR    = os.path.join(BASE_DIR, "tmp_work")

for d in [REC_DIR, LIG_DIR, CFG_DIR, WORK_DIR]:
    os.makedirs(d, exist_ok=True)

# AutoDockTools scripts (jp_214 conda env)
ADT_PYTHON = "/home/cycheng/miniforge3/envs/jp_214/bin/python"
ADT_PREFIX = "/home/cycheng/miniforge3/envs/jp_214/MGLToolsPckgs/AutoDockTools/Utilities24"
PREP_REC   = os.path.join(ADT_PREFIX, "prepare_receptor4.py")
PREP_LIG   = os.path.join(ADT_PREFIX, "prepare_ligand4.py")

# Curated target list: (PDB_ID, chain, ligand_resname, metal, ki_nM, description)
TARGETS = [
    # CAII — Zn²⁺
    ("1OQ5", "A", "CEL", "ZN", 9,    "CAII + Acetazolamide"),
    ("1BNN", "A", "AL1", "ZN", 380,  "CAII + aminobenzenesulfonamide"),
    ("3P5A", "A", "IT2", "ZN", 200,  "CAII + dithiocarbamate (S→Zn)"),
    ("1A42", "A", "BZU", "ZN", 15,   "CAII + Acetazolamide analog"),
    ("1YDB", "A", "AZM", "ZN", 2,    "CAII + fluorinated sulfonamide"),
    ("3HS4", "A", "AZM", "ZN", 8300, "CAII + coumarin inhibitor"),
    ("2CBD", "A", "SO3", "ZN", 900,  "CAII + sulfite (direct Zn)"),
    ("1G52", "A", "F2B", "ZN", 100,  "CAII + fluorinated benzensulfonamide"),
    ("1ZNC", "A", "SO4", "ZN", 200,  "CAII-IV + sulfate (direct Zn)"),
    # MMP — Zn²⁺
    ("1GKC", "A", "NFH", "ZN", 27,   "MMP-2 + phosphonate inhibitor"),
    ("1MMQ", "A", "RRS", "ZN", 670,  "MMP-1 + peptide hydroxamate"),
    ("1JAQ", "A", "01S", "ZN", 10,   "MMP-8 + tripeptide hydroxamate"),
    ("2OVX", "A", "4MR", "ZN", 26,   "MMP-9 + hydroxamate"),
    # ACE — Zn²⁺
    ("1O86", "A", "LPR", "ZN", 1.7,  "ACE + Captopril"),
    ("2C6N", "A", "LPR", "ZN", 0.27, "ACE + Lisinopril"),
    ("1UZE", "A", "EAL", "ZN", 2.2,  "ACE + Enalaprilat"),
    # HIV Integrase — Mg²⁺
    ("3L2U", "A", "ELV", "MG", 2.0,  "HIV-IN + Elvitegravir analog"),
    ("3S3M", "A", "DLU", "MG", 3.0,  "HIV-IN + Raltegravir analog"),
    # PHD2/Fe metalloenzymes
    ("2W0D", "A", "CGS", "ZN", 1.0,  "MMP-9 + CGS27023A hydroxamate"),
    ("2G1M", "A", "4HG", "FE", 50,   "PHD2 + 4HG inhibitor (Fe)"),
]

def run(cmd, cwd=None, silent=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if not silent and result.returncode != 0:
        print(f"  [WARN] cmd: {cmd}")
        if result.stderr: print(f"  stderr: {result.stderr[:200]}")
    return result.returncode == 0, result.stdout, result.stderr

def extract_chain(pdb_path, chain, out_path, exclude_resnames: set | None = None):
    """Extract ATOM (protein) records for given chain + metal HETATM records.

    exclude_resnames: residue names to omit from HETATM records (e.g. the
    co-crystallized ligand).  Metal ions (single-element residue names) and
    cofactors NOT in exclude_resnames are kept so that the receptor PDBQT
    contains the catalytic metal but NOT the bound ligand.
    """
    exclude = exclude_resnames or set()
    with open(pdb_path) as f:
        lines = f.readlines()
    kept = []
    for line in lines:
        if line.startswith("ATOM  ") and line[21] == chain:
            kept.append(line)
        elif line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname not in exclude:
                kept.append(line)
        elif line.startswith(("TER", "END", "CONECT", "MASTER")):
            kept.append(line)
    with open(out_path, "w") as f:
        f.writelines(kept)

def extract_ligand(pdb_path, lig_resname, out_path):
    """Extract ligand HETATM records (first occurrence of resname)"""
    with open(pdb_path) as f:
        lines = f.readlines()
    # Find residue number of first occurrence
    lig_lines = [l for l in lines if l.startswith("HETATM") and l[17:20].strip() == lig_resname]
    if not lig_lines:
        return False
    # Use only first residue number to avoid duplicates in multi-chain structures
    first_resnum = lig_lines[0][22:26].strip()
    first_chain  = lig_lines[0][21]
    kept = [l for l in lig_lines if l[22:26].strip() == first_resnum and l[21] == first_chain]
    with open(out_path, "w") as f:
        f.write("REMARK  Ligand %s from %s\n" % (lig_resname, os.path.basename(pdb_path)))
        f.writelines(kept)
        f.write("END\n")
    return len(kept) > 0

def get_ligand_center(lig_pdb):
    """Compute ligand geometric center (x, y, z)"""
    coords = []
    with open(lig_pdb) as f:
        for line in f:
            if line.startswith("HETATM") or line.startswith("ATOM  "):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    coords.append((x, y, z))
                except:
                    pass
    if not coords:
        return None
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    return cx, cy, cz

def prep_receptor(pdb_path, out_pdbqt):
    """Run prepare_receptor4.py — keep metal ions, add H, remove waters"""
    cmd = (f"{ADT_PYTHON} {PREP_REC} "
           f"-r {pdb_path} -o {out_pdbqt} "
           f"-A hydrogens -U nphs_lps_waters_nonstdres")
    ok, _, err = run(cmd)
    if not ok:
        # Retry without -U flag (some structures need it)
        cmd2 = (f"{ADT_PYTHON} {PREP_REC} "
                f"-r {pdb_path} -o {out_pdbqt} -A hydrogens")
        ok, _, _ = run(cmd2)
    return ok and os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 100

def prep_ligand(pdb_path, out_pdbqt):
    """Convert ligand PDB to PDBQT via OpenBabel, then prepare_ligand4.py"""
    # Step 1: OpenBabel PDB -> MOL2 (adds H, assigns charges)
    mol2_path = pdb_path.replace(".pdb", ".mol2")
    ok, _, _ = run(f"obabel {pdb_path} -O {mol2_path} -h --gen3d 2>/dev/null")
    if not ok or not os.path.exists(mol2_path):
        # Fallback: try direct PDB → PDBQT with obabel
        ok, _, _ = run(f"obabel {pdb_path} -O {out_pdbqt} -xh 2>/dev/null")
        return ok and os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 50

    # Step 2: prepare_ligand4.py from mol2
    cmd = (f"{ADT_PYTHON} {PREP_LIG} "
           f"-l {mol2_path} -o {out_pdbqt} -A hydrogens")
    ok, _, _ = run(cmd)
    if ok and os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 50:
        return True

    # Fallback: obabel direct
    ok, _, _ = run(f"obabel {pdb_path} -O {out_pdbqt} -xh 2>/dev/null")
    return ok and os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 50

def write_config(pdb_id, center, size=(20, 20, 20)):
    cfg_path = os.path.join(CFG_DIR, f"{pdb_id}_config.txt")
    with open(cfg_path, "w") as f:
        f.write(f"receptor = {os.path.join(REC_DIR, pdb_id + '_receptor.pdbqt')}\n")
        f.write(f"ligand_directory = {LIG_DIR}\n")
        f.write(f"output_directory = {os.path.join(BASE_DIR, 'results', pdb_id)}\n")
        f.write(f"opencl_binary_path = /opt/vina-gpu/AutoDock-Vina-GPU-2.1\n")
        f.write(f"center_x = {center[0]:.3f}\n")
        f.write(f"center_y = {center[1]:.3f}\n")
        f.write(f"center_z = {center[2]:.3f}\n")
        f.write(f"size_x = {size[0]}\n")
        f.write(f"size_y = {size[1]}\n")
        f.write(f"size_z = {size[2]}\n")
        f.write(f"thread = 8000\n")
        f.write(f"search_depth = 1\n")
    return cfg_path

def main():
    ok_count = fail_count = 0
    summary = []

    print(f"\nPreparing {len(TARGETS)} target structures...\n")
    print(f"{'PDB':6} {'Metal':5} {'Lig':5} {'Rec':>6} {'Lig':>6} {'Center':^26} Status")
    print("-"*72)

    for pdb_id, chain, lig_res, metal, ki, desc in TARGETS:
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            print(f"{pdb_id:6} {'':5} {'':5} {'':>6} {'':>6} {'':^26} MISSING PDB")
            fail_count += 1
            continue

        work_rec  = os.path.join(WORK_DIR, f"{pdb_id}_rec.pdb")
        work_lig  = os.path.join(WORK_DIR, f"{pdb_id}_{lig_res}.pdb")
        rec_pdbqt = os.path.join(REC_DIR,  f"{pdb_id}_receptor.pdbqt")
        lig_pdbqt = os.path.join(LIG_DIR,  f"{pdb_id}_{lig_res}.pdbqt")
        os.makedirs(os.path.join(BASE_DIR, "results", pdb_id), exist_ok=True)

        # 1. Extract chain — exclude co-crystallized ligand so it is not part
        #    of the receptor (would block the crystal binding pose during docking)
        extract_chain(pdb_path, chain, work_rec, exclude_resnames={lig_res})

        # 2. Extract ligand
        if not extract_ligand(pdb_path, lig_res, work_lig):
            print(f"{pdb_id:6} {metal:5} {lig_res:5} {'':>6} {'':>6} {'':^26} NO LIGAND FOUND")
            fail_count += 1
            continue

        # 3. Prepare receptor PDBQT
        rec_ok = False
        if os.path.exists(rec_pdbqt) and os.path.getsize(rec_pdbqt) > 100:
            rec_ok = True  # skip if already done
        else:
            rec_ok = prep_receptor(work_rec, rec_pdbqt)
        rec_atoms = sum(1 for l in open(rec_pdbqt) if l.startswith(("ATOM","HETATM"))) if rec_ok else 0

        # 4. Prepare ligand PDBQT
        lig_ok = False
        if os.path.exists(lig_pdbqt) and os.path.getsize(lig_pdbqt) > 50:
            lig_ok = True  # skip if already done
        else:
            lig_ok = prep_ligand(work_lig, lig_pdbqt)
        lig_atoms = sum(1 for l in open(lig_pdbqt) if l.startswith(("ATOM","HETATM"))) if lig_ok else 0

        # 5. Compute docking box from crystal ligand position
        center = get_ligand_center(work_lig)
        cfg_path = None
        if center:
            cfg_path = write_config(pdb_id, center)

        status = "OK" if (rec_ok and lig_ok and center) else "PARTIAL"
        cx = f"({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})" if center else "N/A"
        print(f"{pdb_id:6} {metal:5} {lig_res:5} {rec_atoms:>6} {lig_atoms:>6} {cx:^26} {status}")

        if status == "OK":
            ok_count += 1
        else:
            fail_count += 1

        summary.append({
            "pdb_id": pdb_id, "chain": chain, "ligand": lig_res,
            "metal": metal, "ki_nM": ki, "desc": desc,
            "rec_pdbqt": rec_pdbqt, "lig_pdbqt": lig_pdbqt,
            "config": cfg_path, "center": center,
            "rec_ok": rec_ok, "lig_ok": lig_ok
        })

    print("-"*72)
    print(f"Done: {ok_count} ready, {fail_count} issues\n")

    # Write summary JSON for the RMSD script to consume
    with open(os.path.join(BASE_DIR, "targets_prepared.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: targets_prepared.json")
    print(f"\nNext step: run  python3 run_baseline.py")

if __name__ == "__main__":
    main()
