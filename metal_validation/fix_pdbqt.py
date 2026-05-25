#!/usr/bin/env python3
"""
Fix PDBQT files for Vina compatibility:
1. Receptors: remove lines with corrupted column alignment caused by AutoDockTools
   adding hydrogens to alternate conformations (atom names like HH1A + extra digit
   before resName shifts x/y/z fields right by 1).
   Also strip invalid heavy-atom types (Tl, As, etc.).
2. Ligands: deduplicate multiple ROOT blocks (keep first complete block only).
"""
import os, glob, re, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REC_DIR  = os.path.join(BASE_DIR, "pdbqt_receptor")
LIG_DIR  = os.path.join(BASE_DIR, "pdbqt_ligand")
TMP_DIR  = os.path.join(BASE_DIR, "tmp_lig")

INVALID_HETATM_TYPES = {"Tl","As","Sr","Ba","Cs","Rb","Au","Pt","Ir","Pb","Bi","Xe","Kr"}

def is_coordinate_line_valid(line):
    """
    PDBQT ATOM/HETATM: x at [30:38], y at [38:46], z at [46:54] (0-indexed).
    Return True if all three parse as floats.
    """
    if len(line) < 55:
        return False
    try:
        float(line[30:38])
        float(line[38:46])
        float(line[46:54])
        return True
    except ValueError:
        return False

def is_altconf_B(line):
    """
    Return True if the altLoc field (index 16, col 17) indicates a non-primary
    alternate conformation (B, C, 2, 3, ...).
    Discard these entirely — they're duplicates of the A/primary conformation.
    """
    if len(line) < 17:
        return False
    altloc = line[16]
    return altloc in ('B','C','D','2','3','4','5','6')

def fix_receptor(path):
    with open(path) as f:
        lines = f.readlines()

    fixed = []
    changed = 0
    for line in lines:
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            fixed.append(line)
            continue

        # Drop non-primary alternate conformations (B, C, 2, 3, ...)
        if is_altconf_B(line):
            changed += 1
            continue

        # Drop lines with invalid heavy-atom HETATM types (Tl, As, etc.)
        parts = line.split()
        atype = parts[-1] if parts else ""
        if rec == "HETATM" and atype in INVALID_HETATM_TYPES:
            changed += 1
            continue

        # Drop lines where coordinate columns are unreadable (column-shifted lines)
        if not is_coordinate_line_valid(line):
            changed += 1
            continue

        fixed.append(line)

    if changed:
        shutil.copy2(path, path + ".bak")
        with open(path, "w") as f:
            f.writelines(fixed)
        print(f"  [fixed] {os.path.basename(path)}: removed {changed} bad lines")
    return changed

def fix_ligand(path):
    """Keep only the first complete ROOT...TORSDOF block."""
    with open(path) as f:
        content = f.read()

    # Split on each REMARK Name header — each is the start of a block
    blocks = re.split(r'(?=^REMARK\s+Name)', content, flags=re.MULTILINE)
    if len(blocks) <= 1:
        return 0  # only one section, no deduplication needed

    # Find first block that contains a complete ROOT...TORSDOF
    for i, block in enumerate(blocks):
        if "ROOT" in block and "TORSDOF" in block:
            # Trim at end of TORSDOF line
            end_pos = block.rfind("TORSDOF")
            end_line = block.find("\n", end_pos)
            first_block = block[:end_line+1] if end_line != -1 else block[:end_pos+20]
            if first_block.strip() == content.strip():
                return 0  # already clean
            shutil.copy2(path, path + ".bak")
            with open(path, "w") as f:
                f.write(first_block + "\n")
            print(f"  [fixed] {os.path.basename(path)}: kept block {i+1}/{len(blocks)}")
            return 1

    return 0

def main():
    print("=== Fixing receptor PDBQTs ===")
    total_rec = 0
    for rec in sorted(glob.glob(os.path.join(REC_DIR, "*.pdbqt"))):
        n = fix_receptor(rec)
        total_rec += n
        if n == 0:
            print(f"  [ok]    {os.path.basename(rec)}")

    print(f"\n=== Fixing ligand PDBQTs (ligand dir) ===")
    for lig in sorted(glob.glob(os.path.join(LIG_DIR, "*.pdbqt"))):
        n = fix_ligand(lig)
        if n == 0:
            print(f"  [ok]    {os.path.basename(lig)}")

    print(f"\n=== Fixing ligand PDBQTs (tmp_lig) ===")
    for lig in sorted(glob.glob(os.path.join(TMP_DIR, "*", "*.pdbqt"))):
        n = fix_ligand(lig)
        if n == 0:
            print(f"  [ok]    {os.path.basename(lig)}")

    print(f"\nDone. Receptor lines removed: {total_rec}")
    print("Re-run: sbatch run_baseline.sh")

if __name__ == "__main__":
    main()
