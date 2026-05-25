#!/usr/bin/env python3
"""
Download PDB structures for metal-ion docking validation.
Usage: python download_pdb.py
"""

import os
import sys
import urllib.request
import urllib.error
import time

PDB_DIR = os.path.join(os.path.dirname(__file__), "pdb")
os.makedirs(PDB_DIR, exist_ok=True)

# Target list: (PDB_ID, metal, protein, ligand_name, ki_nM, note)
TARGETS = [
    # CAII — Zn²⁺
    ("1OQ5", "Zn", "CAII",    "Acetazolamide",              9,    "FDA drug; standard benchmark"),
    ("1BNN", "Zn", "CAII",    "p-aminobenzenesulfonamide",  380,  "Simple sulfonamide"),
    ("1CNX", "Zn", "CAII",    "4-methylbenzenesulfonamide", 540,  "Methyl derivative"),
    ("1ZNC", "Zn", "CAII",    "Sulfonamide inhibitor",      200,  "Common benchmark"),
    ("1A42", "Zn", "CAII",    "Acetazolamide analog",       15,   "High affinity"),
    ("1YDB", "Zn", "CAII",    "Fluorinated sulfonamide",    2,    "Very high affinity"),
    ("3HS4", "Zn", "CAII",    "Coumarin inhibitor",         8300, "Non-sulfonamide scaffold"),
    ("2NWP", "Zn", "CAII",    "Glycosyl sulfonamide",       430,  "Large ligand"),
    ("1BV3", "Zn", "CAII",    "Sulfonamide",                150,  "Medium affinity"),
    ("2CBD", "Zn", "CAII",    "Benzenesulfonamide",         900,  "Basic sulfonamide"),
    # MMP — Zn²⁺
    ("1HOV", "Zn", "MMP-2",   "Hydroxamate inhibitor",      0.3,  "Ultra-high affinity"),
    ("1GKC", "Zn", "MMP-2",   "Phosphonate inhibitor",      27,   "Non-hydroxamate"),
    ("1QIB", "Zn", "MMP-9",   "CGS27023A",                  26,   "MMP-9 specific"),
    ("1MMQ", "Zn", "MMP-1",   "Peptide hydroxamate",        670,  "Low selectivity"),
    ("1SLN", "Zn", "MMP-8",   "Batimastat analog",          4,    "Broad MMP inhibitor"),
    # ACE — Zn²⁺
    ("1O86", "Zn", "ACE",     "Captopril",                  1.7,  "First ACE inhibitor; FDA"),
    ("2C6N", "Zn", "ACE",     "Lisinopril",                 0.27, "FDA; highest affinity"),
    ("1UZE", "Zn", "ACE",     "Enalaprilat",                2.2,  "FDA metabolite"),
    # HIV Integrase — Mg²⁺
    ("3L2U", "Mg", "HIV-IN",  "Raltegravir",                2.0,  "FDA; first IN inhibitor"),
    ("2B4J", "Mg", "HIV-IN",  "Diketo acid",                30,   "Early IN inhibitor"),
    # PHD2 — Fe²⁺
    ("3OXM", "Fe", "PHD2",    "N-oxalylglycine analog",     400,  "NOG analog"),
    ("4BKS", "Fe", "PHD2",    "GSK1278863 analog",          1500, "Clinical candidate"),
]

RCSB_URL = "https://files.rcsb.org/download/{}.pdb"

def download_pdb(pdb_id):
    out_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        if size > 1000:
            print(f"  [skip] {pdb_id}.pdb already exists ({size//1024} KB)")
            return True

    url = RCSB_URL.format(pdb_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  [OK]   {pdb_id}.pdb  ({len(data)//1024} KB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"  [ERR]  {pdb_id}: HTTP {e.code}")
        return False
    except Exception as e:
        print(f"  [ERR]  {pdb_id}: {e}")
        return False

def print_summary():
    print("\n" + "="*60)
    print(f"{'PDB':6} {'Metal':5} {'Protein':8} {'Ki(nM)':>10}  Ligand")
    print("-"*60)
    for pdb_id, metal, protein, ligand, ki, note in TARGETS:
        path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        status = "✓" if (os.path.exists(path) and os.path.getsize(path) > 1000) else "✗"
        print(f"{status} {pdb_id:6} {metal:5} {protein:8} {ki:>10.1f}  {ligand}")
    print("="*60)

def main():
    print(f"Downloading {len(TARGETS)} PDB structures to {PDB_DIR}/\n")

    groups = {}
    for t in TARGETS:
        metal = t[1]
        groups.setdefault(metal, []).append(t)

    ok = fail = 0
    for metal, entries in groups.items():
        print(f"\n── {metal}²⁺ structures ({len(entries)}) ──")
        for pdb_id, *rest in entries:
            if download_pdb(pdb_id):
                ok += 1
            else:
                fail += 1
            time.sleep(0.3)  # polite rate limiting

    print_summary()
    print(f"\nDone: {ok} downloaded, {fail} failed")

    if fail > 0:
        print("\nFailed structures — retry manually:")
        for pdb_id, metal, protein, ligand, ki, note in TARGETS:
            path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
            if not os.path.exists(path) or os.path.getsize(path) <= 1000:
                print(f"  wget https://files.rcsb.org/download/{pdb_id}.pdb -O pdb/{pdb_id}.pdb")

if __name__ == "__main__":
    main()
