#!/usr/bin/env python3
"""
gen_qfd_grids_pdbbind.py — Pre-generate QFD grids for all PDBbind targets.

Runs prep_qfd_grids.py in parallel (CPU-only) so the subsequent docking
run (run_qfd_pdbbind_bench.py --resume) can skip grid generation and focus
on GPU throughput.

Usage:
    python3 gen_qfd_grids_pdbbind.py                    # all 18K targets
    python3 gen_qfd_grids_pdbbind.py --workers 16       # explicit workers
    python3 gen_qfd_grids_pdbbind.py --resume           # skip if qfd_esp.bin exists
    python3 gen_qfd_grids_pdbbind.py --ids ids.txt      # custom target list
"""

from __future__ import annotations
import argparse, re, subprocess, sys, time
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

VINA_DIR    = Path("/home/cycheng/Vina-GPU-2.1")
PREP_QFD    = VINA_DIR / "tools/prep/prep_qfd_grids.py"
PDBBIND_VAL = Path("/home/cycheng/LigandScope/validation/pdbbind")
PYTHON      = sys.executable


def parse_config(cfg_path: Path) -> dict:
    vals = {}
    for line in cfg_path.read_text().splitlines():
        m = re.match(r'\s*(\w+)\s*=\s*(.+)', line)
        if m:
            vals[m.group(1).strip()] = m.group(2).strip()
    return vals


def gen_grids(args: Tuple) -> dict:
    pdbid, resume = args
    out_dir  = PDBBIND_VAL / pdbid
    esp_bin  = out_dir / "qfd_esp.bin"
    pipi_bin = out_dir / "qfd_pipi.bin"

    if resume and esp_bin.exists() and pipi_bin.exists():
        return {"pdbid": pdbid, "status": "SKIP"}

    cfg_path = out_dir / "config.txt"
    rec_path = out_dir / "receptor.pdbqt"
    if not cfg_path.exists() or not rec_path.exists():
        return {"pdbid": pdbid, "status": "MISSING"}

    cfg = parse_config(cfg_path)
    required = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
    if not all(k in cfg for k in required):
        return {"pdbid": pdbid, "status": "BAD_CFG"}

    cmd = [
        PYTHON, str(PREP_QFD),
        "--receptor",  str(rec_path),
        "--center_x",  cfg["center_x"],
        "--center_y",  cfg["center_y"],
        "--center_z",  cfg["center_z"],
        "--size_x",    cfg["size_x"],
        "--size_y",    cfg["size_y"],
        "--size_z",    cfg["size_z"],
        "--output_dir", str(out_dir),
        "--no_water",   # no crystal PDB for water grid
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0 or not pipi_bin.exists():
            return {"pdbid": pdbid, "status": "FAIL",
                    "err": r.stderr.decode(errors="ignore")[:200]}
        return {"pdbid": pdbid, "status": "OK"}
    except Exception as e:
        return {"pdbid": pdbid, "status": "FAIL", "err": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-generate QFD grids for PDBbind targets")
    ap.add_argument("--ids",     metavar="FILE",
                    help="File with one PDB ID per line (default: all in validation/pdbbind)")
    ap.add_argument("--workers", type=int, default=16,
                    help="CPU parallel workers (default: 16)")
    ap.add_argument("--resume",  action="store_true",
                    help="Skip targets where qfd_esp.bin + qfd_pipi.bin already exist")
    args = ap.parse_args()

    if args.ids:
        ids = [l.strip() for l in Path(args.ids).read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    else:
        ids = sorted(d.name for d in PDBBIND_VAL.iterdir()
                     if d.is_dir() and (d/"config.txt").exists())

    total = len(ids)
    print(f"Targets: {total:,}   Workers: {args.workers}   Resume: {args.resume}")
    print(f"Estimated time: {total * 3.5 / args.workers / 3600:.1f} h\n")

    tasks = [(pdbid, args.resume) for pdbid in ids]
    t0 = time.time()

    counts = {"OK": 0, "SKIP": 0, "FAIL": 0, "MISSING": 0, "BAD_CFG": 0}
    with mp.Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(gen_grids, tasks), 1):
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "FAIL":
                print(f"  FAIL {r['pdbid']}: {r.get('err','')}", flush=True)
            if i % 500 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (total - i) / rate / 60
                pct = 100 * counts["OK"] / max(i - counts["SKIP"], 1)
                print(f"[{i:>6}/{total}] OK={counts['OK']} SKIP={counts['SKIP']} "
                      f"FAIL={counts['FAIL']}  rate={rate:.1f}/s  ETA={eta:.0f}min  "
                      f"success={pct:.1f}%", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed/60:.1f} min")
    for k, v in sorted(counts.items()):
        print(f"  {k:10s}: {v:>6,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
