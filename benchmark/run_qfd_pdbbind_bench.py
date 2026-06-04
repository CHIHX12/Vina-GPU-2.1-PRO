#!/usr/bin/env python3
"""
run_qfd_pdbbind_bench.py — QFD consensus ranking benchmark on PDBbind targets.

For each target:
  1. Generate QFD grids in out_dir (prep_qfd_grids.py)
  2. Re-dock with Vina-GPU 2.1, cwd=out_dir so qfd_esp.bin is found
  3. Parse log for per-pose rQ / cns
  4. Compute per-pose RMSD to crystal
  5. Compare: Vina rank-1 vs cns rank-1 vs rQ rank-1

Usage:
  python3 run_qfd_pdbbind_bench.py --ids /tmp/qfd_bench_ids.txt
  python3 run_qfd_pdbbind_bench.py --ids /tmp/qfd_bench_ids.txt --workers 8
  python3 run_qfd_pdbbind_bench.py --ids /tmp/qfd_bench_ids.txt --resume
"""

from __future__ import annotations
import argparse, csv, math, multiprocessing as mp, os, re, subprocess, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
VINA_DIR    = Path("/home/cycheng/Vina-GPU-2.1")
VINA_BIN    = VINA_DIR / "AutoDock-Vina-GPU-2-1"
PREP_QFD    = VINA_DIR / "tools/prep/prep_qfd_grids.py"
PDBBIND_VAL = Path("/home/cycheng/LigandScope/validation/pdbbind")
PYTHON      = sys.executable


# ── GPU assignment ────────────────────────────────────────────────────────────
_WORKER_GPU_ID: int = 0

def _gpu_worker_init(n_gpus: int) -> None:
    global _WORKER_GPU_ID
    try:
        worker_idx = mp.current_process()._identity[0] - 1
    except (AttributeError, IndexError):
        worker_idx = 0
    _WORKER_GPU_ID = worker_idx % n_gpus


# ── Config parser ─────────────────────────────────────────────────────────────
def parse_config(cfg_path: Path) -> Dict[str, str]:
    vals: Dict[str, str] = {}
    for line in cfg_path.read_text().splitlines():
        m = re.match(r'\s*(\w+)\s*=\s*(.+)', line)
        if m:
            vals[m.group(1).strip()] = m.group(2).strip()
    return vals


# ── RMSD ──────────────────────────────────────────────────────────────────────
def _pdbqt_heavy(path: Path, model: int = 1) -> List[Tuple[float, float, float]]:
    coords: List[Tuple[float, float, float]] = []
    has_model = False; cur_model = 0; in_target = False
    with path.open(errors="ignore") as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "MODEL":
                has_model = True; cur_model += 1
                in_target = (cur_model == model); continue
            if rec == "ENDMDL":
                if in_target: break
                in_target = False; continue
            if (in_target or (not has_model and model == 1)) and rec in ("ATOM", "HETATM"):
                ad4 = line[77:79].strip() if len(line) > 77 else ""
                if ad4 in ("H", "HD", "HS"): continue
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except (ValueError, IndexError):
                    pass
    return coords


def idx_rmsd(ref: List[Tuple[float, float, float]],
             mob: List[Tuple[float, float, float]]) -> Optional[float]:
    n = min(len(ref), len(mob))
    if n < 3: return None
    return math.sqrt(sum((ref[i][0]-mob[i][0])**2 + (ref[i][1]-mob[i][1])**2 +
                         (ref[i][2]-mob[i][2])**2 for i in range(n)) / n)


# ── Log parser: extract rQ/cns per mode ──────────────────────────────────────
def parse_qfd_log(log_path: Path) -> List[Dict]:
    """Returns list of {mode, affinity, rQ, cns} dicts from QFD table."""
    poses = []
    in_table = False
    for line in log_path.read_text(errors="ignore").splitlines():
        if "rQ" in line and "cns" in line and "affinity" in line:
            in_table = True; continue
        if in_table:
            # Skip separator and unit lines: "---...", "     | (kcal/mol)..."
            stripped = line.strip()
            if not stripped or stripped.startswith("---") or stripped.startswith("|"):
                continue
            # mode line: "   1        -11.3   5   3.0      0.000      0.000"
            m = re.match(r'\s*(\d+)\s+([-\d.]+)\s+(\d+)\s+([\d.]+)', line)
            if m:
                poses.append({
                    "mode":     int(m.group(1)),
                    "affinity": float(m.group(2)),
                    "rQ":       int(m.group(3)),
                    "cns":      float(m.group(4)),
                })
            elif stripped and not stripped.startswith("Writing"):
                break  # end of table
    return poses


def parse_vina_log(log_path: Path) -> List[Dict]:
    """Returns list of {mode, affinity} dicts from plain Vina table (no QFD)."""
    poses = []
    in_table = False
    for line in log_path.read_text(errors="ignore").splitlines():
        if "affinity" in line and "rmsd" in line.lower():
            in_table = True; continue
        if in_table:
            if line.startswith("---"): continue
            m = re.match(r'\s*(\d+)\s+([-\d.]+)', line)
            if m:
                poses.append({"mode": int(m.group(1)), "affinity": float(m.group(2))})
            elif line.strip() and not line.startswith("Writing"):
                break
    return poses


# ── Per-entry pipeline ────────────────────────────────────────────────────────
def process_entry(args: Tuple) -> Dict:
    (pdbid, resume, search_depth, thread) = args
    gpu_id = _WORKER_GPU_ID   # assigned by _gpu_worker_init

    result: Dict = {
        "pdbid":          pdbid,
        "status":         "OK",
        "qfd_active":     False,
        "n_poses":        0,
        "vina1_rmsd":     None,   # Vina rank-1 RMSD
        "cns1_rmsd":      None,   # cns rank-1 RMSD
        "rq1_rmsd":       None,   # rQ rank-1 RMSD
        "best_rmsd":      None,   # best any-mode RMSD
        "vina1_cns":      None,
        "error": "",
    }

    out_dir     = PDBBIND_VAL / pdbid
    rec_pdbqt   = out_dir / "receptor.pdbqt"
    lig_pdbqt   = out_dir / "ligand.pdbqt"
    cfg_orig    = out_dir / "config.txt"
    qfd_out     = out_dir / "vina_qfd_out.pdbqt"
    qfd_log     = out_dir / "vina_qfd_log.txt"
    cfg_qfd     = out_dir / "config_qfd.txt"

    if not rec_pdbqt.exists() or not lig_pdbqt.exists():
        result.update(status="SKIP_PREP", error="receptor/ligand.pdbqt missing"); return result

    if not cfg_orig.exists():
        result.update(status="SKIP_CFG", error="config.txt missing"); return result

    # ── 1. QFD grid generation ────────────────────────────────────────────────
    qfd_esp = out_dir / "qfd_esp.bin"
    if not resume or not qfd_esp.exists():
        cfg_vals = parse_config(cfg_orig)
        required = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
        if not all(k in cfg_vals for k in required):
            result.update(status="FAIL_CFG", error="missing box params"); return result

        cmd = [PYTHON, str(PREP_QFD),
               "--receptor",  str(rec_pdbqt),
               "--center_x",  cfg_vals["center_x"],
               "--center_y",  cfg_vals["center_y"],
               "--center_z",  cfg_vals["center_z"],
               "--size_x",    cfg_vals["size_x"],
               "--size_y",    cfg_vals["size_y"],
               "--size_z",    cfg_vals["size_z"],
               "--output_dir", str(out_dir)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except Exception as e:
            result.update(status="FAIL_QFD_GRID", error=str(e)); return result

        if not qfd_esp.exists():
            result.update(status="FAIL_QFD_GRID", error="qfd_esp.bin not created"); return result

    # ── 2. Dock with QFD (cwd=out_dir so Vina finds qfd_esp.bin) ─────────────
    if not resume or not qfd_out.exists():
        cfg_orig_vals = parse_config(cfg_orig)
        cfg_qfd.write_text("\n".join([
            f"receptor           = {rec_pdbqt}",
            f"ligand             = {lig_pdbqt}",
            f"center_x           = {cfg_orig_vals['center_x']}",
            f"center_y           = {cfg_orig_vals['center_y']}",
            f"center_z           = {cfg_orig_vals['center_z']}",
            f"size_x             = {cfg_orig_vals['size_x']}",
            f"size_y             = {cfg_orig_vals['size_y']}",
            f"size_z             = {cfg_orig_vals['size_z']}",
            f"num_modes          = 10",
            f"search_depth       = {search_depth}",
            f"thread             = {thread}",
            f"out                = {qfd_out}",
            f"opencl_binary_path = {VINA_DIR}",
            f"gpu_id             = {gpu_id}",
        ]))
        env = {**os.environ, "VINA_GPU_HOME": str(VINA_DIR)}
        try:
            r = subprocess.run(
                [str(VINA_BIN), "--config", str(cfg_qfd)],
                capture_output=True, text=True, timeout=300,
                cwd=str(out_dir), env=env)   # ← cwd=out_dir: Vina finds qfd_esp.bin
            qfd_log.write_text(r.stdout + r.stderr)
        except Exception as e:
            result.update(status="FAIL_DOCK", error=str(e)); return result

        if not qfd_out.exists():
            result.update(status="FAIL_DOCK", error="Vina produced no output"); return result

    # ── 3. Parse log for rQ / cns ─────────────────────────────────────────────
    if not qfd_log.exists():
        result.update(status="FAIL_LOG", error="qfd_log missing"); return result

    poses = parse_qfd_log(qfd_log)
    qfd_active = len(poses) > 0 and all("rQ" in p for p in poses)

    if not qfd_active:
        # fallback: plain Vina log (QFD grids not found — shouldn't happen)
        poses = parse_vina_log(qfd_log)

    result["qfd_active"] = qfd_active
    result["n_poses"]    = len(poses)

    if not poses:
        result.update(status="FAIL_PARSE", error="no poses in log"); return result

    # ── 4. Per-pose RMSD ──────────────────────────────────────────────────────
    ref_coords = _pdbqt_heavy(lig_pdbqt)
    if len(ref_coords) < 3:
        result.update(status="FAIL_REF", error="crystal ligand too small"); return result

    pose_rmsds: List[Optional[float]] = []
    for p in poses:
        mob = _pdbqt_heavy(qfd_out, model=p["mode"])
        pose_rmsds.append(idx_rmsd(ref_coords, mob))

    # ── 5. Selection strategies ───────────────────────────────────────────────
    # Vina rank-1 = mode 1 (always index 0 in poses list)
    vina1_idx = 0
    result["vina1_rmsd"] = pose_rmsds[vina1_idx]

    if qfd_active:
        # cns rank-1: mode with lowest cns
        cns1_idx = min(range(len(poses)), key=lambda i: poses[i]["cns"])
        result["cns1_rmsd"]  = pose_rmsds[cns1_idx]
        result["vina1_cns"]  = poses[vina1_idx]["cns"]

        # rQ rank-1: mode with rQ=1
        rq1_idx = next((i for i, p in enumerate(poses) if p["rQ"] == 1), 0)
        result["rq1_rmsd"] = pose_rmsds[rq1_idx]

    # best any mode
    valid = [(r, i) for i, r in enumerate(pose_rmsds) if r is not None]
    if valid:
        best_r, best_i = min(valid)
        result["best_rmsd"] = best_r

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="QFD consensus ranking benchmark on PDBbind salvageable targets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ids",     required=True, metavar="FILE",
                        help="File with one PDB ID per line")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (default: n_gpus)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip targets where vina_qfd_out.pdbqt exists")
    parser.add_argument("--search-depth", type=int, default=10)
    parser.add_argument("--thread",  type=int, default=8000)
    parser.add_argument("--tsv",     metavar="FILE",
                        help="Write TSV results")
    args = parser.parse_args()

    # Detect GPUs
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        n_gpus = len([l for l in r.stdout.strip().splitlines() if l.strip()])
    except Exception:
        n_gpus = 1
    n_gpus = max(n_gpus, 1)
    n_workers = args.workers or n_gpus

    # Load IDs
    id_path = Path(args.ids)
    ids = [l.strip() for l in id_path.read_text().splitlines()
           if l.strip() and not l.startswith("#")]
    print(f"Loaded {len(ids)} IDs from {args.ids}")
    print(f"Engine: Vina-GPU 2.1  ×{n_workers} workers  ({n_gpus} GPU(s))")
    print(f"search_depth={args.search_depth}  thread={args.thread}\n")

    tasks = [(pdbid, args.resume, args.search_depth, args.thread)
             for pdbid in ids]

    results: List[Dict] = []
    t_start = time.time()
    n_total = len(tasks)

    with mp.Pool(processes=n_workers,
                 initializer=_gpu_worker_init, initargs=(n_gpus,)) as pool:
        for i, r in enumerate(pool.imap_unordered(process_entry, tasks), 1):
            results.append(r)
            status = r["status"]
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta_m = (n_total - i) / rate / 60 if rate > 0 else 0

            v1 = r.get("vina1_rmsd")
            c1 = r.get("cns1_rmsd")
            rq1 = r.get("rq1_rmsd")
            best = r.get("best_rmsd")
            if v1 is not None:
                v1s = f"{v1:.2f}Å{'✓' if v1<2 else '✗'}"
                c1s = f"{c1:.2f}Å{'✓' if c1<2 else '✗'}" if c1 is not None else "N/A"
                rq1s = f"{rq1:.2f}Å{'✓' if rq1<2 else '✗'}" if rq1 is not None else "N/A"
                bests = f"{best:.2f}Å" if best is not None else "N/A"
                qf = "QFD" if r.get("qfd_active") else "NoQ"
                line = (f"[{i:>4}/{n_total}] {r['pdbid']:6s}  "
                        f"{qf}  vina1={v1s}  cns1={c1s}  rq1={rq1s}  best={bests}  "
                        f"ETA {eta_m:.0f}min")
            else:
                line = f"[{i:>4}/{n_total}] {r['pdbid']:6s}  {status}  {r.get('error','')}  ETA {eta_m:.0f}min"
            print(line, flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    ok = [r for r in results if r["status"] == "OK"]
    qfd = [r for r in ok if r.get("qfd_active")]

    def _pass(rs, key):
        valid = [r for r in rs if r.get(key) is not None]
        n_pass = sum(1 for r in valid if r[key] < 2.0)
        return n_pass, len(valid)

    print(f"\n{'='*72}")
    print(f"  Targets            : {len(results):>6,}")
    print(f"  OK / QFD-active    : {len(ok):>6,} / {len(qfd):>6,}")
    print(f"  Elapsed            : {elapsed_total/60:>6.1f} min")
    print()
    if qfd:
        n_v1,  tot_v1  = _pass(qfd, "vina1_rmsd")
        n_c1,  tot_c1  = _pass(qfd, "cns1_rmsd")
        n_rq1, tot_rq1 = _pass(qfd, "rq1_rmsd")
        n_best,tot_best = _pass(qfd, "best_rmsd")
        print(f"  Strategy          | RMSD<2Å  | Rate")
        print(f"  ------------------+----------+-------")
        print(f"  Vina rank-1       | {n_v1:>4}/{tot_v1:<4} | {n_v1/tot_v1*100:.1f}%")
        print(f"  cns rank-1        | {n_c1:>4}/{tot_c1:<4} | {n_c1/tot_c1*100:.1f}%")
        print(f"  rQ rank-1         | {n_rq1:>4}/{tot_rq1:<4} | {n_rq1/tot_rq1*100:.1f}%")
        print(f"  Best-of-all poses | {n_best:>4}/{tot_best:<4} | {n_best/tot_best*100:.1f}%")

        # Delta
        print()
        print(f"  Δ cns vs Vina rank-1 : {n_c1-n_v1:+d} targets  ({(n_c1-n_v1)/tot_v1*100:+.1f}%)")
        print(f"  Δ rQ  vs Vina rank-1 : {n_rq1-n_v1:+d} targets  ({(n_rq1-n_v1)/tot_rq1*100:+.1f}%)")
    print(f"{'='*72}")

    # ── TSV ───────────────────────────────────────────────────────────────────
    if args.tsv:
        tsv_p = Path(args.tsv)
        tsv_p.parent.mkdir(parents=True, exist_ok=True)
        fields = ["pdbid","status","qfd_active","n_poses",
                  "vina1_rmsd","cns1_rmsd","rq1_rmsd","best_rmsd","vina1_cns","error"]
        with tsv_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", delimiter="\t")
            w.writeheader()
            for r in sorted(results, key=lambda x: x["pdbid"]):
                w.writerow(r)
        print(f"\nTSV → {tsv_p}")


if __name__ == "__main__":
    main()
