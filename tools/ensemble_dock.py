#!/usr/bin/env python3
"""
ensemble_dock.py — Multi-seed ensemble docking with Vina-GPU 2.1.

Runs Vina-GPU N times (each is a separate GPU launch, which uses different
internal random trajectories due to GPU thread-clock non-determinism), pools
all poses, then RMSD-clusters them to return top-K diverse unique poses.

Usage:
  python3 tools/ensemble_dock.py \\
      --receptor receptor.pdbqt --ligand ligand.pdbqt \\
      --center_x 10 --center_y 20 --center_z 5 \\
      --size_x 25 --size_y 25 --size_z 25 \\
      --n_seeds 5 --top_n 10 --out ligand_ensemble.pdbqt
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VINA_DIR = Path("/home/cycheng/Vina-GPU-2.1")
VINA_BIN = VINA_DIR / "AutoDock-Vina-GPU-2-1"
OCL_CACHE = VINA_DIR  # Kernel*.bin live here


# ---------------------------------------------------------------------------
# PDBQT parsing helpers
# ---------------------------------------------------------------------------

def _pdbqt_heavy_coords(path: Path, model: int = 1) -> list[tuple[float, float, float]]:
    """Return heavy-atom (x,y,z) list for MODEL *model* (1-based) in *path*."""
    coords: list[tuple[float, float, float]] = []
    has_model = False
    cur_model = 0
    in_target = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                has_model = True
                cur_model += 1
                in_target = cur_model == model
                continue
            if rec == "ENDMDL":
                if in_target:
                    break
                in_target = False
                continue
            if (in_target or (not has_model and model == 1)) and rec in ("ATOM", "HETATM"):
                ad4 = line[77:79].strip() if len(line) > 77 else ""
                if ad4 in ("H", "HD", "HS"):
                    continue
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    pass
    return coords


def _count_models(path: Path) -> int:
    """Count MODEL records; returns 1 if none found (single-model file)."""
    n = 0
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line[:5] == "MODEL":
                n += 1
    return n if n > 0 else 1


def _parse_energy(path: Path, model: int) -> float:
    """Return the VINA RESULT energy for MODEL *model* (1-based)."""
    cur_model = 0
    has_model = False
    in_target = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                has_model = True
                cur_model += 1
                in_target = cur_model == model
                continue
            if rec == "ENDMDL":
                in_target = False
                continue
            if (in_target or (not has_model and model == 1)) and "VINA RESULT" in line:
                try:
                    return float(line.split()[3])
                except (IndexError, ValueError):
                    pass
    return 999.0


def _extract_model_lines(path: Path, model: int) -> list[str]:
    """Return all lines (including MODEL/ENDMDL) for MODEL *model* (1-based)."""
    lines: list[str] = []
    cur_model = 0
    has_model = False
    in_target = False
    with open(path, errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "MODEL":
                has_model = True
                cur_model += 1
                in_target = cur_model == model
                if in_target:
                    lines.append(line)
                continue
            if rec == "ENDMDL":
                if in_target:
                    lines.append(line)
                    break
                in_target = False
                continue
            if in_target:
                lines.append(line)
            elif not has_model and model == 1:
                lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# RMSD and clustering
# ---------------------------------------------------------------------------

def rmsd(c1: list, c2: list) -> float:
    n = min(len(c1), len(c2))
    if n < 3:
        return 999.0
    total = sum(
        (c1[i][0] - c2[i][0]) ** 2 +
        (c1[i][1] - c2[i][1]) ** 2 +
        (c1[i][2] - c2[i][2]) ** 2
        for i in range(n)
    )
    return math.sqrt(total / n)


def greedy_cluster(poses: list, cutoff: float, top_n: int) -> list:
    """
    Greedy RMSD clustering.

    *poses* is a list of dicts with keys: energy, coords, run_i, model_j, out_path.
    Returns list of cluster-centre dicts, sorted by energy, length <= top_n.
    """
    sorted_poses = sorted(poses, key=lambda p: p["energy"])
    centers: list = []
    for p in sorted_poses:
        if all(rmsd(p["coords"], c["coords"]) >= cutoff for c in centers):
            # Record RMSD from previous best cluster centre for display
            if centers:
                nearest = min(rmsd(p["coords"], c["coords"]) for c in centers)
                p["nearest_rmsd"] = nearest
            else:
                p["nearest_rmsd"] = None
            centers.append(p)
            if len(centers) >= top_n:
                break
    return centers


# ---------------------------------------------------------------------------
# QFD grid detection
# ---------------------------------------------------------------------------

def _find_qfd_grids(receptor_path: Path) -> list[Path]:
    """Return QFD grid paths if they exist next to the receptor, else []."""
    parent = receptor_path.parent
    names = ["qfd_esp.bin", "qfd_desolv.bin", "qfd_infomap.bin"]
    paths = [parent / n for n in names]
    if all(p.exists() for p in paths):
        return paths
    return []


# ---------------------------------------------------------------------------
# Vina-GPU runner
# ---------------------------------------------------------------------------

def _detect_n_gpus() -> int:
    """Return number of NVIDIA GPUs via nvidia-smi, default 1 on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        n = len([l for l in out.strip().splitlines() if l.strip()])
        return max(n, 1)
    except Exception:
        return 1


def run_vina(
    run_idx: int,
    receptor: Path,
    ligand: Path,
    box: dict,
    workdir: Path,
    n_poses: int,
    search_depth: int,
    thread: int,
    gpu_id: int,
    seed: int | None,
    verbose: bool,
    qfd_grids: list[Path],
) -> Path | None:
    """
    Run Vina-GPU for one seed. Returns path to output PDBQT or None on failure.
    """
    run_dir = workdir / f"run_{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)

    out_pdbqt = run_dir / "vina_out.pdbqt"
    log_path = run_dir / "vina.log"

    # Copy QFD grids into run dir if present
    for gf in qfd_grids:
        dst = run_dir / gf.name
        if not dst.exists():
            shutil.copy2(gf, dst)

    # Write config
    cfg_lines = [
        f"receptor           = {receptor.resolve()}",
        f"ligand             = {ligand.resolve()}",
        f"out                = {out_pdbqt.resolve()}",
        f"center_x           = {box['cx']}",
        f"center_y           = {box['cy']}",
        f"center_z           = {box['cz']}",
        f"size_x             = {box['sx']}",
        f"size_y             = {box['sy']}",
        f"size_z             = {box['sz']}",
        f"num_modes          = {n_poses}",
        f"search_depth       = {search_depth}",
        f"thread             = {thread}",
        f"gpu_id             = {gpu_id}",
        f"opencl_binary_path = {OCL_CACHE}",
    ]
    if seed is not None:
        cfg_lines.append(f"seed               = {seed}")

    cfg_path = run_dir / "config.txt"
    cfg_path.write_text("\n".join(cfg_lines) + "\n")

    cmd = [str(VINA_BIN), "--config", str(cfg_path)]
    env = os.environ.copy()
    env["VINA_GPU_HOME"] = str(VINA_DIR)

    if verbose:
        print(f"[run_{run_idx}] Starting on GPU {gpu_id} (seed={seed}) ...")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(run_dir),
            env=env,
            capture_output=not verbose,
            text=True,
            timeout=600,
        )
        log_path.write_text(
            (result.stdout or "") + "\n" + (result.stderr or "")
        )
        if not out_pdbqt.exists():
            print(
                f"[WARNING] run_{run_idx}: Vina-GPU finished but output not found "
                f"(exit {result.returncode}). Check {log_path}",
                file=sys.stderr,
            )
            return None
        if verbose:
            print(f"[run_{run_idx}] Done — {out_pdbqt}")
        return out_pdbqt
    except subprocess.TimeoutExpired:
        print(f"[WARNING] run_{run_idx}: timed out after 600 s.", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[WARNING] run_{run_idx}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_ensemble(clusters: list, out_path: Path, verbose: bool) -> None:
    """Write renumbered ensemble PDBQT with REMARK headers per cluster."""
    with open(out_path, "w") as fh:
        fh.write("REMARK  Ensemble docking output — Vina-GPU 2.1\n")
        for rank, c in enumerate(clusters, start=1):
            fh.write(
                f"REMARK ENSEMBLE rank={rank} "
                f"seed_run={c['run_i']} "
                f"orig_model={c['model_j']} "
                f"energy={c['energy']:.2f}\n"
            )
            raw_lines = _extract_model_lines(c["out_path"], c["model_j"])
            # Replace the MODEL line with the new rank number
            for line in raw_lines:
                if line[:5] == "MODEL":
                    fh.write(f"MODEL {rank:>8}\n")
                else:
                    fh.write(line)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _parse_box_from_config(config_path: Path) -> dict:
    """Parse center_x/y/z and size_x/y/z from a Vina config file."""
    box: dict = {}
    key_map = {
        "center_x": "cx", "center_y": "cy", "center_z": "cz",
        "size_x": "sx", "size_y": "sy", "size_z": "sz",
    }
    with open(config_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.split("#")[0].strip()
                if k in key_map:
                    try:
                        box[key_map[k]] = float(v)
                    except ValueError:
                        pass
    return box


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-seed ensemble docking with Vina-GPU 2.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--receptor", required=True, help="Receptor PDBQT")
    p.add_argument("--ligand", required=True, help="Ligand PDBQT")
    p.add_argument("--config", default=None, help="Optional Vina config file (overrides center/size)")
    p.add_argument("--center_x", type=float, default=None)
    p.add_argument("--center_y", type=float, default=None)
    p.add_argument("--center_z", type=float, default=None)
    p.add_argument("--size_x", type=float, default=None)
    p.add_argument("--size_y", type=float, default=None)
    p.add_argument("--size_z", type=float, default=None)
    p.add_argument("--out", default=None, help="Output PDBQT (default: <ligand_stem>_ensemble.pdbqt)")
    p.add_argument("--n_seeds", type=int, default=5, help="Number of independent Vina-GPU runs")
    p.add_argument("--poses_per_seed", type=int, default=10, help="num_modes per run")
    p.add_argument("--rmsd_cutoff", type=float, default=2.0, help="RMSD cutoff for clustering (Å)")
    p.add_argument("--top_n", type=int, default=10, help="Max unique poses to output")
    p.add_argument("--search_depth", type=int, default=10, help="MC+BFGS rounds per trajectory")
    p.add_argument("--thread", type=int, default=8000, help="OpenCL threads")
    p.add_argument("--gpu_id", type=int, default=0, help="Base GPU device index")
    p.add_argument("--base_seed", type=int, default=None,
                   help="If given, pass explicit seeds: base_seed + i*10000 per run")
    p.add_argument("--workdir", default=None, help="Working directory (default: temp, auto-cleaned)")
    p.add_argument("--keep_workdir", action="store_true", help="Do not delete workdir on exit")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    receptor = Path(args.receptor).resolve()
    ligand = Path(args.ligand).resolve()

    if not receptor.exists():
        sys.exit(f"ERROR: receptor not found: {receptor}")
    if not ligand.exists():
        sys.exit(f"ERROR: ligand not found: {ligand}")
    if not VINA_BIN.exists():
        sys.exit(f"ERROR: Vina-GPU binary not found: {VINA_BIN}")

    # Resolve box parameters
    box: dict = {}
    if args.config:
        cfg_path = Path(args.config).resolve()
        if not cfg_path.exists():
            sys.exit(f"ERROR: config not found: {cfg_path}")
        box = _parse_box_from_config(cfg_path)

    # CLI values override config file values
    for attr, key in [
        ("center_x", "cx"), ("center_y", "cy"), ("center_z", "cz"),
        ("size_x", "sx"), ("size_y", "sy"), ("size_z", "sz"),
    ]:
        v = getattr(args, attr)
        if v is not None:
            box[key] = v

    missing = [k for k in ("cx", "cy", "cz", "sx", "sy", "sz") if k not in box]
    if missing:
        sys.exit(
            f"ERROR: Box parameters not fully specified. Missing: {missing}. "
            "Provide --center_x/y/z and --size_x/y/z (or --config with those keys)."
        )

    out_path = Path(args.out) if args.out else ligand.parent / f"{ligand.stem}_ensemble.pdbqt"

    # Detect GPUs and set up parallel workers
    n_gpus = _detect_n_gpus()
    if args.verbose:
        print(f"Detected {n_gpus} GPU(s)")

    # Check for QFD grids next to receptor
    qfd_grids = _find_qfd_grids(receptor)
    if qfd_grids and args.verbose:
        print(f"QFD grids found — will copy into each run subdir: {[g.name for g in qfd_grids]}")

    # Set up workdir
    _tmp_owned = False
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="ensemble_dock_"))
        _tmp_owned = True

    if args.verbose:
        print(f"Workdir: {workdir}")

    try:
        _run_ensemble(
            args, receptor, ligand, box, out_path, workdir,
            n_gpus, qfd_grids,
        )
    finally:
        if _tmp_owned and not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        elif args.keep_workdir:
            print(f"Workdir retained: {workdir}")


def _run_ensemble(args, receptor, ligand, box, out_path, workdir, n_gpus, qfd_grids):
    n_seeds = args.n_seeds

    print(
        f"Ensemble docking: {n_seeds} seeds × {args.poses_per_seed} poses/seed "
        f"= up to {n_seeds * args.poses_per_seed} total poses"
    )

    # Launch runs in parallel (max 2 concurrent to avoid GPU contention)
    max_workers = min(n_seeds, max(n_gpus, 2))
    futures = {}
    results: dict[int, Path | None] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i in range(n_seeds):
            gpu_id = args.gpu_id + (i % n_gpus)
            seed = (args.base_seed + i * 10000) if args.base_seed is not None else None
            fut = pool.submit(
                run_vina,
                run_idx=i,
                receptor=receptor,
                ligand=ligand,
                box=box,
                workdir=workdir,
                n_poses=args.poses_per_seed,
                search_depth=args.search_depth,
                thread=args.thread,
                gpu_id=gpu_id,
                seed=seed,
                verbose=args.verbose,
                qfd_grids=qfd_grids,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            run_idx = futures[fut]
            try:
                results[run_idx] = fut.result()
            except Exception as exc:
                print(f"[WARNING] run_{run_idx} raised: {exc}", file=sys.stderr)
                results[run_idx] = None

    # Collect all poses
    all_poses: list[dict] = []
    for run_i in range(n_seeds):
        out_pdbqt = results.get(run_i)
        if out_pdbqt is None:
            continue
        n_models = _count_models(out_pdbqt)
        for model_j in range(1, n_models + 1):
            coords = _pdbqt_heavy_coords(out_pdbqt, model_j)
            if len(coords) < 3:
                continue
            energy = _parse_energy(out_pdbqt, model_j)
            all_poses.append({
                "energy": energy,
                "coords": coords,
                "run_i": run_i,
                "model_j": model_j,
                "out_path": out_pdbqt,
                "nearest_rmsd": None,
            })

    if len(all_poses) < 3:
        sys.exit(
            f"ERROR: Only {len(all_poses)} valid pose(s) collected across all runs. "
            "Cannot cluster. Check that Vina-GPU ran successfully."
        )

    successful_runs = sum(1 for v in results.values() if v is not None)
    print(f"Collected {len(all_poses)} poses from {successful_runs}/{n_seeds} successful runs")

    # Cluster
    clusters = greedy_cluster(all_poses, args.rmsd_cutoff, args.top_n)
    print(f"Clusters found:   {len(clusters)}  (cutoff={args.rmsd_cutoff} Å)")

    # Write output
    write_ensemble(clusters, out_path, args.verbose)
    print(f"Top-{len(clusters)} written → {out_path}")

    # Summary table
    _print_summary(clusters)


def _print_summary(clusters: list) -> None:
    header = f"{'#':>4}  {'Energy':>8}  {'Run':>6}  {'OrigModel':>10}  {'RMSD_from_prev':>14}"
    print()
    print("Ensemble results:")
    print(header)
    print("-" * len(header))
    for rank, c in enumerate(clusters, start=1):
        rmsd_str = (
            f"{c['nearest_rmsd']:.1f} Å"
            if c.get("nearest_rmsd") is not None
            else "—"
        )
        print(
            f"{rank:>4}  {c['energy']:>8.2f}  "
            f"run{c['run_i']:>3}  {c['model_j']:>10}  {rmsd_str:>14}"
        )
    print()


if __name__ == "__main__":
    main()
