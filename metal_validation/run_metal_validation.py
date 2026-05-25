#!/usr/bin/env python3
"""
Run Vina-GPU metal coordination validation.

Docks all targets with the metal-coordination kernel (kernel1.cl modified)
and compares RMSD vs baseline (standard Vina scoring).

Usage: python3 run_metal_validation.py [--search-depth N] [--dry-run]
"""

import os, sys, json, shutil, subprocess, math, glob, re

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
REPO_DIR      = os.path.dirname(BASE_DIR)
SIF           = os.path.join(REPO_DIR, "autodock-vina-gpu.sif")
VINA_SRC      = os.path.join(REPO_DIR, "src", "AutoDock-Vina-GPU-2.1")
RESULTS       = os.path.join(BASE_DIR, "results_metal")
REPORT        = os.path.join(BASE_DIR, "metal_results.tsv")
BASELINE_TSV  = os.path.join(BASE_DIR, "baseline_results.tsv")
# Use pre-compiled metal cache (compiled from modified kernel1.cl)
OCL_CACHE     = "/tmp/vina-metal-cache"

SEARCH_DEPTH  = 20  # default; override via --search-depth N
DRY_RUN       = "--dry-run" in sys.argv

for i, arg in enumerate(sys.argv):
    if arg == "--search-depth" and i + 1 < len(sys.argv):
        SEARCH_DEPTH = int(sys.argv[i + 1])

# ── RMSD helpers ─────────────────────────────────────────────────────────────

def parse_pdbqt_with_types(path):
    """Coordinates + AD atom types for MODEL 1 (or whole file)."""
    coords, types = [], []
    with open(path) as f:
        lines = f.readlines()
    has_model = any(l.startswith("MODEL") for l in lines)
    in_model1 = not has_model
    for line in lines:
        if line.startswith("MODEL"):
            in_model1 = (line.split()[1] == "1")
        elif line.startswith("ENDMDL") and in_model1:
            break
        if in_model1 and line.startswith(("ATOM  ", "HETATM")):
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                t = line[77:79].strip() if len(line) > 77 else ""
                coords.append((x, y, z))
                types.append(t)
            except Exception:
                pass
    return coords, types

def parse_pdbqt_models_with_types(path):
    """List of (coords, types) per MODEL from a Vina output PDBQT."""
    models, cur_c, cur_t = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                cur_c, cur_t = [], []
            elif line.startswith("ENDMDL"):
                if cur_c:
                    models.append((cur_c, cur_t))
                cur_c, cur_t = [], []
            elif line.startswith(("ATOM  ", "HETATM")):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    t = line[77:79].strip() if len(line) > 77 else ""
                    cur_c.append((x, y, z)); cur_t.append(t)
                except Exception:
                    pass
    if cur_c:
        models.append((cur_c, cur_t))
    return models

def rmsd_plain(c1, c2):
    n = min(len(c1), len(c2))
    if n == 0:
        return float("inf")
    return math.sqrt(sum((c1[i][0]-c2[i][0])**2 +
                         (c1[i][1]-c2[i][1])**2 +
                         (c1[i][2]-c2[i][2])**2
                         for i in range(n)) / n)

def rmsd_sym(c1, t1, c2, t2):
    """Symmetry-aware RMSD: permutes equivalent atoms (same AD type) to minimise RMSD.
    Only handles groups of ≤4 equivalent atoms (≤24 permutations) to stay fast."""
    from collections import defaultdict
    from itertools import permutations, product as iprod

    n = min(len(c1), len(t1), len(c2), len(t2))
    if n == 0:
        return float("inf")
    c1, t1, c2, t2 = list(c1[:n]), list(t1[:n]), list(c2[:n]), list(t2[:n])

    groups = defaultdict(list)
    for i, t in enumerate(t1):
        groups[t].append(i)

    sym_groups = [idxs for idxs in groups.values() if 2 <= len(idxs) <= 4]
    if not sym_groups:
        return rmsd_plain(c1, c2)

    best = float("inf")
    perms_per = [list(permutations(range(len(g)))) for g in sym_groups]
    for combo in iprod(*perms_per):
        c2p = list(c2)
        for gi, perm in enumerate(combo):
            idxs = sym_groups[gi]
            orig = [c2[i] for i in idxs]
            for j, p in enumerate(perm):
                c2p[idxs[j]] = orig[p]
        r = rmsd_plain(c1, c2p)
        if r < best:
            best = r
    return best

def best_rmsd(crystal_coords, crystal_types, models_data):
    """Returns (top1_rmsd, best_rmsd) using symmetry-aware RMSD."""
    if not models_data or not crystal_coords:
        return float("inf"), float("inf")
    rmsds = [rmsd_sym(crystal_coords, crystal_types, mc, mt) for mc, mt in models_data]
    return rmsds[0], min(rmsds)

# ── Load baseline for comparison ─────────────────────────────────────────────

def load_baseline():
    baseline = {}
    if not os.path.exists(BASELINE_TSV):
        return baseline
    with open(BASELINE_TSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("PDB_ID"):
                continue
            parts = line.split("\t")
            if len(parts) >= 5:
                pdb = parts[0]
                try:
                    top1 = float(parts[3])
                except (ValueError, IndexError):
                    top1 = None
                try:
                    best = float(parts[4])
                except (ValueError, IndexError):
                    best = None
                baseline[pdb] = (top1, best)
    return baseline

# ── Per-target docking ────────────────────────────────────────────────────────

def run_target(entry):
    pdb_id    = entry["pdb_id"]
    lig_pdbqt = entry["lig_pdbqt"]
    rec_pdbqt = entry["rec_pdbqt"]
    config    = entry["config"]

    if not entry.get("rec_ok") or not entry.get("lig_ok"):
        return pdb_id, None, None, "PREP_FAILED"

    for path in [rec_pdbqt, lig_pdbqt, config]:
        if not path or not os.path.exists(path):
            return pdb_id, None, None, f"MISSING:{os.path.basename(str(path))}"

    # Temp ligand dir (Vina-GPU needs a directory)
    tmp_lig_dir = os.path.join(BASE_DIR, "tmp_lig", pdb_id + "_metal")
    os.makedirs(tmp_lig_dir, exist_ok=True)
    lig_res = entry["ligand"]
    tmp_lig = os.path.join(tmp_lig_dir, f"{lig_res}.pdbqt")
    shutil.copy2(lig_pdbqt, tmp_lig)

    out_dir = os.path.join(RESULTS, pdb_id)
    os.makedirs(out_dir, exist_ok=True)

    # Build per-target config with our overrides
    run_cfg = os.path.join(BASE_DIR, "tmp_lig", f"{pdb_id}_metal_run.txt")
    with open(config) as f:
        lines = f.readlines()
    with open(run_cfg, "w") as f:
        for line in lines:
            if line.startswith("ligand_directory"):
                f.write(f"ligand_directory = {tmp_lig_dir}\n")
            elif line.startswith("output_directory"):
                f.write(f"output_directory = {out_dir}\n")
            elif line.startswith("opencl_binary_path"):
                f.write(f"opencl_binary_path = {OCL_CACHE}\n")
            elif line.startswith("receptor"):
                f.write(f"receptor = {rec_pdbqt}\n")
            elif line.startswith("search_depth"):
                f.write(f"search_depth = {SEARCH_DEPTH}\n")
            else:
                f.write(line)

    # Run with VINA_GPU_HOME pointing at our modified kernel source
    cmd = (f"singularity run --nv "
           f"-B {BASE_DIR}:{BASE_DIR} "
           f"-B {VINA_SRC}:{VINA_SRC} "
           f"--env VINA_GPU_HOME={VINA_SRC} "
           f"--env OPENCL_BINARY_PATH={OCL_CACHE} "
           f"{SIF} --config {run_cfg}")

    print(f"  [{pdb_id}] docking...", end="", flush=True)

    if DRY_RUN:
        print(f" [DRY RUN]")
        return pdb_id, 0.0, 0.0, "DRY_RUN"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f" FAILED (exit {result.returncode})")
        out_combined = (result.stdout + result.stderr).strip()
        if out_combined:
            print(f"  {out_combined[:300]}")
        return pdb_id, None, None, "RUN_FAILED"

    out_files = glob.glob(os.path.join(out_dir, "*.pdbqt"))
    if not out_files:
        return pdb_id, None, None, "NO_OUTPUT"

    out_file = out_files[0]
    crystal_coords, crystal_types = parse_pdbqt_with_types(lig_pdbqt)
    docked_models = parse_pdbqt_models_with_types(out_file)

    r_top1, r_best = best_rmsd(crystal_coords, crystal_types, docked_models)
    print(f" top1={r_top1:.2f}Å  best={r_best:.2f}Å  ({len(docked_models)} poses)")
    return pdb_id, r_top1, r_best, "OK"


def main():
    json_path = os.path.join(BASE_DIR, "targets_prepared.json")
    if not os.path.exists(json_path):
        print("ERROR: targets_prepared.json not found — run prepare_all.py first")
        sys.exit(1)
    if not os.path.exists(SIF):
        print(f"ERROR: SIF not found at {SIF}")
        sys.exit(1)

    # Ensure metal cache exists (compiled from modified kernel1.cl)
    if not os.path.exists(os.path.join(OCL_CACHE, "Kernel1_Opt.bin")):
        print(f"WARNING: Metal kernel cache not found at {OCL_CACHE}")
        print("  The first target will trigger compilation (slow ~30s)")
    os.makedirs(OCL_CACHE, exist_ok=True)

    with open(json_path) as f:
        targets = json.load(f)

    baseline = load_baseline()

    print(f"\n{'='*72}")
    print(f"Metal coordination validation — {len(targets)} targets  (sd={SEARCH_DEPTH})")
    print(f"  Kernel source: {VINA_SRC}/OpenCL/src/kernels/kernel1.cl")
    print(f"  OCL cache:     {OCL_CACHE}")
    if DRY_RUN:
        print("  *** DRY RUN — no actual docking ***")
    print(f"{'='*72}\n")

    rows = []
    for entry in targets:
        pdb_id, r_top1, r_best, status = run_target(entry)
        ki    = entry.get("ki_nM", "?")
        metal = entry.get("metal", "?")
        desc  = entry.get("desc", "")
        rows.append((pdb_id, metal, ki, r_top1, r_best, status, desc))

    # Summary with delta vs baseline
    print(f"\n{'='*90}")
    hdr = f"{'PDB':6} {'Metal':5} {'Ki(nM)':>9}  {'top1(metal)':>11} {'best(metal)':>11}  {'Δtop1':>7}  {'Δbest':>7}  Status"
    print(hdr)
    print(f"{'-'*90}")

    ok_top1 = ok_best = 0
    total = 0

    with open(REPORT, "w") as rep:
        rep.write("PDB_ID\tMetal\tKi_nM\tRMSD_top1\tRMSD_best\tdelta_top1\tdelta_best\tStatus\tDescription\n")
        for pdb_id, metal, ki, r_top1, r_best, status, desc in rows:
            r1_s = f"{r_top1:.2f}" if r_top1 is not None else "N/A"
            rb_s = f"{r_best:.2f}" if r_best is not None else "N/A"

            b_top1, b_best = baseline.get(pdb_id, (None, None))
            d1_s = d2_s = "   N/A"
            if r_top1 is not None and b_top1 is not None:
                d1 = r_top1 - b_top1
                d1_s = f"{d1:+.2f}" + ("↓" if d1 < 0 else "↑" if d1 > 0.1 else " ")
            if r_best is not None and b_best is not None:
                d2 = r_best - b_best
                d2_s = f"{d2:+.2f}" + ("↓" if d2 < 0 else "↑" if d2 > 0.1 else " ")

            flag = ""
            if r_top1 is not None:
                total += 1
                if r_top1 < 2.0: ok_top1 += 1; flag += " ✓top1"
                if r_best < 2.0: ok_best += 1; flag += " ✓best"

            print(f"{pdb_id:6} {metal:5} {str(ki):>9}  {r1_s:>11} {rb_s:>11}  {d1_s:>7}  {d2_s:>7}  {status}{flag}")
            rep.write(f"{pdb_id}\t{metal}\t{ki}\t{r1_s}\t{rb_s}\t{d1_s.strip()}\t{d2_s.strip()}\t{status}\t{desc}\n")

    print(f"{'-'*90}")
    if total > 0:
        print(f"Top-1 RMSD < 2Å: {ok_top1}/{total} = {100*ok_top1/total:.0f}%  "
              f"(baseline: 0/{total} = 0%)")
        print(f"Best  RMSD < 2Å: {ok_best}/{total} = {100*ok_best/total:.0f}%")
    print(f"\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
