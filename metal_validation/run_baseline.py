#!/usr/bin/env python3
"""
Run Vina-GPU baseline docking on all prepared targets and compute RMSD.
Usage: python3 run_baseline.py [--dry-run]
"""

import os, sys, json, shutil, subprocess, math, glob

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SIF          = "/home/cycheng/Vina-GPU-2.1/autodock-vina-gpu.sif"
RESULTS      = os.path.join(BASE_DIR, "results")
REPORT       = os.path.join(BASE_DIR, "baseline_results.tsv")
SEARCH_DEPTH = 20   # steps per thread; 1 is too coarse for meaningful RMSD

DRY_RUN = "--dry-run" in sys.argv

# ── RMSD calculation (no external deps) ──────────────────────────────────────

def parse_pdbqt_coords(path):
    """Return list of (x,y,z) for ATOM/HETATM lines in MODEL 1 (or whole file if no MODELs)"""
    coords = []
    with open(path) as f:
        lines = f.readlines()
    has_model = any(l.startswith("MODEL") for l in lines)
    in_model1 = not has_model  # if no MODEL records, treat entire file as model 1
    for line in lines:
        if line.startswith("MODEL"):
            in_model1 = (line.split()[1] == "1")
        elif line.startswith("ENDMDL") and in_model1:
            break
        if in_model1 and line.startswith(("ATOM  ", "HETATM")):
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords.append((x, y, z))
            except:
                pass
    return coords

def parse_pdbqt_models(path):
    """Return list of coordinate lists, one per MODEL"""
    models = []
    current = []
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                current = []
            elif line.startswith("ENDMDL"):
                if current:
                    models.append(current)
                    current = []
            elif line.startswith(("ATOM  ", "HETATM")):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    current.append((x, y, z))
                except:
                    pass
    if current:
        models.append(current)
    return models

def rmsd(coords1, coords2):
    """Heavy-atom RMSD between two coordinate lists (same length assumed)"""
    n = min(len(coords1), len(coords2))
    if n == 0:
        return float("inf")
    s = sum((coords1[i][0]-coords2[i][0])**2 +
            (coords1[i][1]-coords2[i][1])**2 +
            (coords1[i][2]-coords2[i][2])**2
            for i in range(n))
    return math.sqrt(s / n)

def best_rmsd(crystal_coords, docked_models):
    """RMSD of top pose (model 1), also return best across all models"""
    if not docked_models or not crystal_coords:
        return float("inf"), float("inf")
    top1  = rmsd(crystal_coords, docked_models[0])
    best  = min(rmsd(crystal_coords, m) for m in docked_models)
    return top1, best

# ── Per-target docking ────────────────────────────────────────────────────────

def run_target(entry):
    pdb_id   = entry["pdb_id"]
    lig_res  = entry["ligand"]
    rec_pdbqt = entry["rec_pdbqt"]
    lig_pdbqt = entry["lig_pdbqt"]
    config    = entry["config"]

    if not entry.get("rec_ok") or not entry.get("lig_ok"):
        return pdb_id, None, None, "PREP_FAILED"

    for path in [rec_pdbqt, lig_pdbqt, config]:
        if not path or not os.path.exists(path):
            return pdb_id, None, None, f"MISSING:{os.path.basename(str(path))}"

    # Create per-target ligand dir (Vina-GPU needs a directory)
    tmp_lig_dir = os.path.join(BASE_DIR, "tmp_lig", pdb_id)
    os.makedirs(tmp_lig_dir, exist_ok=True)
    tmp_lig = os.path.join(tmp_lig_dir, f"{lig_res}.pdbqt")
    shutil.copy2(lig_pdbqt, tmp_lig)

    out_dir = os.path.join(RESULTS, pdb_id)
    os.makedirs(out_dir, exist_ok=True)

    # Build per-target config (override ligand_directory and output_directory)
    run_cfg = os.path.join(BASE_DIR, "tmp_lig", f"{pdb_id}_run.txt")
    with open(config) as f:
        lines = f.readlines()
    with open(run_cfg, "w") as f:
        for line in lines:
            if line.startswith("ligand_directory"):
                f.write(f"ligand_directory = {tmp_lig_dir}\n")
            elif line.startswith("output_directory"):
                f.write(f"output_directory = {out_dir}\n")
            elif line.startswith("opencl_binary_path"):
                # Use a writable host dir so the program recompiles from new .cl source
                # (avoids loading stale Kernel2_Opt.bin baked from old API into SIF)
                ocl_cache = os.path.join(BASE_DIR, "ocl_cache")
                f.write(f"opencl_binary_path = {ocl_cache}\n")
            elif line.startswith("receptor"):
                # Use path inside container
                f.write(f"receptor = {rec_pdbqt}\n")
            elif line.startswith("search_depth"):
                # Override search_depth for meaningful sampling (original configs have depth=1)
                f.write(f"search_depth = {SEARCH_DEPTH}\n")
            else:
                f.write(line)

    # Bind-mount the repo so the SIF recompiles kernels from the updated host
    # sources (kernel2.cl with const fix) instead of the stale sources inside the SIF.
    # VINA_GPU_HOME override points compilation at the host's OpenCL kernel directory.
    REPO_DIR = os.path.dirname(BASE_DIR)
    VINA_SRC = os.path.join(REPO_DIR, "AutoDock-Vina-GPU-2.1")
    cmd = (f"singularity run --nv "
           f"-B {BASE_DIR}:{BASE_DIR} "
           f"-B {VINA_SRC}:{VINA_SRC} "
           f"--env VINA_GPU_HOME={VINA_SRC} "
           f"{SIF} --config {run_cfg}")

    print(f"  [{pdb_id}] docking...", end="", flush=True)

    if DRY_RUN:
        print(f" [DRY RUN] cmd: {cmd[:80]}...")
        return pdb_id, 0.0, 0.0, "DRY_RUN"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f" FAILED (exit {result.returncode})")
        out_combined = (result.stdout + result.stderr).strip()
        if out_combined:
            print(f"  output: {out_combined[:400]}")
        return pdb_id, None, None, "RUN_FAILED"

    # Find output file
    out_files = glob.glob(os.path.join(out_dir, "*.pdbqt"))
    if not out_files:
        print(f" NO OUTPUT")
        return pdb_id, None, None, "NO_OUTPUT"

    out_file = out_files[0]

    # Load crystal ligand coords (from PDBQT prep — heavy atoms only)
    crystal_coords = parse_pdbqt_coords(lig_pdbqt)
    docked_models  = parse_pdbqt_models(out_file)

    r_top1, r_best = best_rmsd(crystal_coords, docked_models)
    print(f" top1={r_top1:.2f}Å  best={r_best:.2f}Å  ({len(docked_models)} poses)")
    return pdb_id, r_top1, r_best, "OK"


def main():
    json_path = os.path.join(BASE_DIR, "targets_prepared.json")
    if not os.path.exists(json_path):
        print("ERROR: targets_prepared.json not found — run prepare_all.py first")
        sys.exit(1)

    with open(json_path) as f:
        targets = json.load(f)

    if not os.path.exists(SIF):
        print(f"ERROR: SIF not found at {SIF}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Baseline docking — {len(targets)} targets  (SIF: {SIF})")
    if DRY_RUN:
        print("  *** DRY RUN — no actual docking ***")
    print(f"{'='*70}\n")

    rows = []
    for entry in targets:
        pdb_id, r_top1, r_best, status = run_target(entry)
        ki = entry.get("ki_nM", "?")
        metal = entry.get("metal", "?")
        desc  = entry.get("desc", "")
        rows.append((pdb_id, metal, ki, r_top1, r_best, status, desc))

    # Print and save summary
    print(f"\n{'='*70}")
    print(f"{'PDB':6} {'Metal':5} {'Ki(nM)':>9}  {'RMSD_top1':>10} {'RMSD_best':>10}  Status")
    print(f"{'-'*70}")

    ok_top1 = ok_best = 0
    total = 0

    with open(REPORT, "w") as rep:
        rep.write("PDB_ID\tMetal\tKi_nM\tRMSD_top1\tRMSD_best\tStatus\tDescription\n")
        for pdb_id, metal, ki, r_top1, r_best, status, desc in rows:
            r1_s = f"{r_top1:.2f}" if r_top1 is not None else "N/A"
            rb_s = f"{r_best:.2f}" if r_best is not None else "N/A"
            flag = ""
            if r_top1 is not None:
                total += 1
                if r_top1 < 2.0: ok_top1 += 1; flag += " ✓top1"
                if r_best < 2.0: ok_best += 1; flag += " ✓best"
            print(f"{pdb_id:6} {metal:5} {ki:>9}  {r1_s:>10} {rb_s:>10}  {status}{flag}")
            rep.write(f"{pdb_id}\t{metal}\t{ki}\t{r1_s}\t{rb_s}\t{status}\t{desc}\n")

    print(f"{'-'*70}")
    if total > 0:
        print(f"Top-1 RMSD < 2Å: {ok_top1}/{total} = {100*ok_top1/total:.0f}%  "
              f"(baseline target: > 50% after metal fix)")
        print(f"Best  RMSD < 2Å: {ok_best}/{total} = {100*ok_best/total:.0f}%")
    print(f"\nSaved: {REPORT}")

if __name__ == "__main__":
    main()
