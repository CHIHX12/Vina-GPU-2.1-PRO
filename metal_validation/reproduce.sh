#!/bin/bash
# reproduce.sh — End-to-end metal validation reproduction script
#
# Prerequisites:
#   1. autodock-vina-gpu.sif (Singularity image) at repo root
#   2. AutoDockTools (conda env jp_214): prepare_receptor4.py, prepare_ligand4.py
#   3. OpenBabel: obabel
#   4. Internet access (RCSB PDB download)
#
# Hardware: tested on 2× RTX 6000 Ada (48 GB) with CUDA 12.x + OpenCL
#
# Usage:
#   cd metal_validation
#   bash reproduce.sh [--depth N]   # default depth=32
#
# Expected result: Best RMSD <2Å for 20/20 targets (100%)

set -euo pipefail
DEPTH=${2:-32}
for arg in "$@"; do [[ "$arg" == "--depth" ]] && DEPTH="${@: $((${#@}))}" && break; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo " Vina-GPU Metal Coordination Validation"
echo " Search depth: $DEPTH  |  $(date)"
echo "================================================================"

# ── Step 1: Download PDB files ────────────────────────────────────────
echo ""
echo "[1/3] Downloading PDB structures..."
python3 download_pdb.py

# ── Step 2: Prepare receptor + ligand PDBQT files ────────────────────
echo ""
echo "[2/3] Preparing receptor/ligand PDBQT files..."
python3 prepare_all.py

# ── Step 3: Run docking validation ───────────────────────────────────
echo ""
echo "[3/3] Running metal coordination docking (search_depth=$DEPTH)..."
python3 run_metal_validation.py --search-depth "$DEPTH"

echo ""
echo "================================================================"
echo " Results saved to: metal_results.tsv"
echo " Expected: Best RMSD <2Å = 20/20 (100%)"
echo "================================================================"
