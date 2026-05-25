#!/bin/bash
# reproduce.sh — Metal validation reproduction script
#
# Minimal prerequisites (pdbqt files are pre-included in the repo):
#   1. autodock-vina-gpu.sif at repo root  (build once: see README Quick Start)
#   2. NVIDIA GPU with CUDA drivers
#
# Optional (only needed to regenerate pdbqt from scratch):
#   - AutoDockTools: prepare_receptor4.py, prepare_ligand4.py
#   - OpenBabel: obabel
#   - Internet access (RCSB PDB download)
#
# Usage:
#   cd metal_validation
#   bash reproduce.sh             # default search_depth=32
#   bash reproduce.sh --depth 4   # faster, still ~90% accuracy
#
# Expected: Best RMSD < 2Å = 18-20/20 (20/20 confirmed over 9 runs)

set -euo pipefail
DEPTH=32
for i in "$@"; do [[ "$i" == "--depth" ]] && shift && DEPTH="$1" && break; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo " Vina-GPU 2.1 PRO — Metalloenzyme Validation"
echo " search_depth=$DEPTH  |  $(date)"
echo "================================================================"

# ── Pre-check SIF ────────────────────────────────────────────────────
SIF="$(dirname "$SCRIPT_DIR")/autodock-vina-gpu.sif"
if [ ! -f "$SIF" ]; then
    echo ""
    echo "ERROR: autodock-vina-gpu.sif not found at repo root."
    echo "  Build it first:"
    echo "    docker build -t autodock-vina-gpu ."
    echo "    singularity build autodock-vina-gpu.sif docker-daemon://autodock-vina-gpu:latest"
    exit 1
fi

# ── Step 1: Download PDB files (skip if already present) ─────────────
PDB_COUNT=$(ls pdb/*.pdb 2>/dev/null | wc -l)
if [ "$PDB_COUNT" -ge 20 ]; then
    echo "[1/3] PDB files already present ($PDB_COUNT files) — skipping download"
else
    echo "[1/3] Downloading PDB structures..."
    python3 download_pdb.py
fi

# ── Step 2: Prepare PDBQT files (skip if already present) ────────────
REC_COUNT=$(ls pdbqt_receptor/*.pdbqt 2>/dev/null | wc -l)
LIG_COUNT=$(ls pdbqt_ligand/*.pdbqt 2>/dev/null | wc -l)
if [ "$REC_COUNT" -ge 20 ] && [ "$LIG_COUNT" -ge 20 ]; then
    echo "[2/3] PDBQT files already present (rec=$REC_COUNT lig=$LIG_COUNT) — skipping prep"
else
    echo "[2/3] Preparing receptor/ligand PDBQT files..."
    echo "      (requires AutoDockTools + OpenBabel)"
    python3 prepare_all.py
fi

# ── Step 3: Run docking ───────────────────────────────────────────────
echo "[3/3] Running metal coordination docking (search_depth=$DEPTH)..."
python3 run_metal_validation.py --search-depth "$DEPTH"

echo ""
echo "================================================================"
echo " Results: metal_results.tsv"
echo " Expected: Best RMSD < 2Å = 18-20/20"
echo "================================================================"
