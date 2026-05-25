#!/bin/bash
# reproduce.sh — LigandScope self-docking validation
#
# Prerequisites (pdbqt files are pre-included in the repo):
#   1. autodock-vina-gpu.sif at repo root  (build once: see README Quick Start)
#   2. NVIDIA GPU with CUDA drivers
#
# Usage:
#   cd ligandscope_validation
#   bash reproduce.sh              # all 12 targets
#   bash reproduce.sh --pdb 2ZV2   # single target
#
# Expected: Top-1 RMSD < 2Å = 12/12 (100%)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================================"
echo " Vina-GPU 2.1 PRO — LigandScope Self-Docking Validation"
echo " $(date)"
echo "================================================================"

# ── Pre-check SIF ─────────────────────────────────────────────────────
SIF="$REPO_DIR/autodock-vina-gpu.sif"
if [ ! -f "$SIF" ]; then
    echo ""
    echo "ERROR: autodock-vina-gpu.sif not found at repo root."
    echo "  Build it first:"
    echo "    docker build -t autodock-vina-gpu ."
    echo "    singularity build autodock-vina-gpu.sif docker-daemon://autodock-vina-gpu:latest"
    exit 1
fi

# ── Run validation ────────────────────────────────────────────────────
echo "[1/1] Running self-docking on 12 targets..."
python3 "$SCRIPT_DIR/run_ligandscope.py" --gpu-only "$@"

echo ""
echo "================================================================"
echo " Results: ligandscope_validation/ligandscope_results.tsv"
echo " Expected: Top-1 RMSD < 2Å = 12/12 (100%)"
echo "================================================================"
