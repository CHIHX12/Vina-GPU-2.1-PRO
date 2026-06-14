#!/bin/bash
# Synergy-vs-competition experiment for two ligands on one receptor.
#
# Docks ligand A alone, ligand B alone, and A+B together (same receptor + box),
# then prints an energy comparison that tells you whether the two ligands
# coexist cooperatively or compete for the binding site.
#
# Usage:
#   bash compare_solo_vs_codock.sh [gpu_id]
#
# Inputs (in this folder; override via env vars to reuse for any pair):
#   RECEPTOR=receptor.pdbqt  LIGA=ligandA.pdbqt  LIGB=ligandB.pdbqt  BOX=box_raw.txt
# BOX file: one line "centerX centerY centerZ sizeX sizeY sizeZ".
#
# Requires a FREE GPU (no other CUDA process — else CL_OUT_OF_HOST_MEMORY).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

GPU="${1:-0}"
BIN="${VINA_GPU_BIN:-/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1}"
OCL="${OPENCL_BINARY_PATH:-/home/cycheng/Vina-GPU-2.1}"
RECEPTOR="${RECEPTOR:-receptor.pdbqt}"
LIGA="${LIGA:-ligandA.pdbqt}"
LIGB="${LIGB:-ligandB.pdbqt}"
BOX="${BOX:-box_raw.txt}"
THREAD="${THREAD:-8000}"
DEPTH="${DEPTH:-64}"   # fixed deep search for accuracy (override with DEPTH=.. for speed)

read -r CX CY CZ SX SY SZ < "$BOX"
BOXARGS="--center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ"
COMMON="--receptor $RECEPTOR --opencl_binary_path $OCL $BOXARGS --thread $THREAD --search_depth $DEPTH --gpu_id $GPU"

echo "Receptor : $RECEPTOR"
echo "Ligand A : $LIGA"
echo "Ligand B : $LIGB"
echo "Box      : center=($CX,$CY,$CZ) size=($SX,$SY,$SZ)"
echo

echo "[1/3] docking A alone ..."
$BIN $COMMON --ligand "$LIGA" --out soloA.pdbqt 2>&1 | grep -iE "done in" || true

echo "[2/3] docking B alone ..."
$BIN $COMMON --ligand "$LIGB" --out soloB.pdbqt 2>&1 | grep -iE "done in" || true

echo "[3/3] docking A + B together ..."
$BIN $COMMON --ligand "$LIGA" --ligand2 "$LIGB" --out coA.pdbqt --out2 coB.pdbqt \
    2>&1 | grep -iE "Dual coupling|done in" || true

echo
python3 "$HERE/analyze_codock.py" soloA.pdbqt soloB.pdbqt coA.pdbqt coB.pdbqt
