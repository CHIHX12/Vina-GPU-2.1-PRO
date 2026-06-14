#!/bin/bash
# Vina-GPU capability test suite. Usage: bash run_all.sh [gpu_id]
# Runs single-ligand, metal, dual-ligand, and flexible-receptor docking, and reports
# best-of-N RMSD vs the crystal pose (lower = correct). Requires a FREE GPU.
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
BIN=/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
export VINA_GPU_HOME=/home/cycheng/Vina-GPU-2.1/src/AutoDock-Vina-GPU-2.1
OBRMS=/home/cycheng/miniforge3/envs/jp_214/bin/obrms
GPU="${1:-0}"

# parse box.txt -> CX CY CZ SX SY SZ  (handles "key = val" lines or 6 space-separated numbers)
getbox(){
  if grep -q center_x "$1"; then
    CX=$(grep center_x "$1"|grep -oE '[-0-9.]+'); CY=$(grep center_y "$1"|grep -oE '[-0-9.]+'); CZ=$(grep center_z "$1"|grep -oE '[-0-9.]+')
    SX=$(grep size_x "$1"|grep -oE '[0-9.]+'|head -1); SY=$(grep size_y "$1"|grep -oE '[0-9.]+'|head -1); SZ=$(grep size_z "$1"|grep -oE '[0-9.]+'|head -1)
  else read CX CY CZ SX SY SZ < "$1"; fi
}
best(){ $OBRMS "$1" "$2" 2>/dev/null | awk '{print $NF}' | sort -n | head -1; }

echo "=== Vina-GPU capability tests (GPU $GPU) ==="

# ---- 1. single-ligand (drug-like; expect ~0.3 A) ----
d=01_single_ligand; getbox $d/box.txt
$BIN --receptor $d/receptor.pdbqt --ligand $d/ligand.pdbqt --out $d/out.pdbqt \
     --opencl_binary_path /home/cycheng/Vina-GPU-2.1 \
     --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ \
     --thread 8000 --search_depth 20 --gpu_id $GPU >/dev/null 2>&1
echo "1. single-ligand (5ofu)   best RMSD = $(best $d/crystal.pdbqt $d/out.pdbqt) A   (expect ~0.3, PASS<2)"

# ---- 2. metal (Zn carbonic anhydrase; expect ~1.3 A) ----
d=02_metal; getbox $d/box.txt
$BIN --receptor $d/receptor.pdbqt --ligand $d/ligand.pdbqt --out $d/out.pdbqt \
     --opencl_binary_path /home/cycheng/Vina-GPU-2.1 \
     --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ \
     --thread 8000 --search_depth 20 --gpu_id $GPU >/dev/null 2>&1
echo "2. metal (1A42 Zn/CA)     best RMSD = $(best $d/crystal.pdbqt $d/out.pdbqt) A   (expect ~1.3, PASS<2)"

# ---- 3. dual-ligand co-docking (ternary; cofactor+substrate) ----
d=03_dual; getbox $d/box_raw.txt
$BIN --receptor $d/receptor.pdbqt --ligand $d/ligandA.pdbqt --ligand2 $d/ligandB.pdbqt \
     --out $d/outA.pdbqt --out2 $d/outB.pdbqt \
     --opencl_binary_path /home/cycheng/Vina-GPU-2.1 \
     --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ \
     --thread 8000 --search_depth 20 --gpu_id $GPU >/dev/null 2>&1
echo "3. dual co-docking (1RX2)  ligA best = $(best $d/crystalA.pdbqt $d/outA.pdbqt) A   ligB best = $(best $d/crystalB.pdbqt $d/outB.pdbqt) A   (flexible cofactors = hard)"

# ---- 4. flexible receptor (induced fit) ----
d=04_flex; getbox $d/box.txt
$BIN --receptor $d/receptor_rigid.pdbqt --flex $d/receptor_flex.pdbqt --ligand $d/ligand.pdbqt \
     --out $d/out.pdbqt --opencl_binary_path /home/cycheng/Vina-GPU-2.1 \
     --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ \
     --thread 8000 --search_depth 20 --gpu_id $GPU >/dev/null 2>&1
echo "4. flexible receptor (1bjr) best RMSD = $(best $d/crystal.pdbqt $d/out.pdbqt) A   (induced-fit; beta feature)"
echo "=== done. RMSD < 2 A = correctly reproduced the crystal pose. ==="
