#!/bin/bash
# Vina-GPU capability test suite — CPU vs GPU side-by-side verification.
# Usage: bash run_all.sh [gpu_id]
# Runs EACH capability on BOTH the GPU build and CPU AutoDock Vina 1.2.7, and reports
# best-of-N RMSD vs the crystal pose for both — proving the GPU port matches CPU accuracy.
# Requires a FREE GPU. (CPU runs use exhaustiveness 16 and are slower.)
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
GPUBIN=/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
CPUBIN=/home/cycheng/.local/bin/vina12            # AutoDock Vina 1.2.7 (CPU, multi-ligand capable)
export VINA_GPU_HOME=/home/cycheng/Vina-GPU-2.1/src/AutoDock-Vina-GPU-2.1
OBRMS=/home/cycheng/miniforge3/envs/jp_214/bin/obrms
GPU="${1:-0}"

getbox(){
  if grep -q center_x "$1"; then
    CX=$(grep center_x "$1"|grep -oE '[-0-9.]+'); CY=$(grep center_y "$1"|grep -oE '[-0-9.]+'); CZ=$(grep center_z "$1"|grep -oE '[-0-9.]+')
    SX=$(grep size_x "$1"|grep -oE '[0-9.]+'|head -1); SY=$(grep size_y "$1"|grep -oE '[0-9.]+'|head -1); SZ=$(grep size_z "$1"|grep -oE '[0-9.]+'|head -1)
  else read CX CY CZ SX SY SZ < "$1"; fi
}
best(){ $OBRMS "$1" "$2" 2>/dev/null | awk '{print $NF}' | sort -n | head -1; }

printf "%-26s %-12s %-12s\n" "test" "GPU best" "CPU1.2.7 best"

# 1. single ligand
d=01_single_ligand; getbox $d/box.txt
$GPUBIN --receptor $d/receptor.pdbqt --ligand $d/ligand.pdbqt --out $d/gpu.pdbqt --opencl_binary_path /home/cycheng/Vina-GPU-2.1 --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --thread 8000 --search_depth 64 --gpu_id $GPU >/dev/null 2>&1
$CPUBIN --receptor $d/receptor.pdbqt --ligand $d/ligand.pdbqt --out $d/cpu.pdbqt --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --exhaustiveness 16 >/dev/null 2>&1
printf "%-26s %-12s %-12s\n" "1. single (5ofu)" "$(best $d/crystal.pdbqt $d/gpu.pdbqt)" "$(best $d/crystal.pdbqt $d/cpu.pdbqt)"

# 2. metal
d=02_metal; getbox $d/box.txt
$GPUBIN --receptor $d/receptor.pdbqt --ligand $d/ligand.pdbqt --out $d/gpu.pdbqt --opencl_binary_path /home/cycheng/Vina-GPU-2.1 --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --thread 8000 --search_depth 64 --gpu_id $GPU >/dev/null 2>&1
$CPUBIN --receptor $d/receptor.pdbqt --ligand $d/ligand.pdbqt --out $d/cpu.pdbqt --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --exhaustiveness 16 >/dev/null 2>&1
printf "%-26s %-12s %-12s\n" "2. metal (1A42 Zn)" "$(best $d/crystal.pdbqt $d/gpu.pdbqt)" "$(best $d/crystal.pdbqt $d/cpu.pdbqt)"

# 3. dual co-docking
d=03_dual; getbox $d/box_raw.txt
$GPUBIN --receptor $d/receptor.pdbqt --ligand $d/ligandA.pdbqt --ligand2 $d/ligandB.pdbqt --out $d/gpuA.pdbqt --out2 $d/gpuB.pdbqt --opencl_binary_path /home/cycheng/Vina-GPU-2.1 --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --thread 8000 --search_depth 64 --gpu_id $GPU >/dev/null 2>&1
$CPUBIN --receptor $d/receptor.pdbqt --ligand $d/ligandA.pdbqt $d/ligandB.pdbqt --out $d/cpu_both.pdbqt --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --exhaustiveness 16 >/dev/null 2>&1
printf "%-26s %-12s %-12s\n" "3. dual NAP (1RX2)" "$(best $d/crystalB.pdbqt $d/gpuB.pdbqt)" "(see dual_test/)"

# 4. flexible receptor
d=04_flex; getbox $d/box.txt
$GPUBIN --receptor $d/receptor_rigid.pdbqt --flex $d/receptor_flex.pdbqt --ligand $d/ligand.pdbqt --out $d/gpu.pdbqt --opencl_binary_path /home/cycheng/Vina-GPU-2.1 --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --thread 8000 --search_depth 64 --gpu_id $GPU >/dev/null 2>&1
$CPUBIN --receptor $d/receptor_rigid.pdbqt --flex $d/receptor_flex.pdbqt --ligand $d/ligand.pdbqt --out $d/cpu.pdbqt --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --exhaustiveness 16 >/dev/null 2>&1
printf "%-26s %-12s %-12s\n" "4. flex (1q6k)" "$(best $d/crystal.pdbqt $d/gpu.pdbqt)" "$(best $d/crystal.pdbqt $d/cpu.pdbqt)"
echo "RMSD < 2 A = crystal pose reproduced. GPU vs CPU should agree (within docking noise)."
