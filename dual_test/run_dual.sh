#!/bin/bash
# run_dual.sh PDB_ID [gpu_id]
# Runs the two prepared ligands as a two-ligand pocket with:
#   (1) Vina-GPU dual co-docking  -> <ID>/gpu_ligA.pdbqt, gpu_ligB.pdbqt
#   (2) AutoDock Vina 1.2.7 native simultaneous multi-ligand  -> <ID>/cpu_both.pdbqt
# so the GPU dual can be validated against the CPU 1.2.x reference and the crystal.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ID="$1"; GPU="${2:-1}"
D="$HERE/$ID"
[ -f "$D/box.txt" ] || { echo "prepare first: python3 prepare_target.py $ID"; exit 1; }
read CX CY CZ SX SY SZ < "$D/box.txt"
GPUBIN=/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
CPUBIN=/home/cycheng/.local/bin/vina12
export VINA_GPU_HOME=/home/cycheng/Vina-GPU-2.1/src/AutoDock-Vina-GPU-2.1

echo "=== $ID : $(cat $D/info.txt)"
echo "--- (1) Vina-GPU dual co-docking ---"
$GPUBIN --receptor "$D/rec.pdbqt" --ligand "$D/ligA.pdbqt" --ligand2 "$D/ligB.pdbqt" \
        --out "$D/gpu_ligA.pdbqt" --out2 "$D/gpu_ligB.pdbqt" \
        --opencl_binary_path /home/cycheng/Vina-GPU-2.1 \
        --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ \
        --thread 8000 --search_depth 20 --gpu_id $GPU 2>&1 | grep -iE "dual =|done in|Err-|Writing" || true

echo "--- (2) AutoDock Vina 1.2.7 native simultaneous multi-ligand ---"
$CPUBIN --receptor "$D/rec.pdbqt" --ligand "$D/ligA.pdbqt" "$D/ligB.pdbqt" \
        --out "$D/cpu_both.pdbqt" \
        --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ \
        --exhaustiveness 16 --num_modes 9 2>&1 | grep -iE "error|mode|Writing|^ +[0-9]" | tail -6 || true
echo "done: $ID"
