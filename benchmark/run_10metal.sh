#!/bin/bash
# Benchmark: 10 metal-coordinated targets (AD4Zn + QFD pipi scoring)
set -e
BIN=/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
METAL=/home/cycheng/LigandScope/data/Metal_enzymes
BIN_PATH=/home/cycheng/Vina-GPU-2.1
OUTDIR=/home/cycheng/Vina-GPU-2.1/benchmark/10metal/out
export VINA_GPU_HOME=$BIN_PATH
export VINA_LS_METAL_WEIGHT=0  # pipi grid alone is optimal (Phase 5 benchmark 2026-06-05)

declare -A CX=( [1A42]=-4.1  [1BNN]=-4.0  [1G52]=-4.3  [1GKC]=65.6
                [1JAQ]=27.2  [1MMQ]=49.5  [1O86]=40.6  [1OQ5]=17.5
                [1UZE]=40.5  [1YDB]=-5.3 )
declare -A CY=( [1A42]=5.2   [1BNN]=4.9   [1G52]=5.6   [1GKC]=31.1
                [1JAQ]=58.8  [1MMQ]=-37.8 [1O86]=32.8  [1OQ5]=6.5
                [1UZE]=35.4  [1YDB]=3.2  )
declare -A CZ=( [1A42]=14.5  [1BNN]=15.3  [1G52]=14.5  [1GKC]=117.8
                [1JAQ]=51.8  [1MMQ]=47.1  [1O86]=47.3  [1OQ5]=13.0
                [1UZE]=47.1  [1YDB]=15.6  )
declare -A LIG=( [1A42]=1A42_BZU    [1BNN]=1BNN_AL1  [1G52]=1G52_F2B
                 [1GKC]=1GKC_NFH    [1JAQ]=1JAQ_01S  [1MMQ]=1MMQ_RRS
                 [1O86]=1O86_LPR    [1OQ5]=1OQ5_CEL  [1UZE]=1UZE_EAL
                 [1YDB]=1YDB_AZM   )

TARGETS=(1A42 1BNN 1G52 1GKC 1JAQ 1MMQ 1O86 1OQ5 1UZE 1YDB)

echo "=== 10 Metal Targets Benchmark (AD4Zn) ==="
TOTAL_START=$(date +%s%N)

for T in "${TARGETS[@]}"; do
    REC=$METAL/${T}/${T}_receptor.pdbqt
    LG=$METAL/${T}/${LIG[$T]}.pdbqt
    OUT=$OUTDIR/${T}_out.pdbqt

    printf "  %-6s  " "$T"
    START=$(date +%s%N)
    $BIN --receptor "$REC" \
         --ligand "$LG" \
         --out "$OUT" \
         --opencl_binary_path "$BIN_PATH" \
         --center_x ${CX[$T]} --center_y ${CY[$T]} --center_z ${CZ[$T]} \
         --size_x 25 --size_y 25 --size_z 25 \
         --thread 8000 --search_depth 20 --num_modes 9 \
         --ad4zn --gpu_id 0 2>&1 | grep -E "DIAG|AutoDock|Affinity|mode 1"
    END=$(date +%s%N)
    echo "  → $T wall: $(( (END-START)/1000000 ))ms"
done

TOTAL_END=$(date +%s%N)
echo ""
echo "=== TOTAL 10 metal: $(( (TOTAL_END-TOTAL_START)/1000000 ))ms ==="
