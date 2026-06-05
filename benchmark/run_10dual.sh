#!/bin/bash
# Benchmark: 10 dual-ligand pairs against 2bm2
set -e
BIN=/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
export VINA_GPU_HOME=/home/cycheng/Vina-GPU-2.1
REC=/home/cycheng/Vina-GPU-2.1/src/AutoDock-Vina-GPU-2.1/input_file_example/2bm2_protein.pdbqt
LDIR=/home/cycheng/Vina-GPU-2.1/src/AutoDock-Vina-GPU-2.1/test
BIN_PATH=/home/cycheng/Vina-GPU-2.1
OUTDIR=/home/cycheng/Vina-GPU-2.1/benchmark/10dual/out

echo "=== 10 Dual-Ligand Pairs Benchmark ==="
# 10 pairs: (1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20)
TOTAL_START=$(date +%s%N)

for i in $(seq 1 2 19); do
    j=$((i+1))
    LA=$LDIR/drugbank${i}.pdbqt
    LB=$LDIR/drugbank${j}.pdbqt
    OUT_A=$OUTDIR/pair${i}_${j}_A_out.pdbqt
    OUT_B=$OUTDIR/pair${i}_${j}_B_out.pdbqt

    printf "  Pair (%2d,%2d)  " "$i" "$j"
    START=$(date +%s%N)
    $BIN --receptor "$REC" \
         --ligand "$LA" --ligand2 "$LB" \
         --out "$OUT_A" --out2 "$OUT_B" \
         --opencl_binary_path "$BIN_PATH" \
         --center_x 40.415 --center_y 110.986 --center_z 82.673 \
         --size_x 30 --size_y 30 --size_z 30 \
         --thread 8000 --search_depth 20 --gpu_id 0 2>&1 | grep -E "DIAG.*done|AutoDock"
    END=$(date +%s%N)
    echo "  → pair($i,$j) wall: $(( (END-START)/1000000 ))ms"
done

TOTAL_END=$(date +%s%N)
echo ""
echo "=== TOTAL 10 dual pairs: $(( (TOTAL_END-TOTAL_START)/1000000 ))ms ==="
