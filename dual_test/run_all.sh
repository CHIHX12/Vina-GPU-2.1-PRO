#!/bin/bash
# Run dual docking + analysis on all prepared targets; write summary.tsv
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
echo -e "PDB\tligA(tors)\tligB(tors)\tGPU_A\tCPU_A\tGPU_B\tCPU_B" > summary.tsv
for ID in 1RX2 1U72 1HVY 3EQH 1DF7; do
  [ -f "$ID/box.txt" ] || continue
  bash run_dual.sh $ID 1 >/dev/null 2>&1
  python3 analyze.py $ID 2>/dev/null | grep -vE "no MODEL" > "$ID/result.txt"
done
echo "ALL_DONE"
