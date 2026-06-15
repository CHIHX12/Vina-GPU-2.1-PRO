#!/bin/bash
# Benchmark cavity-biased init across targets: baseline vs VINA_CAVITY_INIT, best-of-N spyrmsd.
BIN=/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
SPY=/home/cycheng/miniforge3/bin/python
RMS=/home/cycheng/Vina-GPU-2.1/tests/bench/spyrmsd_best.py
export VINA_GPU_HOME=/home/cycheng/Vina-GPU-2.1/src/AutoDock-Vina-GPU-2.1
GPU=1
printf "%-8s %-12s %-12s %-12s\n" "target" "baseline" "cav-init" "rescue+init"
for t in 1U72 1DF7 3EQH 1HVY 4DFR; do
  d=/home/cycheng/Vina-GPU-2.1/dual_test/$t
  [ -f $d/box.txt ] || continue
  read CX CY CZ SX SY SZ < $d/box.txt
  B="--receptor $d/rec.pdbqt --ligand $d/ligA.pdbqt --opencl_binary_path /home/cycheng/Vina-GPU-2.1 --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --thread 8000 --search_depth 64 --gpu_id $GPU"
  $BIN $B --out /tmp/bb_$t.pdbqt >/dev/null 2>&1
  VINA_CAVITY_INIT=1 $BIN $B --out /tmp/bi_$t.pdbqt >/dev/null 2>&1
  VINA_CAVITY=1 VINA_CAVITY_INIT=1 $BIN $B --out /tmp/br_$t.pdbqt >/dev/null 2>&1
  r0=$($SPY $RMS $d/ligA_xtal.pdbqt /tmp/bb_$t.pdbqt 2>/dev/null)
  r1=$($SPY $RMS $d/ligA_xtal.pdbqt /tmp/bi_$t.pdbqt 2>/dev/null)
  r2=$($SPY $RMS $d/ligA_xtal.pdbqt /tmp/br_$t.pdbqt 2>/dev/null)
  printf "%-8s %-12s %-12s %-12s\n" "$t" "$r0" "$r1" "$r2"
done
echo "DONE"
