#!/bin/bash
# Full 18-target metal validation with SD=32 (new default)
BIN=/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
METAL=/home/cycheng/LigandScope/data/Metal_enzymes
OUTDIR=/home/cycheng/Vina-GPU-2.1/benchmark/metal_revalidation
BIN_PATH=/home/cycheng/Vina-GPU-2.1
export VINA_GPU_HOME=$BIN_PATH
export VINA_LS_METAL_WEIGHT=0  # pipi grid alone is optimal (Phase 5 benchmark 2026-06-05)

declare -A CX=([1A42]=-4.14 [1BNN]=-3.98 [1G52]=-4.32 [1GKC]=65.61
               [1JAQ]=27.22 [1MMQ]=49.47 [1O86]=40.55 [1OQ5]=17.48
               [1UZE]=40.46 [1YDB]=-5.31 [2C6N]=-25.58 [2G1M]=40.06
               [2OVX]=24.78 [2W0D]=-13.37 [3HS4]=-5.42 [3L2U]=-38.42
               [3P5A]=-5.52 [3S3M]=-39.41)
declare -A CY=([1A42]=5.18  [1BNN]=4.93  [1G52]=5.57  [1GKC]=31.08
               [1JAQ]=58.79 [1MMQ]=-37.76 [1O86]=32.80 [1OQ5]=6.49
               [1UZE]=35.43 [1YDB]=3.25  [2C6N]=-17.16 [2G1M]=19.58
               [2OVX]=8.44  [2W0D]=24.70 [3HS4]=3.10  [3L2U]=33.31
               [3P5A]=2.44  [3S3M]=32.60)
declare -A CZ=([1A42]=14.52 [1BNN]=15.26 [1G52]=14.51 [1GKC]=117.84
               [1JAQ]=51.81 [1MMQ]=47.08 [1O86]=47.29 [1OQ5]=12.98
               [1UZE]=47.14 [1YDB]=15.59 [2C6N]=-33.71 [2G1M]=11.70
               [2OVX]=50.41 [2W0D]=-24.88 [3HS4]=15.10 [3L2U]=-21.30
               [3P5A]=15.19 [3S3M]=-20.08)
declare -A LIG=([1A42]=1A42_BZU [1BNN]=1BNN_AL1 [1G52]=1G52_F2B
                [1GKC]=1GKC_NFH [1JAQ]=1JAQ_01S [1MMQ]=1MMQ_RRS
                [1O86]=1O86_LPR [1OQ5]=1OQ5_CEL [1UZE]=1UZE_EAL
                [1YDB]=1YDB_AZM [2C6N]=2C6N_LPR [2G1M]=2G1M_4HG
                [2OVX]=2OVX_4MR [2W0D]=2W0D_CGS [3HS4]=3HS4_AZM
                [3L2U]=3L2U_ELV [3P5A]=3P5A_IT2 [3S3M]=3S3M_DLU)

TARGETS=(1A42 1BNN 1G52 1GKC 1JAQ 1MMQ 1O86 1OQ5 1UZE 1YDB
         2C6N 2G1M 2OVX 2W0D 3HS4 3L2U 3P5A 3S3M)

echo "Target  ms    best_affinity"
for T in "${TARGETS[@]}"; do
    t_start=$(date +%s%N)
    $BIN \
      --receptor $METAL/${T}/${T}_receptor.pdbqt \
      --ligand   $METAL/${T}/${LIG[$T]}.pdbqt \
      --out      $OUTDIR/${T}_out.pdbqt \
      --opencl_binary_path $BIN_PATH \
      --center_x ${CX[$T]} --center_y ${CY[$T]} --center_z ${CZ[$T]} \
      --size_x 25 --size_y 25 --size_z 25 \
      --num_modes 9 --gpu_id 0 --ad4zn > /dev/null 2>&1
    t_end=$(date +%s%N)
    ms=$(( (t_end-t_start)/1000000 ))
    best=$(grep "VINA RESULT" $OUTDIR/${T}_out.pdbqt 2>/dev/null | head -1 | awk '{print $4}')
    echo "$T  ${ms}ms  ${best:-N/A}"
done
