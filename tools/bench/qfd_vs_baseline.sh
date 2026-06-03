#!/usr/bin/env bash
# qfd_vs_baseline.sh — Run QFD vs baseline comparison on Metal_enzymes dataset
# Usage: bash qfd_vs_baseline.sh [output_dir]

set -euo pipefail

export VINA_GPU_HOME="/home/cycheng/Vina-GPU-2.1/src/AutoDock-Vina-GPU-2.1"
METAL_DIR="/home/cycheng/LigandScope/data/Metal_enzymes"
VINA="/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1"
PREP_QFD="/home/cycheng/Vina-GPU-2.1/tools/prep/prep_qfd_grids.py"
OUT_DIR="${1:-/tmp/qfd_bench_$(date +%Y%m%d_%H%M%S)}"
OCL_CACHE="$OUT_DIR/ocl_cache"

mkdir -p "$OUT_DIR" "$OCL_CACHE"

echo "QFD vs Baseline benchmark — $(date)"
echo "Output: $OUT_DIR"
echo ""

TARGETS=$(ls "$METAL_DIR" | grep -v '\.tsv$')

for TARGET in $TARGETS; do
    TARGET_DIR="$METAL_DIR/$TARGET"
    [[ -d "$TARGET_DIR" ]] || continue

    RECEPTOR="$TARGET_DIR/${TARGET}_receptor.pdbqt"
    CRYSTAL="$TARGET_DIR/${TARGET}_ligand.pdbqt"
    LIGAND_SRC=$(ls "$TARGET_DIR/${TARGET}"_*.pdbqt 2>/dev/null | grep -v receptor | grep -v ligand | head -1)

    [[ -f "$RECEPTOR" ]] || { echo "SKIP $TARGET: no receptor"; continue; }
    [[ -f "$CRYSTAL"  ]] || { echo "SKIP $TARGET: no crystal";  continue; }
    [[ -f "$LIGAND_SRC" ]] || { echo "SKIP $TARGET: no ligand";  continue; }

    # Get box center from crystal ligand
    CENTER_X=$(python3 -c "
import re
coords=[]
for line in open('$CRYSTAL'):
    if line.startswith(('ATOM','HETATM')):
        coords.append(float(line[30:38]))
print(f'{sum(coords)/len(coords):.3f}')
")
    CENTER_Y=$(python3 -c "
coords=[]
for line in open('$CRYSTAL'):
    if line.startswith(('ATOM','HETATM')):
        coords.append(float(line[38:46]))
print(f'{sum(coords)/len(coords):.3f}')
")
    CENTER_Z=$(python3 -c "
coords=[]
for line in open('$CRYSTAL'):
    if line.startswith(('ATOM','HETATM')):
        coords.append(float(line[46:54]))
print(f'{sum(coords)/len(coords):.3f}')
")

    echo "── $TARGET  box center: ($CENTER_X, $CENTER_Y, $CENTER_Z)"

    for MODE in baseline qfd; do
        WORK="$OUT_DIR/${TARGET}_${MODE}"
        mkdir -p "$WORK/ligands" "$WORK/output"
        cp "$RECEPTOR" "$WORK/receptor.pdbqt"
        cp "$LIGAND_SRC" "$WORK/ligands/"

        cat > "$WORK/config.txt" <<EOF
receptor           = $WORK/receptor.pdbqt
ligand_directory   = $WORK/ligands
output_directory   = $WORK/output
opencl_binary_path = $OCL_CACHE
center_x = $CENTER_X
center_y = $CENTER_Y
center_z = $CENTER_Z
size_x   = 22
size_y   = 22
size_z   = 22
thread   = 8000
search_depth = 20
EOF

        if [[ "$MODE" == "qfd" ]]; then
            python3 "$PREP_QFD" \
                --receptor "$WORK/receptor.pdbqt" \
                --center_x "$CENTER_X" --center_y "$CENTER_Y" --center_z "$CENTER_Z" \
                --size_x 22 --size_y 22 --size_z 22 \
                --output_dir "$WORK" 2>/dev/null
        fi

        START=$(date +%s%3N)
        "$VINA" --config "$WORK/config.txt" > "$WORK/vina.log" 2>&1
        END=$(date +%s%3N)
        ELAPSED=$(( END - START ))

        OUT_PDBQT=$(ls "$WORK/output/"*.pdbqt 2>/dev/null | head -1)
        if [[ -z "$OUT_PDBQT" ]]; then
            echo "  $MODE: FAILED (no output)"
            continue
        fi

        python3 - "$OUT_PDBQT" "$CRYSTAL" "$MODE" "$ELAPSED" <<'PYEOF'
import sys, numpy as np

def parse(path):
    models, cur = [], {'coords':[],'score':None}
    for line in open(path):
        if line.startswith('MODEL'): cur = {'coords':[],'score':None}
        elif 'VINA RESULT' in line:
            p = line.split(); cur['score'] = float(p[3])
        elif line.startswith(('ATOM','HETATM')):
            cur['coords'].append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
        elif line.startswith('ENDMDL'):
            if cur['coords']: models.append(cur)
    if not models and cur['coords']: models.append(cur)
    return models

out_f, crystal_f, mode, elapsed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
poses = parse(out_f)
crystal = parse(crystal_f)
if not poses or not crystal:
    print(f"  {mode}: parse error")
    sys.exit(0)
ref = np.array(crystal[0]['coords'])
rmsds = []
for p in poses:
    c = np.array(p['coords'])
    if len(c) == len(ref):
        rmsds.append(float(np.sqrt(np.mean(np.sum((c-ref)**2,axis=1)))))
    else:
        rmsds.append(999.0)
top1 = rmsds[0]
best = min(rmsds)
t1_ok = "✅" if top1 < 2.0 else "❌"
best_ok = "✅" if best < 2.0 else "❌"
print(f"  {mode:8s}: top-1={top1:.3f}Å{t1_ok}  best={best:.3f}Å{best_ok}  ({elapsed}ms, {len(poses)} poses)")
# Save RMSD summary
with open(out_f.replace('.pdbqt','.rmsd'), 'w') as f:
    f.write(f"top1={top1:.3f}\nbest={best:.3f}\n")
PYEOF
    done
    echo ""
done

echo "Done — results in $OUT_DIR"
