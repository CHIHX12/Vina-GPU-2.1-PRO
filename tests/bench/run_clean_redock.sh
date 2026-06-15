#!/bin/bash
# Clean PDBbind redocking benchmark: ligand pdbqt prepared FROM <id>_ligand.sdf (same molecule as the
# obrms reference) so obrms's symmetry-corrected graph matching is reliable. Compares baseline vs
# cavity-biased init. Targets must exist under LigandScope/data/P-L.
PLROOT=/home/cycheng/LigandScope/data/P-L
BIN=/home/cycheng/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
PY2=/home/cycheng/miniforge3/envs/jp_214/bin/python
OBABEL=/home/cycheng/miniforge3/envs/jp_214/bin/obabel
OBRMS=/home/cycheng/miniforge3/envs/jp_214/bin/obrms
PYB=/home/cycheng/miniforge3/bin/python
export VINA_GPU_HOME=/home/cycheng/Vina-GPU-2.1/src/AutoDock-Vina-GPU-2.1
GPU=1
TARGETS="1stp 1bcd 1tng 1cps 1ppc 1add 1bra 3ptb"
best(){ $OBRMS "$1" "$2" 2>/dev/null | awk '{print $NF}' | grep -E '^[0-9]+\.[0-9]+$' | sort -n | head -1; }
printf "%-7s %-6s %-10s %-10s\n" "target" "rot" "baseline" "cav-init"
for t in $TARGETS; do
  P=$(find $PLROOT -maxdepth 2 -type d -name "$t" | head -1)
  [ -z "$P" ] && continue
  [ -f "$P/${t}_ligand.sdf" ] && [ -f "$P/${t}_protein.pdb" ] || continue
  D=redock/$t; mkdir -p $D
  cp $P/${t}_ligand.sdf $D/ref.sdf
  $OBABEL $P/${t}_ligand.sdf -O $D/ligand.pdbqt >/dev/null 2>&1
  [ -f $D/receptor.pdbqt ] || $PY2 /home/cycheng/miniforge3/envs/jp_214/bin/prepare_receptor4.py -r $P/${t}_protein.pdb -o $D/receptor.pdbqt -A hydrogens >/dev/null 2>&1
  read CX CY CZ SX SY SZ <<< $($PYB - $D/ref.sdf <<'PY'
import sys
from rdkit import Chem
m=Chem.SDMolSupplier(sys.argv[1],removeHs=True)[0]; c=m.GetConformer()
xs=[c.GetAtomPosition(i) for i in range(m.GetNumAtoms())]
import statistics as st
print(f"{st.mean(p.x for p in xs):.2f} {st.mean(p.y for p in xs):.2f} {st.mean(p.z for p in xs):.2f} "
      f"{max(max(p.x for p in xs)-min(p.x for p in xs)+8,20):.0f} {max(max(p.y for p in xs)-min(p.y for p in xs)+8,20):.0f} {max(max(p.z for p in xs)-min(p.z for p in xs)+8,20):.0f}")
PY
)
  rot=$(grep -c "^BRANCH" $D/ligand.pdbqt 2>/dev/null)
  B="--receptor $D/receptor.pdbqt --ligand $D/ligand.pdbqt --opencl_binary_path /home/cycheng/Vina-GPU-2.1 --center_x $CX --center_y $CY --center_z $CZ --size_x $SX --size_y $SY --size_z $SZ --thread 8000 --search_depth 64 --gpu_id $GPU"
  $BIN $B --out $D/base.pdbqt >/dev/null 2>&1
  VINA_CAVITY_INIT=1 $BIN $B --out $D/init.pdbqt >/dev/null 2>&1
  printf "%-7s %-6s %-10s %-10s\n" "$t" "$rot" "$(best $D/ref.sdf $D/base.pdbqt)" "$(best $D/ref.sdf $D/init.pdbqt)"
done
echo "CLEAN_DONE"
