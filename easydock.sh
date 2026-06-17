#!/bin/bash
# easydock — foolproof one-command molecular docking with Vina-GPU 2.1 PRO.
#
# For people who are NOT docking experts: just give a receptor and a ligand.
# Everything else (file conversion, binding box, GPU choice, output) is automatic.
#
#   ./easydock.sh  receptor.pdb  ligand.sdf
#   ./easydock.sh  receptor.pdb  "CCO"                       # SMILES directly
#   ./easydock.sh  protein.pdb   drugs/                      # a folder of ligands
#   ./easydock.sh  receptor.pdb  ligA.sdf ligB.sdf --codock  # dock A+B together (co-docking)
#
# Accepts receptor as .pdb or .pdbqt; ligands as .sdf .mol2 .mol .pdb .pdbqt .smi or a SMILES string.
#
# Options (all optional — sensible defaults):
#   --site "X Y Z"   binding-site center (default: auto — a bound ligand in the receptor, else whole protein)
#   --size N         box size in Å (default: auto)
#   --quality Q      fast | balanced | best   (default: balanced)
#   --codock         co-dock all ligands together as one complex (default: dock each separately)
#   --gpu ID         GPU index (default: auto — the freest GPU)
#   --out DIR        output folder (default: ./easydock_out)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src/AutoDock-Vina-GPU-2.1"
BIN="$SRC/AutoDock-Vina-GPU-2-1"
OBABEL="$(command -v obabel || echo /home/cycheng/miniforge3/envs/jp_214/bin/obabel)"

die() { echo "❌ $*" >&2; exit 1; }
say() { echo "▸ $*"; }

[ -x "$BIN" ] || die "docking engine not built. Run:  cd $SRC && make source"
[ -x "$OBABEL" ] || die "Open Babel not found (needed to convert your files). Install it or 'conda activate' the env."

# ---- parse args ----
RECEPTOR=""; LIGS=(); SITE=""; SIZE=""; QUALITY="balanced"; CODOCK=0; GPU="auto"; OUT="./easydock_out"
while [ $# -gt 0 ]; do
  case "$1" in
    --site)    SITE="$2"; shift 2;;
    --size)    SIZE="$2"; shift 2;;
    --quality) QUALITY="$2"; shift 2;;
    --codock)  CODOCK=1; shift;;
    --gpu)     GPU="$2"; shift 2;;
    --out)     OUT="$2"; shift 2;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) if [ -z "$RECEPTOR" ]; then RECEPTOR="$1"; else LIGS+=("$1"); fi; shift;;
  esac
done
[ -n "$RECEPTOR" ] || die "give a receptor first:  ./easydock.sh receptor.pdb ligand.sdf"
[ "${#LIGS[@]}" -gt 0 ] || die "give at least one ligand (file, folder, or SMILES)."
mkdir -p "$OUT"; PREP="$OUT/_prepared"; mkdir -p "$PREP"

case "$QUALITY" in
  fast)     DEPTH=8;;
  balanced) DEPTH=32;;
  best)     DEPTH=64;;
  *) die "--quality must be fast, balanced, or best";;
esac

# max heavy-atom extent (Å) of a prepared ligand PDBQT — used to size the box sensibly
lig_extent() { awk '/^(ATOM|HETATM)/{x=substr($0,31,8)+0;y=substr($0,39,8)+0;z=substr($0,47,8)+0;n++;if(n==1){xm=xM=x;ym=yM=y;zm=zM=z}if(x<xm)xm=x;if(x>xM)xM=x;if(y<ym)ym=y;if(y>yM)yM=y;if(z<zm)zm=z;if(z>zM)zM=z}END{a=xM-xm;b=yM-ym;c=zM-zm;m=a;if(b>m)m=b;if(c>m)m=c;printf "%.1f",m}' "$1"; }

# ---- pick the freest GPU (never fights the user's running workload) ----
if [ "$GPU" = "auto" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  [ -n "$GPU" ] || GPU=0
fi
say "Using GPU $GPU, quality=$QUALITY (search_depth=$DEPTH)"

# ---- prepare the receptor → PDBQT (protein only) ----
prep_receptor() {
  local in="$1" out="$PREP/receptor.pdbqt"
  if [[ "$in" == *.pdbqt ]]; then cp "$in" "$out"; echo "$out"; return; fi
  # protein atoms only (drop HETATM/waters), then convert
  grep -E '^(ATOM|TER)' "$in" > "$PREP/_protein.pdb" || true
  [ -s "$PREP/_protein.pdb" ] || die "no protein ATOM records found in $in"
  "$OBABEL" "$PREP/_protein.pdb" -O "$out" -xr >/dev/null 2>&1 || die "failed to convert receptor $in"
  echo "$out"
}

# ---- auto-detect the binding box from a bound ligand in the receptor PDB ----
auto_box() {
  local pdb="$1"
  [[ "$pdb" == *.pdb ]] || { echo ""; return; }
  # bound ligand = HETATM that is not water / common ions / cryo-additives
  awk '/^HETATM/ {
        r=substr($0,18,3); gsub(/ /,"",r);
        if (r!~/^(HOH|WAT|NA|CL|K|MG|ZN|CA|SO4|PO4|GOL|EDO|ACT|DMS|PEG|MPD|CL1)$/) {
          x=substr($0,31,8)+0; y=substr($0,39,8)+0; z=substr($0,47,8)+0;
          n++; sx+=x; sy+=y; sz+=z;
          if(n==1){xmin=xmax=x;ymin=ymax=y;zmin=zmax=z}
          if(x<xmin)xmin=x; if(x>xmax)xmax=x; if(y<ymin)ymin=y; if(y>ymax)ymax=y; if(z<zmin)zmin=z; if(z>zmax)zmax=z;
        }
      }
      END{ if(n>=5){ printf "%.3f %.3f %.3f %.1f %.1f %.1f\n", sx/n, sy/n, sz/n,
                     (xmax-xmin)+10, (ymax-ymin)+10, (zmax-zmin)+10 } }' "$pdb"
}

# ---- prepare one ligand (any format / SMILES) → PDBQT ----
prep_ligand() {
  local in="$1"; local tag="$2"; local out="$PREP/lig_${tag}.pdbqt"
  if [[ "$in" == *.pdbqt && -f "$in" ]]; then cp "$in" "$out"
  elif [ -f "$in" ]; then "$OBABEL" "$in" -O "$out" -p 7.4 --gen3d >/dev/null 2>&1 || \
                          "$OBABEL" "$in" -O "$out" -p 7.4 >/dev/null 2>&1 || die "failed to convert ligand $in"
  else "$OBABEL" -:"$in" -O "$out" -p 7.4 --gen3d >/dev/null 2>&1 || die "not a valid file or SMILES: $in"; fi
  [ -s "$out" ] || die "ligand preparation produced an empty file for $in"
  echo "$out"
}

say "Preparing receptor: $RECEPTOR"
REC_PDBQT="$(prep_receptor "$RECEPTOR")"

# expand a folder argument into its files
EXPANDED=()
for L in "${LIGS[@]}"; do
  if [ -d "$L" ]; then for f in "$L"/*; do [ -f "$f" ] && EXPANDED+=("$f"); done
  else EXPANDED+=("$L"); fi
done

say "Preparing ${#EXPANDED[@]} ligand(s)..."
LIG_PDBQT=(); i=0
for L in "${EXPANDED[@]}"; do i=$((i+1)); LIG_PDBQT+=("$(prep_ligand "$L" "$i")"); done

# ---- auto-deepen search for flexible ligands ----
# A flexible ligand (many rotatable bonds) has a much larger conformational space; the default depth
# can leave it under-sampled. Bump depth by the largest ligand's rotatable-bond (BRANCH) count so big
# ligands get more MC+BFGS rounds automatically. Never lowers the user's chosen quality.
MAXROT=0
for L in "${LIG_PDBQT[@]}"; do r=$(grep -c "^BRANCH" "$L" 2>/dev/null || echo 0); [ "$r" -gt "$MAXROT" ] && MAXROT=$r; done
if   [ "$MAXROT" -gt 15 ]; then [ "$DEPTH" -lt 96 ] && { DEPTH=96; say "Large flexible ligand ($MAXROT rotatable bonds) → deepening search to $DEPTH"; }
elif [ "$MAXROT" -gt 10 ]; then [ "$DEPTH" -lt 64 ] && { DEPTH=64; say "Flexible ligand ($MAXROT rotatable bonds) → deepening search to $DEPTH"; }
fi

# ---- binding box ----
# Sensible box size = biggest ligand's extent + 10 Å of search room (min 22). A tight box gives
# accurate, interpretable energies; a huge box makes the search wander and the scores unreliable.
LIGSIZE=22
for L in "${LIG_PDBQT[@]}"; do e=$(lig_extent "$L"); s=$(awk -v e="$e" 'BEGIN{printf "%.0f", e+10}'); [ "$s" -gt "$LIGSIZE" ] && LIGSIZE=$s; done
if [ -n "$SITE" ]; then
  read -r CX CY CZ <<< "$SITE"; SX=${SIZE:-$LIGSIZE}; SY=$SX; SZ=$SX
  say "Binding box: center ($CX, $CY, $CZ) you gave, size ${SX} Å (from ligand size)"
else
  BOX="$(auto_box "$RECEPTOR")"
  if [ -n "$BOX" ]; then
    read -r CX CY CZ bx by bz <<< "$BOX"
    # box must hold both the pocket AND the ligand being docked
    SX=$(awk -v a="$bx" -v b="$LIGSIZE" 'BEGIN{print (a>b)?a:b}')
    SY=$(awk -v a="$by" -v b="$LIGSIZE" 'BEGIN{print (a>b)?a:b}')
    SZ=$(awk -v a="$bz" -v b="$LIGSIZE" 'BEGIN{print (a>b)?a:b}')
    say "Binding box: AUTO from a bound ligand in the receptor → center ($CX, $CY, $CZ), size (${SX}×${SY}×${SZ} Å)"
  else
    read -r CX CY CZ <<< "$(awk '/^(ATOM|HETATM)/{x=substr($0,31,8)+0;y=substr($0,39,8)+0;z=substr($0,47,8)+0;n++;sx+=x;sy+=y;sz+=z}END{printf "%.3f %.3f %.3f",sx/n,sy/n,sz/n}' "$REC_PDBQT")"
    SX=${SIZE:-30}; SY=$SX; SZ=$SX
    say "⚠ No bound ligand found — docking near the protein centre (center ($CX, $CY, $CZ), size ${SX} Å). For accuracy pass --site \"X Y Z\"."
  fi
fi

# ---- print the affinity from an output pdbqt ----
report() { local f="$1" name="$2"; local e
  e=$(grep -m1 "REMARK VINA RESULT" "$f" 2>/dev/null | awk '{print $4}')
  printf "  %-28s  %s kcal/mol  →  %s\n" "$name" "${e:-?}" "$f"; }

echo; say "Docking on GPU $GPU ..."; echo
export VINA_GPU_HOME="$SRC"
COMMON=(--receptor "$REC_PDBQT" --center_x "$CX" --center_y "$CY" --center_z "$CZ"
        --size_x "$SX" --size_y "$SY" --size_z "$SZ"
        --thread 8000 --search_depth "$DEPTH" --gpu_id "$GPU" --opencl_binary_path "$SRC")

if [ "$CODOCK" -eq 1 ] && [ "${#LIG_PDBQT[@]}" -ge 2 ]; then
  OUTF="$OUT/codock_complex.pdbqt"
  say "Co-docking ${#LIG_PDBQT[@]} ligands together..."
  "$BIN" "${COMMON[@]}" --co_dock "${LIG_PDBQT[@]}" --out "$OUTF" >/dev/null
  echo; echo "✅ Done. Co-docked complex:"; report "$OUTF" "complex (${#LIG_PDBQT[@]} ligands)"
else
  echo; echo "✅ Done. Results (best binding energy — more negative = stronger):"
  for L in "${LIG_PDBQT[@]}"; do
    base="$(basename "$L" .pdbqt)"; OUTF="$OUT/${base}_docked.pdbqt"
    "$BIN" "${COMMON[@]}" --ligand "$L" --out "$OUTF" >/dev/null
    report "$OUTF" "$base"
  done
fi
echo; say "Output folder: $OUT"
