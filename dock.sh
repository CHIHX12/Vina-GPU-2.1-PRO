#!/bin/bash
# dock.sh — One-command virtual screening with Vina-GPU 2.1 PRO
#
# Usage:
#   ./dock.sh --receptor REC.pdbqt --ligands LIGDIR/ --ref REF_LIG.pdbqt [options]
#   ./dock.sh --receptor REC.pdbqt --ligands LIGDIR/ --box "cx cy cz"    [options]
#
# Required:
#   --receptor / -r  PATH   Receptor PDBQT file
#   --ligands  / -l  DIR    Directory of ligand PDBQT files
#
# Box (one required):
#   --ref       PATH   Co-crystal / reference ligand → auto-compute box center + size
#   --box       "X Y Z"  Box center coordinates (manual)
#   --size      N or "SX SY SZ"  Box size in Å (default: auto from --ref, else 25)
#
# Options:
#   --out  / -o  DIR   Output directory   (default: ./output)
#   --depth      N     search_depth       (default: 4)
#   --gpu        ID    GPU: all, 0, 1, 0,1  (default: all)
#   --threads    N     OpenCL threads     (default: 8000)
#   --sif        PATH  Singularity image  (default: ./autodock-vina-gpu.sif)

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
RECEPTOR=""; LIGANDS=""; REF_LIG=""; BOX_CENTER=""
BOX_SIZE="auto"
OUT_DIR="$(pwd)/output"
DEPTH=4; GPU="all"; THREADS=8000
SIF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/autodock-vina-gpu.sif"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --receptor|-r) RECEPTOR="$2"; shift 2 ;;
        --ligands|-l)  LIGANDS="$2";  shift 2 ;;
        --ref)         REF_LIG="$2";  shift 2 ;;
        --box)         BOX_CENTER="$2"; shift 2 ;;
        --size)        BOX_SIZE="$2"; shift 2 ;;
        --out|-o)      OUT_DIR="$2";  shift 2 ;;
        --depth)       DEPTH="$2";   shift 2 ;;
        --gpu)         GPU="$2";     shift 2 ;;
        --threads)     THREADS="$2"; shift 2 ;;
        --sif)         SIF="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── validate required args ────────────────────────────────────────────────────
err() { echo "ERROR: $*" >&2; exit 1; }

[[ -z "$RECEPTOR" ]] && err "--receptor is required"
[[ -z "$LIGANDS"  ]] && err "--ligands is required"

if [[ -z "$BOX_CENTER" && -z "$REF_LIG" ]]; then
    echo "ERROR: box not specified." >&2
    echo "" >&2
    echo "  Option A — auto-box from a reference/co-crystal ligand:" >&2
    echo "    --ref /path/to/ref_ligand.pdbqt" >&2
    echo "" >&2
    echo "  Option B — manual box center:" >&2
    echo "    --box \"10.5 20.3 -5.1\"   (and optionally --size 25)" >&2
    exit 1
fi

[[ ! -f "$RECEPTOR" ]] && err "receptor not found: $RECEPTOR"
[[ ! -d "$LIGANDS"  ]] && err "ligand directory not found: $LIGANDS"
[[ ! -f "$SIF"      ]] && {
    echo "ERROR: SIF not found: $SIF" >&2
    echo "  Build it first:" >&2
    echo "    docker build -t autodock-vina-gpu ." >&2
    echo "    singularity build autodock-vina-gpu.sif docker-daemon://autodock-vina-gpu:latest" >&2
    exit 1
}

# ── resolve absolute paths ────────────────────────────────────────────────────
RECEPTOR=$(realpath "$RECEPTOR")
LIGANDS=$(realpath "$LIGANDS")
OUT_DIR=$(mkdir -p "$OUT_DIR" && realpath "$OUT_DIR")
SIF=$(realpath "$SIF")

# ── auto-compute box from reference ligand ────────────────────────────────────
if [[ -n "$REF_LIG" ]]; then
    [[ ! -f "$REF_LIG" ]] && err "reference ligand not found: $REF_LIG"
    REF_LIG=$(realpath "$REF_LIG")

    BOX_INFO=$(python3 - "$REF_LIG" "$BOX_SIZE" <<'PYEOF'
import sys, math

path      = sys.argv[1]
user_size = sys.argv[2]   # "auto", "25", or "sx sy sz"

coords = []
with open(path) as f:
    for line in f:
        if line.startswith(('ATOM  ', 'HETATM')):
            try:
                coords.append((float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])))
            except Exception:
                pass

if not coords:
    print("ERROR: no ATOM/HETATM records found in reference ligand", file=sys.stderr)
    sys.exit(1)

cx = sum(x for x, y, z in coords) / len(coords)
cy = sum(y for x, y, z in coords) / len(coords)
cz = sum(z for x, y, z in coords) / len(coords)

if user_size == "auto":
    xs = [x for x, y, z in coords]
    ys = [y for x, y, z in coords]
    zs = [z for x, y, z in coords]
    # ligand extent + 10 Å margin on each side
    sx = max(max(xs) - min(xs) + 10, 15)
    sy = max(max(ys) - min(ys) + 10, 15)
    sz = max(max(zs) - min(zs) + 10, 15)
else:
    parts = user_size.split()
    if len(parts) == 1:
        sx = sy = sz = float(parts[0])
    elif len(parts) == 3:
        sx, sy, sz = float(parts[0]), float(parts[1]), float(parts[2])
    else:
        print(f"ERROR: --size expects 1 or 3 values, got: {user_size}", file=sys.stderr)
        sys.exit(1)

print(f"{cx:.3f} {cy:.3f} {cz:.3f} {sx:.1f} {sy:.1f} {sz:.1f}")
PYEOF
    )

    BOX_CENTER="$(echo "$BOX_INFO" | awk '{print $1, $2, $3}')"
    BOX_SIZE="$(echo "$BOX_INFO" | awk '{print $4, $5, $6}')"
fi

# ── parse box center / size ───────────────────────────────────────────────────
read CX CY CZ <<< "$BOX_CENTER"
if [[ "$BOX_SIZE" == "auto" ]]; then
    SX=25; SY=25; SZ=25
elif [[ $(echo "$BOX_SIZE" | wc -w) -eq 1 ]]; then
    SX="$BOX_SIZE"; SY="$BOX_SIZE"; SZ="$BOX_SIZE"
else
    read SX SY SZ <<< "$BOX_SIZE"
fi

# ── write config ──────────────────────────────────────────────────────────────
OCL_CACHE="/tmp/vina-gpu-cache-$$"
mkdir -p "$OCL_CACHE"

CONFIG="$OUT_DIR/config.txt"
cat > "$CONFIG" <<CFGEOF
receptor           = $RECEPTOR
ligand_directory   = $LIGANDS
output_directory   = $OUT_DIR
opencl_binary_path = $OCL_CACHE

center_x = $CX
center_y = $CY
center_z = $CZ
size_x   = $SX
size_y   = $SY
size_z   = $SZ

thread       = $THREADS
search_depth = $DEPTH
CFGEOF

# ── summary ───────────────────────────────────────────────────────────────────
LIG_COUNT=$(ls "$LIGANDS"/*.pdbqt 2>/dev/null | wc -l)

echo "================================================================"
echo " Vina-GPU 2.1 PRO — Virtual Screening"
echo "================================================================"
printf "  Receptor : %s\n" "$(basename "$RECEPTOR")"
printf "  Ligands  : %s  (%d ligands)\n" "$LIGANDS" "$LIG_COUNT"
if [[ -n "$REF_LIG" ]]; then
    printf "  Box      : center=(%.3f, %.3f, %.3f)  size=(%.1f×%.1f×%.1f Å)\n" \
           "$CX" "$CY" "$CZ" "$SX" "$SY" "$SZ"
    printf "  Box src  : auto from %s\n" "$(basename "$REF_LIG")"
else
    printf "  Box      : center=(%.3f, %.3f, %.3f)  size=(%.1f×%.1f×%.1f Å)\n" \
           "$CX" "$CY" "$CZ" "$SX" "$SY" "$SZ"
fi
printf "  GPU      : %s   depth=%s   threads=%s\n" "$GPU" "$DEPTH" "$THREADS"
printf "  Output   : %s\n" "$OUT_DIR"
echo "================================================================"
echo ""

# ── bind mounts (same absolute paths inside/outside container) ────────────────
REC_DIR=$(dirname "$RECEPTOR")
BINDS="-B ${REC_DIR}:${REC_DIR}:ro"
BINDS="$BINDS -B ${LIGANDS}:${LIGANDS}:ro"
BINDS="$BINDS -B ${OUT_DIR}:${OUT_DIR}"
BINDS="$BINDS -B ${OCL_CACHE}:${OCL_CACHE}"

# ── run ───────────────────────────────────────────────────────────────────────
singularity run --nv \
    $BINDS \
    "$SIF" \
    --config "$CONFIG" \
    --gpu_id "$GPU"

echo ""
echo "================================================================"
echo " Done.  Results → $OUT_DIR"
echo "================================================================"
