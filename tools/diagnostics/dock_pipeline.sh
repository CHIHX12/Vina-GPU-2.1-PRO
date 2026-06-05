#!/usr/bin/env bash
# dock_pipeline.sh — Vina-GPU 2.1 一鍵對接管道
#
# 用法：
#   # 自動搜尋框（從共晶配體）
#   bash tools/dock_pipeline.sh -r protein.pdb -s ligands.smi -l ref_ligand.sdf -o results/
#
#   # 手動搜尋框
#   bash tools/dock_pipeline.sh -r protein.pdb -s ligands.smi \
#       -x 10.0,20.0,30.0 -X 20,20,20 -o results/
#
#   # 完整選項
#   bash tools/dock_pipeline.sh \
#       -r protein.pdb         # 受體 PDB（需含 H）
#       -s compounds.smi       # SMILES 批次檔（每行：SMILES  名稱）
#       -l ref_ligand.sdf      # 共晶配體（自動搜尋框，與 -x/-X 擇一）
#       -x "10.5,20.1,30.2"    # 搜尋框中心 X,Y,Z
#       -X "20,20,20"          # 搜尋框尺寸 X,Y,Z（Å）
#       -o results/            # 輸出目錄（預設：./docking_out）
#       -t 8000                # GPU threads（預設：8000）
#       -d 32                  # search_depth（預設：32）
#       -g all                 # GPU ID（預設：all）
#       -p 8.0                 # autobox padding（預設：8.0 Å）
#       -n 9                   # 輸出 pose 數（預設：9）
#       --pH 7.4               # 配體質子化 pH（預設：7.4）
#
# 管道步驟：
#   1. 受體準備：mk_prepare_receptor.py (meeko)
#   2. 配體準備：smiles_to_pdbqt.py (dimorphite-dl + RDKit ETKDGv3 + meeko)
#   3. GPU 對接：AutoDock-Vina-GPU-2-1
#   4. 結果彙整：results.tsv（按結合親和力排序）
#
# 依賴：conda activate vina_docking（包含 meeko, rdkit, dimorphite_dl==1.2.5）
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── 工具路徑（從此腳本位置自動推算）─────────────────────────────────────────
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_VINA_DIR="$(dirname "$(dirname "$_SCRIPT_DIR")")"
_VINA_BIN="$_VINA_DIR/AutoDock-Vina-GPU-2-1"
_SMILES_PY="$_SCRIPT_DIR/smiles_to_pdbqt.py"

# ── 預設值 ───────────────────────────────────────────────────────────────────
_RECEPTOR=""
_SMILES_FILE=""
_REF_LIGAND=""
_CENTER=""
_SIZE=""
_OUTDIR="./docking_out"
_THREAD=8000
_DEPTH=32
_GPU_ID="all"
_PADDING=8.0
_NUM_MODES=9
_PH=7.4

# ── 解析參數 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--receptor)      _RECEPTOR="$2";    shift 2 ;;
        -s|--smiles)        _SMILES_FILE="$2"; shift 2 ;;
        -l|--ref-ligand)    _REF_LIGAND="$2";  shift 2 ;;
        -x|--center)        _CENTER="$2";      shift 2 ;;
        -X|--size)          _SIZE="$2";        shift 2 ;;
        -o|--output)        _OUTDIR="$2";      shift 2 ;;
        -t|--thread)        _THREAD="$2";      shift 2 ;;
        -d|--depth)         _DEPTH="$2";       shift 2 ;;
        -g|--gpu)           _GPU_ID="$2";      shift 2 ;;
        -p|--padding)       _PADDING="$2";     shift 2 ;;
        -n|--num-modes)     _NUM_MODES="$2";   shift 2 ;;
        --pH)               _PH="$2";          shift 2 ;;
        -h|--help)
            head -40 "${BASH_SOURCE[0]}" | grep "^#" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "未知參數: $1"; exit 1 ;;
    esac
done

# ── 必要參數檢查 ─────────────────────────────────────────────────────────────
[[ -z "$_RECEPTOR"    ]] && { echo "錯誤：缺少 -r receptor.pdb"; exit 1; }
[[ -z "$_SMILES_FILE" ]] && { echo "錯誤：缺少 -s ligands.smi";  exit 1; }
[[ ! -f "$_RECEPTOR"  ]] && { echo "錯誤：找不到受體：$_RECEPTOR"; exit 1; }
[[ ! -f "$_SMILES_FILE" ]] && { echo "錯誤：找不到 SMILES 檔：$_SMILES_FILE"; exit 1; }

if [[ -z "$_REF_LIGAND" && -z "$_CENTER" ]]; then
    echo "錯誤：需要 -l ref_ligand.sdf（自動框）或 -x cx,cy,cz -X sx,sy,sz（手動框）"
    exit 1
fi
if [[ -n "$_REF_LIGAND" && ! -f "$_REF_LIGAND" ]]; then
    echo "錯誤：找不到參考配體：$_REF_LIGAND"
    exit 1
fi

# ── 確認 conda 環境 ──────────────────────────────────────────────────────────
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "警告：CONDA_PREFIX 未設定，請先 conda activate vina_docking"
fi
_BIN="${CONDA_PREFIX:-}/bin"
_PY="PYTHONNOUSERSITE=1 python3"

# ── 建立輸出目錄結構 ─────────────────────────────────────────────────────────
mkdir -p "$_OUTDIR/receptor" "$_OUTDIR/ligands" "$_OUTDIR/poses"
_LOG="$_OUTDIR/pipeline.log"
exec > >(tee -a "$_LOG") 2>&1

echo "════════════════════════════════════════════════════════"
echo " Vina-GPU 2.1 一鍵對接管道"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  受體     : $_RECEPTOR"
echo "  配體 SMI : $_SMILES_FILE"
echo "  輸出目錄 : $_OUTDIR"
echo "  GPU      : $_GPU_ID  threads=$_THREAD  depth=$_DEPTH"
echo ""

# ── STEP 1：受體準備 ─────────────────────────────────────────────────────────
echo "═══ STEP 1：受體準備 ═══"
_REC_BASE="$_OUTDIR/receptor/receptor"
_REC_PDBQT="${_REC_BASE}.pdbqt"
_BOX_FILE="${_REC_BASE}_vina_box.txt"

if [[ -n "$_REF_LIGAND" ]]; then
    echo "  模式：autobox（參考配體 $_REF_LIGAND，padding=${_PADDING}Å）"
    eval "$_PY \"$_BIN/mk_prepare_receptor.py\"" \
        --read_pdb "$_RECEPTOR" \
        -o "$_REC_BASE" -p \
        --box_enveloping "$_REF_LIGAND" \
        --padding "$_PADDING"
    echo "  → 受體：$_REC_PDBQT"
    echo "  → 搜尋框：$_BOX_FILE"
else
    echo "  模式：手動搜尋框（center=$_CENTER，size=$_SIZE）"
    eval "$_PY \"$_BIN/mk_prepare_receptor.py\"" \
        --read_pdb "$_RECEPTOR" \
        -o "$_REC_BASE" -p
    echo "  → 受體：$_REC_PDBQT"
    # 寫入 vina_box.txt（手動框）
    IFS=',' read -r _CX _CY _CZ <<< "$_CENTER"
    IFS=',' read -r _SX _SY _SZ <<< "$_SIZE"
    cat > "$_BOX_FILE" <<EOF
center_x = ${_CX}
center_y = ${_CY}
center_z = ${_CZ}
size_x   = ${_SX}
size_y   = ${_SY}
size_z   = ${_SZ}
EOF
    echo "  → 搜尋框：$_BOX_FILE"
fi

[[ ! -f "$_REC_PDBQT" ]] && { echo "錯誤：受體 PDBQT 未生成：$_REC_PDBQT"; exit 1; }
echo ""

# ── 解析搜尋框 ────────────────────────────────────────────────────────────────
_cx=$(grep center_x "$_BOX_FILE" | awk '{print $3}')
_cy=$(grep center_y "$_BOX_FILE" | awk '{print $3}')
_cz=$(grep center_z "$_BOX_FILE" | awk '{print $3}')
_sx=$(grep size_x   "$_BOX_FILE" | awk '{print $3}')
_sy=$(grep size_y   "$_BOX_FILE" | awk '{print $3}')
_sz=$(grep size_z   "$_BOX_FILE" | awk '{print $3}')

echo "  搜尋框："
echo "    center = ${_cx}, ${_cy}, ${_cz}"
echo "    size   = ${_sx} × ${_sy} × ${_sz} Å"
echo ""

# ── STEP 2：配體準備 ─────────────────────────────────────────────────────────
echo "═══ STEP 2：配體準備（SMILES → PDBQT）═══"
_LIG_DIR="$_OUTDIR/ligands"
_T0=$SECONDS

eval "$_PY \"$_SMILES_PY\"" \
    -i "$_SMILES_FILE" \
    -o "$_LIG_DIR" \
    --pH "$_PH" \
    -v

_n_pdbqt=$(find "$_LIG_DIR" -name "*.pdbqt" | wc -l)
echo "  → 已生成 ${_n_pdbqt} 個 PDBQT（耗時 $(( SECONDS - _T0 )) 秒）"
echo ""

[[ "$_n_pdbqt" -eq 0 ]] && { echo "錯誤：沒有成功的配體 PDBQT"; exit 1; }

# ── STEP 3：GPU 對接 ─────────────────────────────────────────────────────────
echo "═══ STEP 3：GPU 對接（Vina-GPU 2.1）═══"
_POSES_DIR="$_OUTDIR/poses"
_T1=$SECONDS

"$_VINA_BIN" \
    --receptor          "$_REC_PDBQT" \
    --ligand_directory  "$_LIG_DIR" \
    --output_directory  "$_POSES_DIR" \
    --center_x "$_cx" --center_y "$_cy" --center_z "$_cz" \
    --size_x   "$_sx" --size_y   "$_sy" --size_z   "$_sz" \
    --thread       "$_THREAD" \
    --search_depth "$_DEPTH" \
    --gpu_id       "$_GPU_ID" \
    --num_modes    "$_NUM_MODES" \
    --opencl_binary_path "$_VINA_DIR"

_n_out=$(find "$_POSES_DIR" -name "*_out.pdbqt" | wc -l)
echo ""
echo "  → 對接完成：${_n_out} 個結果（耗時 $(( SECONDS - _T1 )) 秒）"
echo ""

# ── STEP 4：彙整結果 → results.tsv ──────────────────────────────────────────
echo "═══ STEP 4：結果彙整 ═══"
_TSV="$_OUTDIR/results.tsv"

{
    printf "Rank\tName\tBest_Affinity_kcal_mol\tMode2_Affinity\tMode3_Affinity\tFile\n"
    # 從每個 _out.pdbqt 的 REMARK VINA RESULT 行提取結合親和力
    for f in "$_POSES_DIR"/*_out.pdbqt; do
        [[ -f "$f" ]] || continue
        _name=$(basename "$f" _out.pdbqt)
        # 讀取前 3 個 REMARK VINA RESULT 行
        _modes=($(grep "^REMARK VINA RESULT" "$f" | awk '{print $4}'))
        _e1="${_modes[0]:-NA}"
        _e2="${_modes[1]:-NA}"
        _e3="${_modes[2]:-NA}"
        printf "%s\t%s\t%s\t%s\t%s\n" "?" "$_name" "$_e1" "$_e2" "$_e3"
    done
} | sort -t$'\t' -k3 -n | awk -v OFS='\t' 'NR==1{print; next} {$1=NR-1; print}' > "$_TSV"

_n_results=$(( $(wc -l < "$_TSV") - 1 ))
echo "  → $_TSV（$_n_results 筆結果，按結合親和力排序）"
echo ""

# 顯示前 10 名
echo "═══ 前 10 名配體 ═══"
head -11 "$_TSV" | column -t -s $'\t'
echo ""

echo "════════════════════════════════════════════════════════"
echo " 完成！總耗時 ${SECONDS} 秒"
echo " 結果：$_OUTDIR/"
echo "   受體 PDBQT : $_REC_PDBQT"
echo "   配體 PDBQT : $_LIG_DIR/"
echo "   對接姿勢   : $_POSES_DIR/"
echo "   排名表格   : $_TSV"
echo "   完整日誌   : $_LOG"
echo "════════════════════════════════════════════════════════"
