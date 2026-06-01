#!/usr/bin/env bash
# setup_env.sh — AutoDock Vina-GPU 2.1 工具環境設定（2024-2025 最新工具鏈）
#
# 用法：
#   source /path/to/Vina-GPU-2.1/tools/setup_env.sh
#
# 或加入 ~/.bashrc：
#   echo 'source /path/to/Vina-GPU-2.1/tools/setup_env.sh' >> ~/.bashrc
#
# 工具鏈說明：
#   受體準備：mk_prepare_receptor.py (meeko) — 取代 prepare_receptor4.py
#   配體準備：mk_prepare_ligand.py    (meeko) — 取代 prepare_ligand4.py
#   SMILES→PDBQT：smiles_to_pdbqt.py（dimorphite-dl + RDKit + meeko）
#
#   舊工具 (prepare_receptor4.py / prepare_ligand4.py) 仍保留為備用。

# ── 確認 conda env 已啟動 ──────────────────────────────────────────────────
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "[setup_env] 警告：CONDA_PREFIX 未設定，請先 conda activate jp_214"
    return 1
fi
_BIN="$CONDA_PREFIX/bin"

# ── Vina-GPU 2.1 binary ────────────────────────────────────────────────────
_VINA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VINA_BIN="$_VINA_DIR/src/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1"
export VINA_OCL="$_VINA_DIR"
alias vina-gpu="$VINA_BIN --opencl_binary_path $VINA_OCL"

# ── 受體準備（meeko Python 3，官方推薦）────────────────────────────────────
#
# 用法：
#   prep_receptor proteinH.pdb receptor.pdbqt
#
# 注意：輸入 PDB 必須先加好 H（用 reduce2 或 pdb2pqr）
#   reduce2 protein.pdb > proteinH.pdb
#
prep_receptor() {
    local input_pdb="${1:?用法: prep_receptor <input.pdb> [output.pdbqt]}"
    local output="${2:-${input_pdb%.pdb}.pdbqt}"
    local basename="${output%.pdbqt}"
    PYTHONNOUSERSITE=1 python3 "$_BIN/mk_prepare_receptor.py" \
        --read_pdb "$input_pdb" -o "$basename" -p
    echo "→ 受體輸出：${basename}.pdbqt"
}

# 帶 autobox（從共晶配體自動計算搜尋框）
prep_receptor_autobox() {
    local input_pdb="${1:?用法: prep_receptor_autobox <input.pdb> <ref_ligand.sdf> [output_basename]}"
    local ref_lig="${2:?需要參考配體 SDF 檔}"
    local basename="${3:-${input_pdb%.pdb}}"
    PYTHONNOUSERSITE=1 python3 "$_BIN/mk_prepare_receptor.py" \
        --read_pdb "$input_pdb" -o "$basename" -p \
        --box_enveloping "$ref_lig" --padding 8.0
    echo "→ 受體輸出：${basename}.pdbqt"
    echo "→ Vina box：${basename}_vina_box.txt"
}

# ── 配體準備（meeko Python 3，從 SDF/mol2 轉 PDBQT）──────────────────────
#
# 用法：
#   prep_ligand ligand.sdf ligand.pdbqt
#   prep_ligand_batch ligands_dir/ output_dir/
#
prep_ligand() {
    local input="${1:?用法: prep_ligand <input.sdf|mol2> [output.pdbqt]}"
    local output="${2:-${input%.*}.pdbqt}"
    PYTHONNOUSERSITE=1 python3 "$_BIN/mk_prepare_ligand.py" \
        -i "$input" -o "$output"
    echo "→ 配體輸出：$output"
}

# 批次 SDF → PDBQT（多配體）
prep_ligand_batch() {
    local indir="${1:?用法: prep_ligand_batch <input_dir/> [output_dir/]}"
    local outdir="${2:-./ligands}"
    mkdir -p "$outdir"
    local count=0
    for f in "$indir"/*.sdf "$indir"/*.mol2; do
        [[ -f "$f" ]] || continue
        local name="$(basename "${f%.*}")"
        PYTHONNOUSERSITE=1 python3 "$_BIN/mk_prepare_ligand.py" \
            -i "$f" -o "$outdir/${name}.pdbqt" 2>/dev/null && (( count++ ))
    done
    echo "→ 批次完成：$count 個配體 → $outdir/"
}

# ── SMILES → PDBQT（dimorphite-dl + RDKit + meeko 完整管道）────────────────
#
# 用法：
#   smiles2pdbqt "C1C2CC3CC1CC(C2)(C3)N" amantadine
#   smiles2pdbqt "C1C2CC3CC1CC(C2)(C3)N" amantadine ./ligands/
#
smiles2pdbqt() {
    local smi="${1:?用法: smiles2pdbqt <SMILES> <name> [outdir]}"
    local name="${2:-mol}"
    local outdir="${3:-.}"
    PYTHONNOUSERSITE=1 python3 \
        "$(dirname "${BASH_SOURCE[0]}")/smiles_to_pdbqt.py" \
        -s "$smi" -n "$name" -o "$outdir"
}

# 批次 .smi 檔（每行：SMILES  名稱）
smiles_batch() {
    local smi_file="${1:?用法: smiles_batch <input.smi> [outdir]}"
    local outdir="${2:-./ligands}"
    [[ -f "$smi_file" ]] || { echo "錯誤：找不到 $smi_file"; return 1; }
    PYTHONNOUSERSITE=1 python3 \
        "$(dirname "${BASH_SOURCE[0]}")/smiles_to_pdbqt.py" \
        -i "$smi_file" -o "$outdir"
}

# ── 舊版 MGLTools 備用（Python 2.7，prepare_receptor4.py 等）──────────────
if [[ -f "$_BIN/prepare_receptor4.py" ]]; then
    alias prepare_receptor4.py="python $_BIN/prepare_receptor4.py"
    alias prepare_ligand4.py="python $_BIN/prepare_ligand4.py"
    alias prepare_gpf4.py="python $_BIN/prepare_gpf4.py"
    alias prepare_dpf4.py="python $_BIN/prepare_dpf4.py"
    alias prepare_flexreceptor4.py="python $_BIN/prepare_flexreceptor4.py"
    alias prepare_pdb_split_alt_confs.py="python $_BIN/prepare_pdb_split_alt_confs.py"
    _MGLTOOLS_STATUS="✓ (備用)"
else
    _MGLTOOLS_STATUS="✗ 未安裝"
fi

# ── 確認 meeko 可用 ────────────────────────────────────────────────────────
if PYTHONNOUSERSITE=1 python3 -c "import meeko" 2>/dev/null; then
    _MEEKO_VER=$(PYTHONNOUSERSITE=1 python3 -c "import meeko; print(meeko.__version__)" 2>/dev/null)
    _MEEKO_STATUS="✓ v${_MEEKO_VER}"
else
    _MEEKO_STATUS="✗ 未安裝 → pip install meeko"
fi

# ── 載入摘要 ───────────────────────────────────────────────────────────────
echo "[setup_env] 已載入 Vina-GPU 2.1 工具鏈"
echo ""
echo "  受體準備："
echo "    prep_receptor <proteinH.pdb>              → meeko ${_MEEKO_STATUS}"
echo "    prep_receptor_autobox <pdb> <ref.sdf>     → meeko + 自動 box"
echo ""
echo "  配體準備："
echo "    prep_ligand <ligand.sdf>                  → meeko ${_MEEKO_STATUS}"
echo "    prep_ligand_batch <sdf_dir/> <out_dir/>   → 批次"
echo "    smiles2pdbqt 'SMILES' name [outdir]       → dimorphite-dl+RDKit+meeko"
echo "    smiles_batch compounds.smi [outdir]       → 批次 SMILES"
echo ""
echo "  舊版 MGLTools：${_MGLTOOLS_STATUS}"
echo "  vina-gpu：$VINA_BIN"
echo ""
echo "快速測試："
echo "  prep_receptor proteinH.pdb"
echo "  smiles2pdbqt 'CC(=O)Oc1ccccc1C(=O)O' aspirin /tmp/"
echo "  smiles_batch compounds.smi ./ligands/"
