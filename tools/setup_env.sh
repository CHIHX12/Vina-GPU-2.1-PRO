#!/usr/bin/env bash
# setup_env.sh — AutoDock/Vina-GPU 工具環境設定
#
# 用法：
#   source /path/to/Vina-GPU-2.1/tools/setup_env.sh
#
# 或加入 ~/.bashrc：
#   echo 'source /path/to/Vina-GPU-2.1/tools/setup_env.sh' >> ~/.bashrc
#
# 說明：
#   - MGLTools (prepare_receptor4.py 等) 需要 Python 2.7；
#     已在 jp_214 conda env 中透過 $CONDA_PREFIX/bin/ 包裝器提供。
#   - SMILES 轉換使用 obabel（Python 3 原生，不需要 MGLTools）。

# ── 確認目前在正確的 conda env ─────────────────────────────────────────────
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "[setup_env] 警告：CONDA_PREFIX 未設定，請先 conda activate jp_214"
    return 1
fi
_MGL_BIN="$CONDA_PREFIX/bin"
if [[ ! -f "$_MGL_BIN/prepare_receptor4.py" ]]; then
    echo "[setup_env] 警告：找不到 MGLTools，請確認已 conda activate jp_214"
    return 1
fi

# ── MGLTools 別名（使用 conda env 內的 python2.7 包裝器）─────────────────
alias prepare_pdb_split_alt_confs.py="python $_MGL_BIN/prepare_pdb_split_alt_confs.py"
alias prepare_receptor4.py="python $_MGL_BIN/prepare_receptor4.py"
alias prepare_ligand4.py="python $_MGL_BIN/prepare_ligand4.py"
alias prepare_gpf4.py="python $_MGL_BIN/prepare_gpf4.py"
alias prepare_dpf4.py="python $_MGL_BIN/prepare_dpf4.py"
alias prepare_flexreceptor4.py="python $_MGL_BIN/prepare_flexreceptor4.py"

# ── Vina-GPU 2.1 PRO binary ───────────────────────────────────────────────
_VINA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VINA_BIN="$_VINA_DIR/src/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1"
export VINA_OCL="$_VINA_DIR"
alias vina-gpu="$VINA_BIN --opencl_binary_path $VINA_OCL"

# ── SMILES → PDBQT（單個）──────────────────────────────────────────────────
# 用法：smiles2pdbqt "CC(=O)O" aspirin
#        smiles2pdbqt "CC(=O)O" aspirin output_dir/
smiles2pdbqt() {
    local smi="$1"
    local name="${2:-mol}"
    local outdir="${3:-.}"
    local name_safe="${name//[^A-Za-z0-9_-]/_}"
    local outfile="$outdir/${name_safe}.pdbqt"
    mkdir -p "$outdir"
    echo "${smi} ${name}" | obabel -ismi -opdbqt --gen3d -h -O "$outfile" 2>/dev/null
    echo "$outfile"
}

# ── SMILES → PDBQT（批次，.smi 檔）────────────────────────────────────────
# 格式：每行 "SMILES  名稱"（名稱可選）
# 用法：smiles_batch compounds.smi output_dir/
smiles_batch() {
    local smi_file="$1"
    local outdir="${2:-./ligands}"
    if [[ -z "$smi_file" || ! -f "$smi_file" ]]; then
        echo "用法：smiles_batch <input.smi> [output_dir]"
        echo "  input.smi 格式：每行 'SMILES  名稱'"
        return 1
    fi
    mkdir -p "$outdir"
    python3 "$(dirname "${BASH_SOURCE[0]}")/smiles_to_pdbqt.py" \
        --input "$smi_file" --outdir "$outdir"
}

# ── 快速確認 ──────────────────────────────────────────────────────────────
echo "[setup_env] 已載入："
echo "  MGLTools aliases → $CONDA_PREFIX/bin/ (python 2.7 wrapper)"
echo "  vina-gpu         → $VINA_BIN"
echo "  smiles2pdbqt     → obabel 單個 SMILES"
echo "  smiles_batch     → obabel 批次 .smi 檔"
echo ""
echo "快速測試："
echo "  smiles2pdbqt 'CC(=O)Oc1ccccc1C(=O)O' aspirin /tmp/"
echo "  smiles_batch compounds.smi ./ligands/"
