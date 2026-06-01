#!/usr/bin/env python3
"""
smiles_to_pdbqt.py — SMILES → PDBQT 批次轉換器（最專業版）

工具鏈（2024-2025 黃金標準）：
  1. dimorphite-dl v1.2.5  — pKa 質子化（NH3+/COO-/His+/Arg+）
  2. RDKit ETKDGv3 + MMFF94 — 3D 構型生成
  3. meeko PDBQTWriterLegacy — AutoDock 原子類型指派 + PDBQT 輸出

為何不用舊工具：
  - obabel --pH：NH3+/COO- 原子類型錯誤（已驗證）
  - prepare_ligand4.py：Python 2.7，AutoDock 4 舊工具
  - meeko 是 Scripps Forli Lab 的官方現代替代品（Vina 官方文件推薦）

驗證結果（pH 7.4）：
  amantadine  [NH3+] → N  + 3×HD  ✓
  ibuprofen   [O-]   → 2×OA, no HD ✓
  histidine   zwitterion → N + 2×NA + 2×OA ✓

用法：
  python3 smiles_to_pdbqt.py -s "CC(=O)Oc1ccccc1C(=O)O" -n aspirin -o ./ligands/
  python3 smiles_to_pdbqt.py -i compounds.smi -o ./ligands/
  echo "CC(=O)O acetic_acid" | python3 smiles_to_pdbqt.py -i - -o ./ligands/
  python3 smiles_to_pdbqt.py -i compounds.smi -o ./ligands/ --no-protonate

安裝：
  pip install meeko dimorphite_dl==1.2.5
  conda install -c conda-forge rdkit gemmi
"""

import argparse
import os
import re
import sys
from pathlib import Path

# ── 依賴檢查 ────────────────────────────────────────────────────────────────
_MISSING = []

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    _MISSING.append("rdkit  →  conda install -c conda-forge rdkit")

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
except ImportError:
    _MISSING.append("meeko  →  pip install meeko")

_DIMORPHITE_OK = False
try:
    from io import StringIO as _StringIO
    from dimorphite_dl.mol import Protonate as _Protonate
    from dimorphite_dl.io import LoadSMIFile as _LoadSMIFile
    _DIMORPHITE_OK = True
except ImportError:
    pass


def _check_deps():
    if _MISSING:
        print("錯誤：缺少必要套件：", file=sys.stderr)
        for m in _MISSING:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)


def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]', '_', name).strip('_') or "mol"


# ── 質子化（dimorphite-dl）─────────────────────────────────────────────────

def _pick_dominant(variants: list[str]) -> str:
    """pH 7.4 優勢態：NH3+ > COO- > 第一個"""
    for v in variants:
        if any(t in v for t in ('[NH3+]', '[NH2+]', '[NH+]', '[N+](')):
            return v
    for v in variants:
        if '[O-]' in v:
            return v
    return variants[0]


def protonate_smiles(smiles: str, pH: float = 7.4) -> str:
    """dimorphite-dl 質子化，失敗則回傳原 SMILES"""
    if not _DIMORPHITE_OK:
        return smiles
    try:
        args = {
            'smiles': smiles,
            'smiles_file': _StringIO(f'{smiles} mol'),
            'min_ph': pH - 0.5, 'max_ph': pH + 0.5,
            'pka_precision': 1.0, 'max_variants': 10,
            'silent': True, 'return_as_list': True, 'label_states': False,
        }
        args['smiles_and_data'] = _LoadSMIFile(args['smiles_file'], args)
        variants = [v.split('\t')[0].strip()
                    for v in _Protonate(args) if v.strip()]
        if variants:
            return _pick_dominant(variants)
    except Exception:
        pass
    return smiles


# ── 3D 生成（RDKit ETKDGv3 + MMFF94）────────────────────────────────────────

def smiles_to_rdkit_mol(smiles: str, n_attempts: int = 3):
    """
    SMILES → RDKit Mol（含 H、3D 座標、MMFF94 最佳化）
    失敗時回傳 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    for _ in range(n_attempts):
        if AllChem.EmbedMolecule(mol, params) == 0:
            AllChem.MMFFOptimizeMolecule(mol)
            return mol
    # ETKDGv3 失敗時退回 ETKDG
    params2 = AllChem.EmbedParameters()
    if AllChem.EmbedMolecule(mol, params2) == 0:
        AllChem.MMFFOptimizeMolecule(mol)
        return mol
    return None


# ── meeko PDBQT 輸出 ─────────────────────────────────────────────────────────

_PREP = None  # 全域 MoleculePreparation 實例（避免重複初始化）

def mol_to_pdbqt_string(mol) -> str | None:
    """RDKit Mol → PDBQT 字串（meeko PDBQTWriterLegacy）"""
    global _PREP
    if _PREP is None:
        _PREP = MoleculePreparation()
    try:
        setup_list = _PREP.prepare(mol)
        pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup_list[0])
        if not is_ok:
            return None
        return pdbqt_str
    except Exception:
        return None


# ── 主轉換函式 ───────────────────────────────────────────────────────────────

def smiles_to_pdbqt(smiles: str, name: str, outdir: str,
                    pH: float = 7.4, protonate: bool = True) -> Path:
    """
    SMILES → PDBQT（完整管道：質子化 → 3D → meeko）

    Returns
    -------
    Path to output .pdbqt，失敗時回傳 None
    """
    os.makedirs(outdir, exist_ok=True)
    safe = sanitize_name(name)
    outfile = Path(outdir) / f"{safe}.pdbqt"

    # 1. 質子化
    input_smi = protonate_smiles(smiles, pH) if protonate else smiles

    # 2. 3D 生成
    mol = smiles_to_rdkit_mol(input_smi)
    if mol is None:
        print(f"  [FAIL] {name}: RDKit 3D 生成失敗", file=sys.stderr)
        return None

    # 3. meeko PDBQT
    pdbqt_str = mol_to_pdbqt_string(mol)
    if pdbqt_str is None:
        print(f"  [FAIL] {name}: meeko PDBQT 輸出失敗", file=sys.stderr)
        return None

    outfile.write_text(pdbqt_str)
    return outfile


def process_file(smi_source, outdir: str, pH: float,
                 verbose: bool, protonate: bool) -> tuple[int, int]:
    ok = fail = 0
    for lineno, line in enumerate(smi_source, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 1)
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol{lineno:05d}"
        out = smiles_to_pdbqt(smiles, name, outdir, pH, protonate)
        if out:
            ok += 1
            if verbose:
                print(f"  OK  {out.name}")
        else:
            fail += 1
    return ok, fail


def main():
    _check_deps()

    parser = argparse.ArgumentParser(
        description="SMILES → PDBQT（dimorphite-dl + RDKit ETKDGv3 + meeko）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("-s", "--smiles", help="單個 SMILES 字串")
    src.add_argument("-i", "--input", metavar="FILE",
                     help=".smi 批次檔（- 表示 stdin）；格式：SMILES  名稱")
    parser.add_argument("-n", "--name", default="mol")
    parser.add_argument("-o", "--outdir", default="./ligands")
    parser.add_argument("--pH", type=float, default=7.4)
    parser.add_argument("--no-protonate", dest="protonate",
                        action="store_false",
                        help="跳過質子化（SMILES 已含正確電荷）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # 狀態報告
    if args.protonate:
        if _DIMORPHITE_OK:
            print(f"[質子化] dimorphite-dl ✓  pH={args.pH}")
        else:
            print("[質子化] ⚠ 未安裝 dimorphite-dl，跳過質子化", file=sys.stderr)
            print("         pip install dimorphite_dl==1.2.5", file=sys.stderr)

    if args.smiles:
        out = smiles_to_pdbqt(args.smiles, args.name, args.outdir,
                              args.pH, args.protonate)
        print(f"輸出：{out}" if out else "失敗")
        sys.exit(0 if out else 1)

    if args.input == "-" or args.input is None:
        src_iter, label = sys.stdin, "stdin"
    else:
        if not os.path.isfile(args.input):
            print(f"錯誤：找不到 {args.input}", file=sys.stderr)
            sys.exit(1)
        src_iter = open(args.input)
        label = args.input

    print(f"轉換：{label}  →  {args.outdir}/  (pH={args.pH})")
    ok, fail = process_file(src_iter, args.outdir, args.pH,
                            args.verbose, args.protonate)
    if args.input and args.input != "-":
        src_iter.close()

    print(f"\n完成：{ok} 成功，{fail} 失敗  →  {args.outdir}/")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
