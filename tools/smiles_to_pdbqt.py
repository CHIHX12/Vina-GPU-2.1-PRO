#!/usr/bin/env python3
"""
smiles_to_pdbqt.py — SMILES → PDBQT 批次轉換器（Python 3，使用 obabel）

質子化策略（重要！）
─────────────────────────────────────────────────────────────────────────────
obabel 的 --pH 旗標有已知錯誤：
  - NH3+（pKa ~10 的一級胺）在 pH 7.4 下仍輸出中性 NH2（NA 類型）
  - COO-（pKa ~4 的羧酸）在 pH 7.4 下保留 HD 氫（改為需要 OA×2，無 HD）

已驗證的正確做法（本腳本實作）：
  1. dimorphite-dl（推薦，pip install dimorphite_dl==1.2.5）
     自動處理 NH3+/COO-/His+/Arg+ 等，選取 pH 7.4 下的優勢態
  2. 直接提供帶電荷的 SMILES（手動或從 ZINC15 下載的 pH 7.4 版本）

驗證結果（pH 7.4，gen3d best）：
  amantadine  [NH3+]SMILES → N  + 3×HD  ✓ （原 obabel --pH 給 NA + 2×HD ✗）
  ibuprofen   [O-]SMILES   → 2×OA, no HD ✓ （原 obabel --pH 給 OA + OA + HD ✗）

用法：
  # 單個 SMILES
  python3 smiles_to_pdbqt.py -s "CC(=O)Oc1ccccc1C(=O)O" -n aspirin -o ./ligands/

  # 批次 .smi 檔（每行：SMILES  名稱）
  python3 smiles_to_pdbqt.py -i compounds.smi -o ./ligands/

  # 從 stdin（管道）
  echo "CC(=O)O acetic_acid" | python3 smiles_to_pdbqt.py -i - -o ./ligands/

  # 跳過 dimorphite-dl（SMILES 已含正確電荷時使用）
  python3 smiles_to_pdbqt.py -i compounds.smi -o ./ligands/ --no-protonate

安裝依賴：
  conda install -c conda-forge openbabel    # 必須（3D 生成 + PDBQT 格式）
  pip install dimorphite_dl==1.2.5          # 強烈建議（NH3+/COO- 正確電荷）
"""

import argparse
import os
import re
import sys
import subprocess
from pathlib import Path

# ── dimorphite-dl 偵測 ──────────────────────────────────────────────────────
_DIMORPHITE_AVAILABLE = False
try:
    from io import StringIO as _StringIO
    from dimorphite_dl.mol import Protonate as _Protonate
    from dimorphite_dl.io import LoadSMIFile as _LoadSMIFile
    _DIMORPHITE_AVAILABLE = True
except ImportError:
    pass


def sanitize_name(name: str) -> str:
    """將分子名稱轉為安全的檔名"""
    return re.sub(r'[^A-Za-z0-9_-]', '_', name).strip('_') or "mol"


def _pick_dominant(variants: list[str]) -> str:
    """
    從多個質子化態中選取 pH 7.4 下的優勢態。

    dimorphite-dl 在 pKa 靠近 pH 時回傳 BOTH（兩種態）。
    在生理 pH 7.4 下：
      - 脂肪族胺（pKa ~10）的 NH3+ 態優先
      - 羧酸（pKa ~4）的 COO- 態優先
    """
    # 優先：含 NH3+/NH2+/NH+ 的態（帶正電胺）
    for v in variants:
        if any(t in v for t in ('[NH3+]', '[NH2+]', '[NH+]', '[N+](')):
            return v
    # 其次：含 [O-] 的態（羧酸根、磷酸根等去質子化）
    for v in variants:
        if '[O-]' in v:
            return v
    return variants[0]


def protonate_smiles(smiles: str, pH: float = 7.4, ph_range: float = 0.5) -> str:
    """
    使用 dimorphite-dl 取得 pH 下的優勢質子化態 SMILES。
    若未安裝 dimorphite-dl，回傳原始 SMILES。
    """
    if not _DIMORPHITE_AVAILABLE:
        return smiles

    try:
        args = {
            'smiles': smiles,
            'smiles_file': _StringIO(f'{smiles} mol'),
            'min_ph': pH - ph_range,
            'max_ph': pH + ph_range,
            'pka_precision': 1.0,
            'max_variants': 10,
            'silent': True,
            'return_as_list': True,
            'label_states': False,
        }
        args['smiles_and_data'] = _LoadSMIFile(args['smiles_file'], args)
        variants = [v.split('\t')[0].strip()
                    for v in _Protonate(args) if v.strip()]
        if variants:
            return _pick_dominant(variants)
    except Exception:
        pass
    return smiles


def smiles_to_pdbqt(smiles: str, name: str, outdir: str,
                    pH: float = 7.4, gen3d: str = "best",
                    protonate: bool = True) -> Path:
    """
    單個 SMILES → PDBQT

    Parameters
    ----------
    smiles    : SMILES 字串（可以是中性或帶電荷）
    name      : 分子名稱（用於檔名）
    outdir    : 輸出目錄
    pH        : 質子化 pH（預設 7.4）
    gen3d     : 3D 生成品質（預設 'best'；注意 'fast'/'fastest' 可能無法正確生成電荷原子的 3D）
    protonate : 是否執行 dimorphite-dl 質子化（預設 True）

    Returns
    -------
    Path to output .pdbqt，失敗時回傳 None
    """
    os.makedirs(outdir, exist_ok=True)
    safe = sanitize_name(name)
    outfile = Path(outdir) / f"{safe}.pdbqt"

    input_smiles = protonate_smiles(smiles, pH) if protonate else smiles

    cmd = [
        "obabel", "-ismi", "-opdbqt",
        "--gen3d", gen3d,
        "-O", str(outfile),
    ]

    result = subprocess.run(
        cmd,
        input=f"{input_smiles} {name}\n",
        capture_output=True, text=True
    )

    if result.returncode != 0 or not outfile.exists():
        print(f"  [FAIL] {name}: {result.stderr.strip()[:120]}", file=sys.stderr)
        return None

    return outfile


def process_file(smi_source, outdir: str, pH: float, gen3d: str,
                 verbose: bool, protonate: bool) -> tuple[int, int]:
    """處理 .smi 檔案或 stdin，逐行轉換"""
    ok = fail = 0
    for lineno, line in enumerate(smi_source, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 1)
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol{lineno:05d}"

        out = smiles_to_pdbqt(smiles, name, outdir, pH, gen3d, protonate)
        if out:
            ok += 1
            if verbose:
                print(f"  OK  {out.name}")
        else:
            fail += 1

    return ok, fail


def main():
    parser = argparse.ArgumentParser(
        description="SMILES → PDBQT 批次轉換（obabel + dimorphite-dl 質子化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("-s", "--smiles", help="單個 SMILES 字串")
    src.add_argument("-i", "--input", metavar="FILE",
                     help=".smi 批次檔案（- 表示 stdin）；格式：SMILES  名稱")

    parser.add_argument("-n", "--name", default="mol",
                        help="分子名稱（--smiles 模式用）")
    parser.add_argument("-o", "--outdir", default="./ligands",
                        help="輸出目錄（預設：./ligands）")
    parser.add_argument("--pH", type=float, default=7.4,
                        help="質子化 pH（預設：7.4）")
    parser.add_argument("--gen3d", default="best",
                        choices=["fastest", "fast", "medium", "best"],
                        help="3D 生成品質（預設：best；charged SMILES 必須用 best）")
    parser.add_argument("--no-protonate", dest="protonate", action="store_false",
                        help="跳過 dimorphite-dl 質子化（SMILES 已含正確電荷時使用）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # 確認 obabel 可用
    if subprocess.run(["which", "obabel"], capture_output=True).returncode != 0:
        print("錯誤：找不到 obabel，請安裝：conda install -c conda-forge openbabel",
              file=sys.stderr)
        sys.exit(1)

    # 質子化後端狀態
    if args.protonate:
        if _DIMORPHITE_AVAILABLE:
            print(f"[質子化] dimorphite-dl ✓  pH={args.pH}  (NH3+/COO-/His+ 正確)")
        else:
            print("[質子化] ⚠ 未找到 dimorphite-dl，使用原始 SMILES（不質子化）",
                  file=sys.stderr)
            print("         請執行：pip install dimorphite_dl==1.2.5", file=sys.stderr)

    # 警告 gen3d!=best 可能影響電荷原子的 3D 生成
    if args.gen3d != "best" and args.protonate and _DIMORPHITE_AVAILABLE:
        print(f"[警告] --gen3d {args.gen3d} 對電荷 SMILES 可能生成 0,0,0 座標，建議使用 --gen3d best",
              file=sys.stderr)

    # 單個 SMILES 模式
    if args.smiles:
        out = smiles_to_pdbqt(args.smiles, args.name, args.outdir,
                               args.pH, args.gen3d, args.protonate)
        if out:
            print(f"輸出：{out}")
            sys.exit(0)
        else:
            sys.exit(1)

    # 批次模式
    if args.input == "-" or args.input is None:
        src_iter = sys.stdin
        label = "stdin"
    else:
        if not os.path.isfile(args.input):
            print(f"錯誤：找不到檔案 {args.input}", file=sys.stderr)
            sys.exit(1)
        src_iter = open(args.input)
        label = args.input

    print(f"轉換：{label}  →  {args.outdir}/  (pH={args.pH}, gen3d={args.gen3d})")
    ok, fail = process_file(src_iter, args.outdir, args.pH, args.gen3d,
                            args.verbose, args.protonate)

    if args.input and args.input != "-":
        src_iter.close()

    print(f"\n完成：{ok} 成功，{fail} 失敗  →  {args.outdir}/")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
