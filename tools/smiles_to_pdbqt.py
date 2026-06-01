#!/usr/bin/env python3
"""
smiles_to_pdbqt.py — SMILES → PDBQT 批次轉換器（Python 3，使用 obabel）

用法：
  # 單個 SMILES
  python3 smiles_to_pdbqt.py -s "CC(=O)Oc1ccccc1C(=O)O" -n aspirin -o ./ligands/

  # 批次 .smi 檔（每行：SMILES  名稱）
  python3 smiles_to_pdbqt.py -i compounds.smi -o ./ligands/

  # 從 stdin（管道）
  echo "CC(=O)O acetic_acid" | python3 smiles_to_pdbqt.py -i - -o ./ligands/

輸出：
  ./ligands/<名稱>.pdbqt  每個分子一個檔案
"""

import argparse
import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path


def sanitize_name(name: str) -> str:
    """將分子名稱轉為安全的檔名"""
    return re.sub(r'[^A-Za-z0-9_-]', '_', name).strip('_') or "mol"


def smiles_to_pdbqt(smiles: str, name: str, outdir: str,
                    pH: float = 7.4, gen3d: str = "best") -> Path:
    """
    單個 SMILES → PDBQT

    Parameters
    ----------
    smiles  : SMILES 字串
    name    : 分子名稱（用於檔名）
    outdir  : 輸出目錄
    pH      : 質子化 pH（預設 7.4）
    gen3d   : 3D 生成品質：'fastest'/'fast'/'medium'/'best'（預設 'best'）

    Returns
    -------
    Path to output .pdbqt
    """
    os.makedirs(outdir, exist_ok=True)
    safe = sanitize_name(name)
    outfile = Path(outdir) / f"{safe}.pdbqt"

    cmd = [
        "obabel", "-ismi", "-opdbqt",
        "--gen3d", gen3d,
        "--pH", str(pH),
        "-O", str(outfile),
    ]

    result = subprocess.run(
        cmd,
        input=f"{smiles} {name}\n",
        capture_output=True, text=True
    )

    if result.returncode != 0 or not outfile.exists():
        print(f"  [FAIL] {name}: {result.stderr.strip()[:120]}", file=sys.stderr)
        return None

    return outfile


def process_file(smi_source, outdir: str, pH: float, gen3d: str,
                 verbose: bool) -> tuple[int, int]:
    """處理 .smi 檔案或 stdin，逐行轉換"""
    ok = fail = 0
    for lineno, line in enumerate(smi_source, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(None, 1)
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else f"mol{lineno:05d}"

        out = smiles_to_pdbqt(smiles, name, outdir, pH, gen3d)
        if out:
            ok += 1
            if verbose:
                print(f"  OK  {out.name}")
        else:
            fail += 1

    return ok, fail


def main():
    parser = argparse.ArgumentParser(
        description="SMILES → PDBQT 批次轉換（使用 obabel）",
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
                        help="3D 生成品質（預設：best）")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # 確認 obabel 可用
    if subprocess.run(["which", "obabel"], capture_output=True).returncode != 0:
        print("錯誤：找不到 obabel，請安裝：conda install -c conda-forge openbabel",
              file=sys.stderr)
        sys.exit(1)

    # 單個 SMILES 模式
    if args.smiles:
        out = smiles_to_pdbqt(args.smiles, args.name, args.outdir,
                               args.pH, args.gen3d)
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
    ok, fail = process_file(src_iter, args.outdir, args.pH, args.gen3d, args.verbose)

    if args.input and args.input != "-":
        src_iter.close()

    print(f"\n完成：{ok} 成功，{fail} 失敗  →  {args.outdir}/")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
