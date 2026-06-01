#!/usr/bin/env python3
"""
patch_meeko_metals.py — 為 meeko 新增稀有金屬 AD4 原子類型

支援新增：Cu, Co, Ni, Mo, W(鎢), V, Pt, Pd, Ru, Rh, Cd, Hg

說明：
  meeko 標準版只定義 Zn/Fe/Mg/Ca/Mn 五種金屬。
  受體含 Cu(銅酶)、Co(維生素B12)、Ni(尿酶)、Mo(亞硫酸氧化酶)、
  Pt/Ru(抗癌藥靶點) 等金屬時會失敗或 skip。
  本腳本直接修改 conda 環境中的 meeko 資料檔，補全這些類型。

用法：
  PYTHONNOUSERSITE=1 python3 tools/patch_meeko_metals.py
  # 或（測試不修改）：
  PYTHONNOUSERSITE=1 python3 tools/patch_meeko_metals.py --dry-run

還原：
  PYTHONNOUSERSITE=1 python3 tools/patch_meeko_metals.py --restore

注意：
  - conda update meeko 後需重新執行
  - 對 Vina 評分的影響：VDW 位阻與受體金屬 H-bond 分類
  - 有機金屬配體（cisplatin 等）還需另外處理（RDKit MMFF 不支援 Pt-N 鍵）

AD4 類型字串慣例：
  - 2 字元，首字大寫（與 meeko 現有金屬一致）
  - W(水) = 'W' 已被佔用 → 鎢使用 'Wt' / 'WT'
  - V(釩) 使用 'Va' / 'VA'（避免 SMARTS 歧義）
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ── 新增金屬定義 ──────────────────────────────────────────────────────────────
# 欄位說明：
#   atomic_num : 原子序
#   atype      : AD4 類型字串（2字元，首字大寫）
#   rmin_half  : LJ 極小值距離的一半 (Å)，參考離子半徑（6配位）
#   epsilon    : LJ 深度 (kcal/mol)
#   sol_vol    : 溶劑化體積 (Å³)，= 4/3 π r³（r = 離子半徑）
#   note       : 參考來源
#
# VDW 參數來源：
#   - 3d 金屬 (Cu/Co/Ni)：參考 Shannon (1976) + AD4 pattern (Zn=0.74,ε=0.55)
#   - 4d/5d 金屬 (Mo/Ru/Rh/Pd/W/Pt)：離子半徑；ε=0.01（較軟，遠離 native）
#   - Cd/Hg：較大離子半徑；ε=0.55（類似 Ca/Zn）
#   - V：離子半徑 0.64Å；ε=0.875（類似 Mg/Mn）

# ── PDB 殘基名稱 → SMILES（用於 residue_chem_templates.json）────────────────
# PDB 慣例：單元素金屬原子的殘基名稱 = 元素符號大寫（1-2 字元）
# 氧化態選用最常見形式（與 FE=[Fe+3], MN=[Mn+2] 保持一致）
NEW_METAL_RESIDUES = [
    # pdb_resname, smiles_str,   pdb_atom_name
    ('CU',  '[Cu+2]',  'CU'),   # copper(II)
    ('CO',  '[Co+2]',  'CO'),   # cobalt(II)
    ('NI',  '[Ni+2]',  'NI'),   # nickel(II)
    ('MO',  '[Mo+4]',  'MO'),   # molybdenum(IV)
    ('W',   '[W+4]',   'W'),    # tungsten(IV) – 注意: 'W'水氧是 AutoDock 標記，PDB 鎢殘基名是 'W'
    ('V',   '[V+3]',   'V'),    # vanadium(III)
    ('PT',  '[Pt+2]',  'PT'),   # platinum(II)
    ('PD',  '[Pd+2]',  'PD'),   # palladium(II)
    ('RU',  '[Ru+3]',  'RU'),   # ruthenium(III)
    ('RH',  '[Rh+3]',  'RH'),   # rhodium(III)
    ('CD',  '[Cd+2]',  'CD'),   # cadmium(II)
    ('HG',  '[Hg+2]',  'HG'),   # mercury(II)
]

NEW_METALS = [
    # ─ 3d 過渡金屬 ─────────────────────────────────────────────────────────
    dict(atomic_num=23,  atype='Va', atype_up='VA', element='V',
         rmin_half=0.64, epsilon=0.875, sol_vol=1.10,
         note='Vanadium(III) r=0.64Å (Shannon); common in haloperoxidases'),
    dict(atomic_num=27,  atype='Co', atype_up='CO', element='Co',
         rmin_half=0.74, epsilon=0.55, sol_vol=1.70,
         note='Cobalt(II) r=0.74Å; Vit-B12, nitrile hydratase'),
    dict(atomic_num=28,  atype='Ni', atype_up='NI', element='Ni',
         rmin_half=0.69, epsilon=0.55, sol_vol=1.38,
         note='Nickel(II) r=0.69Å; urease, Ni-SOD'),
    dict(atomic_num=29,  atype='Cu', atype_up='CU', element='Cu',
         rmin_half=0.73, epsilon=0.55, sol_vol=1.63,
         note='Copper(II) r=0.73Å; azurin, ceruloplasmin'),
    # ─ 4d 過渡金屬 ─────────────────────────────────────────────────────────
    dict(atomic_num=42,  atype='Mo', atype_up='MO', element='Mo',
         rmin_half=0.65, epsilon=0.01, sol_vol=1.15,
         note='Molybdenum(IV) r=0.65Å; sulfite oxidase, xanthine ox.'),
    dict(atomic_num=44,  atype='Ru', atype_up='RU', element='Ru',
         rmin_half=0.68, epsilon=0.01, sol_vol=1.32,
         note='Ruthenium(III) r=0.68Å; anticancer metallodrugs'),
    dict(atomic_num=45,  atype='Rh', atype_up='RH', element='Rh',
         rmin_half=0.67, epsilon=0.01, sol_vol=1.26,
         note='Rhodium(III) r=0.67Å; anticancer, catalysis'),
    dict(atomic_num=46,  atype='Pd', atype_up='PD', element='Pd',
         rmin_half=0.86, epsilon=0.01, sol_vol=2.67,
         note='Palladium(II) r=0.86Å; anticancer'),
    dict(atomic_num=48,  atype='Cd', atype_up='CD', element='Cd',
         rmin_half=0.95, epsilon=0.55, sol_vol=3.59,
         note='Cadmium(II) r=0.95Å; metallothionein, Cd-SOD'),
    # ─ 5d 過渡金屬 ─────────────────────────────────────────────────────────
    dict(atomic_num=74,  atype='Wt', atype_up='WT', element='W',
         rmin_half=0.66, epsilon=0.01, sol_vol=1.21,
         note='Tungsten(IV) r=0.66Å; tungstoenzymes; NOTE: W=water in AD4'),
    dict(atomic_num=78,  atype='Pt', atype_up='PT', element='Pt',
         rmin_half=0.80, epsilon=0.01, sol_vol=2.14,
         note='Platinum(II) r=0.80Å; cisplatin binding sites'),
    dict(atomic_num=80,  atype='Hg', atype_up='HG', element='Hg',
         rmin_half=1.02, epsilon=0.55, sol_vol=4.45,
         note='Mercury(II) r=1.02Å; metallothionein, heavy metal binding'),
]

# ── 擴充金屬：週期表完整覆蓋 ─────────────────────────────────────────────────
# 衝突大寫類型（meeko 已用於其他 AD4 類型，加為大寫 metal 別名會覆蓋其語意）：
#   NA = amide-N acceptor  OS = ester-O acceptor
_FORBIDDEN_UPPER_ATYPES = frozenset({'NA', 'OS'})

def _v(r: float) -> float:
    """Shannon 離子半徑 r (Å) → 溶劑化體積 (4/3)πr³ (Å³)"""
    import math
    return round((4 / 3) * math.pi * r ** 3, 2)

EXTRA_METALS = [
    # ─ 鹼金屬 Group 1 ─────────────────────────────────────────────────────────
    dict(atomic_num=3,  atype='Li', atype_up='LI', element='Li', pdb_res='LI',
         pdb_smiles='[Li+]',  rmin_half=0.76, epsilon=0.018, sol_vol=_v(0.76),
         note='Lithium(I)   r=0.76Å; lithium drugs'),
    dict(atomic_num=11, atype='Na', atype_up=None,  element='Na', pdb_res='NA',
         pdb_smiles='[Na+]',  rmin_half=1.02, epsilon=0.02,  sol_vol=_v(1.02),
         note='Sodium(I)    r=1.02Å; NOTE: NA uppercase conflicts with N-acceptor'),
    dict(atomic_num=19, atype='Ka', atype_up='KA', element='K',  pdb_res='K',
         pdb_smiles='[K+]',   rmin_half=1.38, epsilon=0.00,  sol_vol=_v(1.38),
         note='Potassium(I) r=1.38Å; K channels'),
    dict(atomic_num=37, atype='Rb', atype_up='RB', element='Rb', pdb_res='RB',
         pdb_smiles='[Rb+]',  rmin_half=1.52, epsilon=0.00,  sol_vol=_v(1.52),
         note='Rubidium(I)  r=1.52Å; K-analog in channel studies'),
    dict(atomic_num=55, atype='Cs', atype_up='CS', element='Cs', pdb_res='CS',
         pdb_smiles='[Cs+]',  rmin_half=1.67, epsilon=0.00,  sol_vol=_v(1.67),
         note='Caesium(I)   r=1.67Å; K-channel blocker'),
    # ─ 鹼土金屬 Group 2 ───────────────────────────────────────────────────────
    dict(atomic_num=4,  atype='Be', atype_up='BE', element='Be', pdb_res='BE',
         pdb_smiles='[Be+2]', rmin_half=0.45, epsilon=0.05,  sol_vol=_v(0.45),
         note='Beryllium(II) r=0.45Å; beryllium toxicity'),
    dict(atomic_num=38, atype='Sr', atype_up='SR', element='Sr', pdb_res='SR',
         pdb_smiles='[Sr+2]', rmin_half=1.18, epsilon=0.10,  sol_vol=_v(1.18),
         note='Strontium(II) r=1.18Å; Ca-substitute in crystallography'),
    dict(atomic_num=56, atype='Ba', atype_up='BA', element='Ba', pdb_res='BA',
         pdb_smiles='[Ba+2]', rmin_half=1.35, epsilon=0.10,  sol_vol=_v(1.35),
         note='Barium(II)   r=1.35Å; K-channel blocker'),
    # ─ 後過渡 Period 3 ──────────────────────────────────────────────────────
    dict(atomic_num=13, atype='Al', atype_up='AL', element='Al', pdb_res='AL',
         pdb_smiles='[Al+3]', rmin_half=0.54, epsilon=0.05,  sol_vol=_v(0.54),
         note='Aluminium(III) r=0.54Å; aluminofluoride transition state mimic'),
    # ─ 過渡金屬 Period 4（剩餘）──────────────────────────────────────────────
    dict(atomic_num=21, atype='Sc', atype_up='SC', element='Sc', pdb_res='SC',
         pdb_smiles='[Sc+3]', rmin_half=0.74, epsilon=0.05,  sol_vol=_v(0.74),
         note='Scandium(III) r=0.74Å; 47Sc diagnostic'),
    dict(atomic_num=22, atype='Ti', atype_up='TI', element='Ti', pdb_res='TI',
         pdb_smiles='[Ti+4]', rmin_half=0.61, epsilon=0.05,  sol_vol=_v(0.61),
         note='Titanium(IV)  r=0.61Å; titanocene anticancer'),
    dict(atomic_num=24, atype='Cr', atype_up='CR', element='Cr', pdb_res='CR',
         pdb_smiles='[Cr+3]', rmin_half=0.62, epsilon=0.05,  sol_vol=_v(0.62),
         note='Chromium(III) r=0.62Å; glucose tolerance factor'),
    # ─ 後過渡 Period 4 ──────────────────────────────────────────────────────
    dict(atomic_num=31, atype='Ga', atype_up='GA', element='Ga', pdb_res='GA',
         pdb_smiles='[Ga+3]', rmin_half=0.62, epsilon=0.05,  sol_vol=_v(0.62),
         note='Gallium(III)  r=0.62Å; 68Ga PET, anticancer Fe-mimic'),
    dict(atomic_num=32, atype='Ge', atype_up='GE', element='Ge', pdb_res='GE',
         pdb_smiles='[Ge+4]', rmin_half=0.53, epsilon=0.05,  sol_vol=_v(0.53),
         note='Germanium(IV) r=0.53Å; organogermanium'),
    dict(atomic_num=33, atype='As', atype_up='AS', element='As', pdb_res='AS',
         pdb_smiles='[As+3]', rmin_half=0.58, epsilon=0.05,  sol_vol=_v(0.58),
         note='Arsenic(III)  r=0.58Å; As2O3 anticancer, antiparasitic'),
    dict(atomic_num=34, atype='Se', atype_up='SE', element='Se', pdb_res='SE',
         pdb_smiles='[Se]',   rmin_half=0.50, epsilon=0.16,  sol_vol=_v(0.50),
         note='Selenium      r=0.50Å; selenocysteine, 75Se tracer'),
    # ─ 過渡金屬 Period 5（剩餘）──────────────────────────────────────────────
    dict(atomic_num=39, atype='Yt', atype_up='YT', element='Y',  pdb_res='Y',
         pdb_smiles='[Y+3]',  rmin_half=0.90, epsilon=0.05,  sol_vol=_v(0.90),
         note='Yttrium(III)  r=0.90Å; 90Y/86Y radiolabeled therapy'),
    dict(atomic_num=40, atype='Zr', atype_up='ZR', element='Zr', pdb_res='ZR',
         pdb_smiles='[Zr+4]', rmin_half=0.72, epsilon=0.05,  sol_vol=_v(0.72),
         note='Zirconium(IV) r=0.72Å; 89Zr PET imaging'),
    dict(atomic_num=41, atype='Nb', atype_up='NB', element='Nb', pdb_res='NB',
         pdb_smiles='[Nb+5]', rmin_half=0.64, epsilon=0.01,  sol_vol=_v(0.64),
         note='Niobium(V)    r=0.64Å; niobocene anticancer'),
    dict(atomic_num=43, atype='Tc', atype_up='TC', element='Tc', pdb_res='TC',
         pdb_smiles='[Tc+3]', rmin_half=0.65, epsilon=0.01,  sol_vol=_v(0.65),
         note='Technetium    r=0.65Å; 99mTc radiopharmaceuticals'),
    dict(atomic_num=47, atype='Ag', atype_up='AG', element='Ag', pdb_res='AG',
         pdb_smiles='[Ag+]',  rmin_half=1.15, epsilon=0.01,  sol_vol=_v(1.15),
         note='Silver(I)     r=1.15Å; antimicrobial, 111Ag'),
    # ─ 後過渡 Period 5 ──────────────────────────────────────────────────────
    dict(atomic_num=49, atype='In', atype_up='IN', element='In', pdb_res='IN',
         pdb_smiles='[In+3]', rmin_half=0.80, epsilon=0.05,  sol_vol=_v(0.80),
         note='Indium(III)   r=0.80Å; 111In/113In radiopharmaceuticals'),
    dict(atomic_num=50, atype='Sn', atype_up='SN', element='Sn', pdb_res='SN',
         pdb_smiles='[Sn+2]', rmin_half=0.69, epsilon=0.05,  sol_vol=_v(0.69),
         note='Tin(II)       r=0.69Å; organotin, 117mSn'),
    dict(atomic_num=51, atype='Sb', atype_up='SB', element='Sb', pdb_res='SB',
         pdb_smiles='[Sb+3]', rmin_half=0.76, epsilon=0.05,  sol_vol=_v(0.76),
         note='Antimony(III) r=0.76Å; antiparasitic drugs'),
    dict(atomic_num=52, atype='Te', atype_up='TE', element='Te', pdb_res='TE',
         pdb_smiles='[Te+2]', rmin_half=0.97, epsilon=0.05,  sol_vol=_v(0.97),
         note='Tellurium(II) r=0.97Å; organotellurium'),
    # ─ 過渡金屬 Period 6（剩餘）──────────────────────────────────────────────
    dict(atomic_num=72, atype='Hf', atype_up='HF', element='Hf', pdb_res='HF',
         pdb_smiles='[Hf+4]', rmin_half=0.71, epsilon=0.01,  sol_vol=_v(0.71),
         note='Hafnium(IV)   r=0.71Å; NBTXR3 radioenhancer'),
    dict(atomic_num=73, atype='Ta', atype_up='TA', element='Ta', pdb_res='TA',
         pdb_smiles='[Ta+5]', rmin_half=0.68, epsilon=0.01,  sol_vol=_v(0.68),
         note='Tantalum(V)   r=0.68Å'),
    dict(atomic_num=75, atype='Re', atype_up='RE', element='Re', pdb_res='RE',
         pdb_smiles='[Re+3]', rmin_half=0.63, epsilon=0.01,  sol_vol=_v(0.63),
         note='Rhenium(III)  r=0.63Å; 186Re/188Re radiopharmaceuticals'),
    dict(atomic_num=76, atype='Os', atype_up=None,  element='Os', pdb_res='OS',
         pdb_smiles='[Os+2]', rmin_half=0.63, epsilon=0.01,  sol_vol=_v(0.63),
         note='Osmium(II)    r=0.63Å; NOTE: OS uppercase conflicts with O-acceptor'),
    dict(atomic_num=77, atype='Ir', atype_up='IR', element='Ir', pdb_res='IR',
         pdb_smiles='[Ir+3]', rmin_half=0.68, epsilon=0.01,  sol_vol=_v(0.68),
         note='Iridium(III)  r=0.68Å; anticancer metallodrugs'),
    dict(atomic_num=79, atype='Au', atype_up='AU', element='Au', pdb_res='AU',
         pdb_smiles='[Au+]',  rmin_half=1.37, epsilon=0.01,  sol_vol=_v(1.37),
         note='Gold(I)       r=1.37Å; auranofin, anticancer'),
    # ─ 後過渡 Period 6 ──────────────────────────────────────────────────────
    dict(atomic_num=81, atype='Tl', atype_up='TL', element='Tl', pdb_res='TL',
         pdb_smiles='[Tl+]',  rmin_half=1.50, epsilon=0.05,  sol_vol=_v(1.50),
         note='Thallium(I)   r=1.50Å; K-channel tracer'),
    dict(atomic_num=82, atype='Pb', atype_up='PB', element='Pb', pdb_res='PB',
         pdb_smiles='[Pb+2]', rmin_half=1.19, epsilon=0.05,  sol_vol=_v(1.19),
         note='Lead(II)      r=1.19Å; heavy metal binding'),
    dict(atomic_num=83, atype='Bi', atype_up='BI', element='Bi', pdb_res='BI',
         pdb_smiles='[Bi+3]', rmin_half=1.03, epsilon=0.05,  sol_vol=_v(1.03),
         note='Bismuth(III)  r=1.03Å; antiulcer drugs, 213Bi'),
    # ─ 鑭系 Lanthanides ───────────────────────────────────────────────────────
    dict(atomic_num=57, atype='La', atype_up='LA', element='La', pdb_res='LA',
         pdb_smiles='[La+3]', rmin_half=1.03, epsilon=0.05,  sol_vol=_v(1.03),
         note='Lanthanum(III)      r=1.03Å'),
    dict(atomic_num=58, atype='Ce', atype_up='CE', element='Ce', pdb_res='CE',
         pdb_smiles='[Ce+3]', rmin_half=1.01, epsilon=0.05,  sol_vol=_v(1.01),
         note='Cerium(III)         r=1.01Å'),
    dict(atomic_num=59, atype='Pr', atype_up='PR', element='Pr', pdb_res='PR',
         pdb_smiles='[Pr+3]', rmin_half=0.99, epsilon=0.05,  sol_vol=_v(0.99),
         note='Praseodymium(III)   r=0.99Å'),
    dict(atomic_num=60, atype='Nd', atype_up='ND', element='Nd', pdb_res='ND',
         pdb_smiles='[Nd+3]', rmin_half=0.98, epsilon=0.05,  sol_vol=_v(0.98),
         note='Neodymium(III)      r=0.98Å'),
    dict(atomic_num=62, atype='Sm', atype_up='SM', element='Sm', pdb_res='SM',
         pdb_smiles='[Sm+3]', rmin_half=0.96, epsilon=0.05,  sol_vol=_v(0.96),
         note='Samarium(III)       r=0.96Å; 153Sm Lexidronam radiotherapy'),
    dict(atomic_num=63, atype='Eu', atype_up='EU', element='Eu', pdb_res='EU',
         pdb_smiles='[Eu+3]', rmin_half=0.95, epsilon=0.05,  sol_vol=_v(0.95),
         note='Europium(III)       r=0.95Å; luminescent probe'),
    dict(atomic_num=64, atype='Gd', atype_up='GD', element='Gd', pdb_res='GD',
         pdb_smiles='[Gd+3]', rmin_half=0.94, epsilon=0.05,  sol_vol=_v(0.94),
         note='Gadolinium(III)     r=0.94Å; MRI contrast agent'),
    dict(atomic_num=65, atype='Tb', atype_up='TB', element='Tb', pdb_res='TB',
         pdb_smiles='[Tb+3]', rmin_half=0.92, epsilon=0.05,  sol_vol=_v(0.92),
         note='Terbium(III)        r=0.92Å; luminescent probe'),
    dict(atomic_num=66, atype='Dy', atype_up='DY', element='Dy', pdb_res='DY',
         pdb_smiles='[Dy+3]', rmin_half=0.91, epsilon=0.05,  sol_vol=_v(0.91),
         note='Dysprosium(III)     r=0.91Å; MRI probe'),
    dict(atomic_num=67, atype='Ho', atype_up='HO', element='Ho', pdb_res='HO',
         pdb_smiles='[Ho+3]', rmin_half=0.90, epsilon=0.05,  sol_vol=_v(0.90),
         note='Holmium(III)        r=0.90Å; 166Ho radiotherapy'),
    dict(atomic_num=68, atype='Er', atype_up='ER', element='Er', pdb_res='ER',
         pdb_smiles='[Er+3]', rmin_half=0.89, epsilon=0.05,  sol_vol=_v(0.89),
         note='Erbium(III)         r=0.89Å'),
    dict(atomic_num=69, atype='Tm', atype_up='TM', element='Tm', pdb_res='TM',
         pdb_smiles='[Tm+3]', rmin_half=0.88, epsilon=0.05,  sol_vol=_v(0.88),
         note='Thulium(III)        r=0.88Å; 170Tm radiotherapy'),
    dict(atomic_num=70, atype='Yb', atype_up='YB', element='Yb', pdb_res='YB',
         pdb_smiles='[Yb+3]', rmin_half=0.87, epsilon=0.05,  sol_vol=_v(0.87),
         note='Ytterbium(III)      r=0.87Å; crystallography phasing'),
    dict(atomic_num=71, atype='Lu', atype_up='LU', element='Lu', pdb_res='LU',
         pdb_smiles='[Lu+3]', rmin_half=0.86, epsilon=0.05,  sol_vol=_v(0.86),
         note='Lutetium(III)       r=0.86Å; 177Lu DOTATATE radiotherapy'),
    # ─ 錒系 Actinides（實用範圍）─────────────────────────────────────────────
    dict(atomic_num=89, atype='Ac', atype_up='AC', element='Ac', pdb_res='AC',
         pdb_smiles='[Ac+3]', rmin_half=1.12, epsilon=0.05,  sol_vol=_v(1.12),
         note='Actinium(III)  r=1.12Å; 225Ac targeted alpha therapy'),
    dict(atomic_num=90, atype='Th', atype_up='TH', element='Th', pdb_res='TH',
         pdb_smiles='[Th+4]', rmin_half=0.94, epsilon=0.05,  sol_vol=_v(0.94),
         note='Thorium(IV)    r=0.94Å; 230Th targeted therapy'),
]

# 從 EXTRA_METALS 衍生 residue 模板（pdb_res → SMILES）
EXTRA_METAL_RESIDUES = [
    (m['pdb_res'], m['pdb_smiles'], m['pdb_res'])
    for m in EXTRA_METALS
]


def _all_metals():
    """所有金屬：原始 12 種 + 擴充 EXTRA_METALS"""
    return NEW_METALS + EXTRA_METALS


def _all_metal_residues():
    """所有 residue 模板：原始 + 擴充"""
    return NEW_METAL_RESIDUES + EXTRA_METAL_RESIDUES


def find_meeko_root() -> Path:
    """conda 環境中 meeko 的安裝根目錄"""
    try:
        import meeko as _m
        return Path(_m.__file__).parent
    except ImportError:
        print("錯誤：無法 import meeko。請 conda activate vina_docking", file=sys.stderr)
        sys.exit(1)


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + '.metal_bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  備份：{bak.name}")
    return bak


def restore_backup(meeko_root: Path) -> None:
    """還原所有 .metal_bak 備份"""
    count = 0
    for bak in meeko_root.rglob('*.metal_bak'):
        orig = bak.with_suffix('')  # remove .metal_bak → original suffix
        shutil.copy2(bak, orig)
        bak.unlink()
        print(f"  還原：{orig.relative_to(meeko_root)}")
        count += 1
    print(f"還原完成：{count} 個檔案")


def _atype_variants(m: dict) -> list[str]:
    """
    回傳此金屬的 AD4 類型字串列表（混合大小寫 + 大寫變體）。
    - 若 atype_up 為 None → 不加大寫（衝突保護）
    - 若 atype_up 在 _FORBIDDEN_UPPER_ATYPES → 不加大寫
    """
    variants = [m['atype']]
    up = m.get('atype_up')
    if up and up not in _FORBIDDEN_UPPER_ATYPES and up != m['atype']:
        variants.append(up)
    return variants


def patch_ad4_types_json(path: Path, dry_run: bool) -> int:
    """補充 ad4_types.json（所有金屬）"""
    data = json.loads(path.read_text())
    existing = {e['smarts'] for e in data['ad4_types']}
    added = 0
    for m in _all_metals():
        smarts = f"[#{m['atomic_num']}]"
        if smarts not in existing:
            data['ad4_types'].append({'smarts': smarts, 'atype': m['atype'],
                                      'comment': m['note']})
            added += 1
    if added and not dry_run:
        backup(path)
        path.write_text(json.dumps(data, indent=4))
    return added


def patch_ad4_vdw_json(path: Path, dry_run: bool) -> int:
    """補充 ad4_vdw.json（所有金屬）"""
    data = json.loads(path.read_text())
    existing = {e['smarts'] for e in data['ad4_vdw']}
    added = 0
    for m in _all_metals():
        smarts = f"[#{m['atomic_num']}]"
        if smarts not in existing:
            data['ad4_vdw'].append({'smarts': smarts,
                                    'rmin_half': m['rmin_half'],
                                    'epsilon': m['epsilon']})
            added += 1
    if added and not dry_run:
        backup(path)
        path.write_text(json.dumps(data, indent=4))
    return added


def patch_ad4_desolv_volume_json(path: Path, dry_run: bool) -> int:
    """補充 ad4_desolv_volume.json（所有金屬）"""
    data = json.loads(path.read_text())
    existing = {e['smarts'] for e in data['ad4_desolv_volume']}
    added = 0
    for m in _all_metals():
        smarts = f"[#{m['atomic_num']}]"
        if smarts not in existing:
            data['ad4_desolv_volume'].append({'smarts': smarts,
                                              'ad4_sol_vol': round(m['sol_vol'], 4)})
            added += 1
    if added and not dry_run:
        backup(path)
        path.write_text(json.dumps(data, indent=4))
    return added


def patch_autodock4_atom_types_elements_py(path: Path, dry_run: bool) -> int:
    """補充 autodock4_atom_types_elements.py（所有金屬，跳過衝突大寫）"""
    text = path.read_text()
    new_lines = []
    added = 0
    for m in _all_metals():
        for at in _atype_variants(m):
            if f"'{at}'" not in text:
                new_lines.append(f"    '{at}':  '{m['element']}',  # {m['note']}")
                added += 1
    if not new_lines:
        return 0
    insert_block = '\n'.join(new_lines)
    # 插入到 'W': 'O' 行之後（water AD4 legacy entry）
    new_text = text.replace(
        "    'W':   'O'",
        "    'W':   'O',  # water oxygen (AD4 legacy)\n" + insert_block
    )
    if new_text == text:
        # fallback: insert before closing }
        new_text = text.rstrip()
        if new_text.endswith('}'):
            new_text = new_text[:-1] + '\n' + insert_block + '\n}'
    if not dry_run:
        backup(path)
        path.write_text(new_text)
    return added


def patch_receptor_pdbqt_py(path: Path, dry_run: bool) -> int:
    """補充 receptor_pdbqt.py atom_property_definitions（所有金屬，跳過衝突大寫）"""
    text = path.read_text()
    target = "'MN': 'metal'"
    if target not in text:
        print(f"  警告：在 {path.name} 找不到 '{target}'，跳過")
        return 0
    new_entries = []
    for m in _all_metals():
        for at in _atype_variants(m):
            entry = f"'{at}': 'metal'"
            if entry not in text:
                new_entries.append(entry)
    if not new_entries:
        return 0
    added_str = ', '.join(new_entries)
    new_text = text.replace(
        "'MN': 'metal'",
        f"'MN': 'metal', {added_str}"
    )
    if not dry_run:
        backup(path)
        path.write_text(new_text)
    return len(new_entries)


def patch_molecule_pdbqt_py(path: Path, dry_run: bool) -> int:
    """補充 molecule_pdbqt.py atom_property_definitions（邏輯同 receptor_pdbqt.py）"""
    return patch_receptor_pdbqt_py(path, dry_run)


def patch_residue_chem_templates_json(path: Path, dry_run: bool) -> int:
    """
    補充 residue_chem_templates.json 的殘基模板（所有金屬）

    meeko 用 PDB 殘基名查此表，找到對應 SMILES，
    再以 SMARTS [#Z] 指派 AD4 原子類型。
    範例：'ZN' → {"smiles":"[Zn+2]","atom_name":["ZN"],"link_labels":{}}
    """
    data = json.loads(path.read_text())
    templates = data.get('residue_templates', {})
    added = 0
    for resname, smiles, atom_name in _all_metal_residues():
        if resname not in templates:
            templates[resname] = {
                'smiles': smiles,
                'atom_name': [atom_name],
                'link_labels': {}
            }
            added += 1
    if added and not dry_run:
        backup(path)
        data['residue_templates'] = templates
        path.write_text(json.dumps(data, indent=2))
    return added


def patch_rdkitutils_py(path: Path, dry_run: bool) -> int:
    """
    補充 utils/rdkitutils.py covalent_radius dict（所有金屬）

    meeko 的 find_inter_mols_bonds() 查此表；金屬設為 0.00（不成鍵）。
    若原子序不在表中 → RuntimeError。

    注意：檔案中其他 dict 也有 '    3:' 等行（isotope table），
    必須只檢查 covalent_radius 區塊內是否已有該 entry。
    """
    import re
    text = path.read_text()

    # 找 covalent_radius = { ... } 區塊（DOTALL）
    m_block = re.search(r'(covalent_radius = \{)(.*?)(\n\})', text, re.DOTALL)
    if not m_block:
        print(f"  警告：在 {path.name} 找不到 covalent_radius dict，跳過")
        return 0

    cr_block = m_block.group(2)  # dict 內容（不含頭尾括號）
    new_entries = []
    for m in _all_metals():
        an = m['atomic_num']
        if f'    {an}:' not in cr_block:     # 只在 dict 區塊內檢查
            new_entries.append(
                f'    {an}: 0.00,  # {m["element"]} – no covalent bond (metal ion)'
            )
    if not new_entries:
        return 0

    insert_block = '\n'.join(new_entries)
    # 在 dict 最後一個 entry 之後、結尾 '\n}' 之前插入
    new_text = (text[:m_block.start(3)]
                + '\n' + insert_block
                + '\n}'
                + text[m_block.end(3):])
    if not dry_run:
        backup(path)
        path.write_text(new_text)
    return len(new_entries)


def patch_ambiguous_conflicts(path: Path, dry_run: bool) -> int:
    """
    從 residue_chem_templates.json ambiguous 區段移除與金屬殘基名衝突的項目。

    已知衝突：RU → uridine ['U','U3','U5p','U5']
    掃描全部 _all_metal_residues() 殘基名，移除同名的 ambiguous 項目。
    """
    data = json.loads(path.read_text())
    ambiguous = data.get('ambiguous', {})
    metal_resnames = {r[0] for r in _all_metal_residues()}

    removed = []
    for resname in list(ambiguous.keys()):
        if resname in metal_resnames:
            removed.append(f"{resname}→{ambiguous[resname]}")
            if not dry_run:
                del ambiguous[resname]

    if removed:
        print(f"    移除 ambiguous 衝突：{removed}")
        if not dry_run:
            backup(path)
            data['ambiguous'] = ambiguous
            path.write_text(json.dumps(data, indent=2))

    return len(removed)


def patch_utils_py(path: Path, dry_run: bool) -> int:
    """
    補充 utils/utils.py mini_periodic_table dict（Ac/Th 等超重元素）

    meeko 的 getPdbInfoNoNull() 查此表以取得元素符號；
    缺少 Z=89(Ac)、Z=90(Th) 等原子序 → KeyError。
    """
    text = path.read_text()

    # 找 mini_periodic_table = { ... } 區塊（DOTALL）
    m_block = re.search(r'(mini_periodic_table = \{)(.*?)(\n\})', text, re.DOTALL)
    if not m_block:
        print(f"  警告：在 {path.name} 找不到 mini_periodic_table dict，跳過")
        return 0

    pt_block = m_block.group(2)  # dict 內容（不含頭尾括號）
    new_entries = []
    for m in _all_metals():
        an = m['atomic_num']
        if f'    {an}:' not in pt_block:
            new_entries.append(
                (an, f'    {an}: "{m["element"]}",  # {m["element"]} patched by patch_meeko_metals.py')
            )
    if not new_entries:
        return 0

    # 按 atomic_num 排序後插入
    new_entries.sort(key=lambda t: t[0])
    insert_block = '\n'.join(line for _, line in new_entries)
    new_text = (text[:m_block.start(3)]
                + '\n' + insert_block
                + '\n}'
                + text[m_block.end(3):])
    if not dry_run:
        backup(path)
        path.write_text(new_text)
    return len(new_entries)


def main():
    parser = argparse.ArgumentParser(
        description='為 meeko 新增金屬 AD4 原子類型（週期表完整覆蓋：62 種金屬）'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='只顯示將修改什麼，不實際寫入')
    parser.add_argument('--restore', action='store_true',
                        help='還原所有備份')
    args = parser.parse_args()

    meeko_root = find_meeko_root()
    print(f"meeko 路徑：{meeko_root}")

    if args.restore:
        restore_backup(meeko_root)
        return

    if args.dry_run:
        print("── DRY RUN（不修改） ──────────────────")

    all_m = _all_metals()
    print(f"\n覆蓋 {len(all_m)} 種金屬（基礎 {len(NEW_METALS)} + 擴充 {len(EXTRA_METALS)}）：")
    print(' ', ', '.join(m['element'] for m in all_m))
    print()

    files_and_patchers = [
        # ─ 殘基模板（最重要：讓 mk_prepare_receptor.py 認識新金屬）─────────
        (meeko_root / 'data/residue_chem_templates.json',   patch_residue_chem_templates_json),
        # ─ 移除 ambiguous 衝突（如 RU=uridine 會優先於 RU=ruthenium）────────
        (meeko_root / 'data/residue_chem_templates.json',   patch_ambiguous_conflicts),
        # ─ AD4 原子類型參數（評分函數使用）────────────────────────────────
        (meeko_root / 'data/params/ad4_types.json',         patch_ad4_types_json),
        (meeko_root / 'data/params/ad4_vdw.json',           patch_ad4_vdw_json),
        (meeko_root / 'data/params/ad4_desolv_volume.json', patch_ad4_desolv_volume_json),
        # ─ Python 模組（類型映射與分類）────────────────────────────────────
        (meeko_root / 'utils/autodock4_atom_types_elements.py', patch_autodock4_atom_types_elements_py),
        (meeko_root / 'utils/rdkitutils.py',                    patch_rdkitutils_py),
        (meeko_root / 'utils/utils.py',                         patch_utils_py),
        (meeko_root / 'receptor_pdbqt.py',                  patch_receptor_pdbqt_py),
        (meeko_root / 'molecule_pdbqt.py',                  patch_molecule_pdbqt_py),
    ]

    total_added = 0
    for path, patcher in files_and_patchers:
        if not path.exists():
            print(f"  [SKIP] 找不到：{path}")
            continue
        n = patcher(path, args.dry_run)
        status = 'DRY' if args.dry_run else ('✓' if n > 0 else '=')
        print(f"  [{status}] {path.name}: {n} 項新增")
        total_added += n

    if not args.dry_run:
        # 清除 meeko 的 .pyc 快取，確保修改立即生效
        import importlib
        import meeko
        importlib.invalidate_caches()

    print(f"\n{'DRY RUN ' if args.dry_run else ''}完成：共 {total_added} 項修改")

    if not args.dry_run:
        print("\n驗證中...")
        _verify(meeko_root)


def _verify(meeko_root: Path):
    """驗證 patch 已生效（直接讀取 JSON/py 檔，不依賴模組重載）"""
    all_ok = True
    am = _all_metals()
    amr = _all_metal_residues()

    # 1. autodock4_atom_types_elements.py
    ate_path = meeko_root / 'utils/autodock4_atom_types_elements.py'
    ate_text = ate_path.read_text()
    missing_ate = [m['atype'] for m in am if f"'{m['atype']}'" not in ate_text]
    if missing_ate:
        print(f"  ✗ autodock4_atom_types_elements.py 缺少：{missing_ate}")
        all_ok = False
    else:
        print(f"  ✓ autodock4_atom_types_elements.py: {len(am)} 種金屬全部存在")

    # 2. ad4_types.json
    types_path = meeko_root / 'data/params/ad4_types.json'
    data = json.loads(types_path.read_text())
    atypes = {e['smarts'] for e in data['ad4_types']}
    missing_types = [m['element'] for m in am
                     if f"[#{m['atomic_num']}]" not in atypes]
    if missing_types:
        print(f"  ✗ ad4_types.json 缺少：{missing_types}")
        all_ok = False
    else:
        print(f"  ✓ ad4_types.json: {len(am)} 種金屬類型全部存在")

    # 3. residue_chem_templates.json — 模板存在 + ambiguous 無衝突
    rct_path = meeko_root / 'data/residue_chem_templates.json'
    rct = json.loads(rct_path.read_text())
    templates = rct.get('residue_templates', {})
    ambiguous = rct.get('ambiguous', {})
    missing_res = [r[0] for r in amr if r[0] not in templates]
    conflict_res = [r[0] for r in amr if r[0] in ambiguous]
    if missing_res:
        print(f"  ✗ residue_chem_templates.json 缺少模板：{missing_res}")
        all_ok = False
    else:
        print(f"  ✓ residue_chem_templates.json: {len(amr)} 種金屬模板存在")
    if conflict_res:
        print(f"  ✗ residue_chem_templates.json ambiguous 仍有衝突：{conflict_res}")
        all_ok = False
    else:
        print(f"  ✓ residue_chem_templates.json: ambiguous 無衝突")

    # 4. rdkitutils.py — covalent_radius
    rku_path = meeko_root / 'utils/rdkitutils.py'
    rku_text = rku_path.read_text()
    m_cr = re.search(r'covalent_radius = \{(.*?)\n\}', rku_text, re.DOTALL)
    if m_cr:
        cr_block = m_cr.group(1)
        missing_cr = [m['element'] for m in am if f'    {m["atomic_num"]}:' not in cr_block]
    else:
        missing_cr = [m['element'] for m in am]
    if missing_cr:
        print(f"  ✗ rdkitutils.py covalent_radius 缺少：{missing_cr}")
        all_ok = False
    else:
        print(f"  ✓ rdkitutils.py: {len(am)} 種金屬 covalent_radius 全部存在")

    # 5. utils/utils.py — mini_periodic_table
    utils_path = meeko_root / 'utils/utils.py'
    utils_text = utils_path.read_text()
    m_pt = re.search(r'mini_periodic_table = \{(.*?)\n\}', utils_text, re.DOTALL)
    if m_pt:
        pt_block = m_pt.group(1)
        missing_pt = [m['element'] for m in am if f'    {m["atomic_num"]}:' not in pt_block]
        if missing_pt:
            print(f"  ✗ utils.py mini_periodic_table 缺少：{missing_pt}")
            all_ok = False
        else:
            print(f"  ✓ utils.py mini_periodic_table: {len(am)} 種金屬原子序全部存在")
    else:
        print(f"  ✗ utils.py 找不到 mini_periodic_table")
        all_ok = False

    if all_ok:
        print(f"\n  所有 {len(am)} 種金屬 patch 已正確套用 ✓")
    print("\n範例：")
    print("  mk_prepare_receptor.py --read_pdb gd_mri.pdb -o rec -p  → Gd")
    print("  mk_prepare_receptor.py --read_pdb ru_drug.pdb  -o rec -p  → Ru")
    print("  mk_prepare_receptor.py --read_pdb lu_dotatate.pdb -o rec -p → Lu")


if __name__ == '__main__':
    main()
