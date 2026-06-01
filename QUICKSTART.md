# Vina-GPU 2.1 PRO — 快速指南

硬體基準：2× NVIDIA RTX 6000 Ada，`thread=8000`，`sd=32`

---

## 1. 準備檔案

```
work/
├── receptor.pdbqt      ← 受體（保留金屬，移除共晶配體）
├── ligands/            ← 每個配體一個 .pdbqt 檔
│   ├── mol_001.pdbqt
│   ├── mol_002.pdbqt
│   └── ...
└── output/             ← 自動建立
```

**受體準備：**
```bash
prepare_receptor4.py -r protein.pdb -o receptor.pdbqt -A hydrogens
```

**配體準備：**
```bash
prepare_ligand4.py -l ligand.mol2 -o ligand.pdbqt
```

---

## 2. 執行（直接二進位）

```bash
./AutoDock-Vina-GPU-2-1 \
  --receptor           receptor.pdbqt \
  --ligand_directory   ./ligands \
  --output_directory   ./output \
  --center_x 10.5  --center_y 14.2  --center_z 14.5 \
  --size_x 25  --size_y 25  --size_z 25 \
  --thread 8000  --search_depth 32  --cpu 8 \
  --gpu_id 0,1 \
  --opencl_binary_path /path/to/Vina-GPU-2.1
```

**不需要手動拆分配體資料夾。**  
`--gpu_id 0,1` 自動把所有配體分配給兩張 GPU 同時計算（work-stealing）。

---

## 3. 速度（實測，sd=32，2× RTX 6000 Ada）

| 配體數 | 時間 |
|-------:|------|
| 100 | **38 秒** |
| 500 | **3.2 分鐘** |
| 1,000 | **6.3 分鐘** |
| 5,000 | **32 分鐘** |
| 10,000 | **1.1 小時** |
| 50,000 | **5.3 小時** |
| 100,000 | **10.6 小時** |

> 速率：**2.62 mol/s**（2 GPU）、**1.34 mol/s**（1 GPU）

---

## 4. 參數說明

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `--thread` | `8000` | GPU work-items；RTX / A-series 固定用 8000 |
| `--search_depth` | `32` | MC 步數；32 = 發表級精準度 |
| `--cpu` | `8` | 平行解析配體的 CPU 執行緒數 |
| `--gpu_id` | `0,1` 或 `all` | 指定 GPU；`all` 自動使用全部 |
| `--center_x/y/z` | 口袋中心 | 用 PyMOL 或 Chimera 量測 |
| `--size_x/y/z` | `25`（一般） | 搜尋盒半邊長（Å） |

---

## 5. 多 GPU 機制

```
./AutoDock-Vina-GPU-2-1 --gpu_id 0,1

                std::atomic next_lig = 0
                      │
          ┌───────────┴───────────┐
     GPU 0 thread              GPU 1 thread
     (std::async)              (std::async)
          │                         │
     fetch_add() → 拿 0,2,4...  fetch_add() → 拿 1,3,5...
          │                         │
     各處理 ~N/2 個配體          各處理 ~N/2 個配體
          └───────────┬───────────┘
                 合併結果輸出
```

- 兩張 GPU **真正同時運行**（`std::async::launch::async`）
- 哪張快就多拿，自動負載平衡
- **無需手動拆資料夾**

---

## 6. 大規模篩選建議（三階段）

```
1 億配體
   ↓  Stage 0：RDKit 過濾  (MW≤500, rotbond≤10)
1 千萬
   ↓  Stage 1：sd=1, cpu=4, --gpu_id 0,1   → 27 mol/s → ~4.3 天
10 萬
   ↓  Stage 2：sd=8, cpu=2, --gpu_id 0,1   → ~1 mol/s → ~28 小時
1 千
   ↓  Stage 3：sd=32, cpu=8, --gpu_id 0,1  → 2.62 mol/s → ~6 分鐘
最終排名結果
```

config 範本在 `example/` 目錄：
- `example/config_screen.txt`   — Stage 1
- `example/config_balanced.txt` — Stage 2
- `example/config_accurate.txt` — Stage 3

---

## 7. 金屬酵素注意事項

支援 Zn、Mg、Fe、Mo、Mn、Ca 等金屬：

- 受體準備時**保留**金屬原子，移除共晶配體
- `prepare_receptor4.py` 會自動指定正確原子類型
- Mo 等稀有金屬已通過驗證（3NRZ，RMSD best = 0.64 Å）

---

## 8. 輸出格式

```
output/mol_001_out.pdbqt   ← 9 個姿態，按 Vina 分數排列
```

第一個 REMARK 行即為最佳分數：
```
REMARK VINA RESULT:    -8.500      0.000      0.000
```

---

*硬體：2× NVIDIA RTX 6000 Ada Generation（VRAM 48 GB each）*  
*驗證：20 個金屬酵素靶點，sd=32，Best RMSD < 2 Å = 19/20（95 %）*
