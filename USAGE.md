# Vina-GPU 2.1 PRO — 完整使用指南

> **傻瓜版：** 看第 1 節就夠用。進階需求再往下翻。

---

## 1. 三種模式，一眼選定

```
有 crystal 共晶結構？
  ├─ 是  → 用 --ref 自動算口袋位置（最簡單）
  └─ 否  → 手動填 --center_x/y/z 或用 --config 檔

配體數量？
  ├─ 1 個  → --ligand ligand.pdbqt
  ├─ 多個  → --ligand_directory ./ligands/
  └─ 雙配體同時 dock  → --ligand A.pdbqt --ligand2 B.pdbqt
```

---

## 2. 模式 A — 單配體 + 自動口袋（有 crystal）

```bash
./AutoDock-Vina-GPU-2-1 \
  --receptor   receptor.pdbqt \
  --ligand     my_ligand.pdbqt \
  --ref        crystal_ligand.pdbqt \   # ← 自動計算 box center + size
  --out        result.pdbqt \
  --thread 8000  --search_depth 32  --cpu 8 \
  --gpu_id 0,1 \
  --opencl_binary_path /path/to/Vina-GPU-2.1
```

輸出 log：
```
Auto-box from crystal_ligand.pdbqt:
  center = (-4.032,  4.467, 14.702)
  size   = (20.0 x 21.4 x 20.0) A
```

---

## 3. 模式 B — 配體資料夾 + 自動口袋（最常用）

```bash
./AutoDock-Vina-GPU-2-1 \
  --receptor         receptor.pdbqt \
  --ligand_directory ./ligands/ \
  --output_directory ./output/ \          # ← 自動建立，每個配體存一個 _out.pdbqt
  --ref              crystal_ligand.pdbqt \
  --thread 8000  --search_depth 32  --cpu 8 \
  --gpu_id 0,1 \
  --opencl_binary_path /path/to/Vina-GPU-2.1
```

- `./ligands/` 內每個 `.pdbqt` 各自跑一次
- `./output/` 內自動產生對應的 `xxx_out.pdbqt`
- **不需要手動拆資料夾**，兩張 GPU 自動分配

---

## 4. 模式 C — 配體資料夾 + 手動口袋

```bash
./AutoDock-Vina-GPU-2-1 \
  --receptor         receptor.pdbqt \
  --ligand_directory ./ligands/ \
  --output_directory ./output/ \
  --center_x 10.5  --center_y 14.2  --center_z 14.5 \
  --size_x 25  --size_y 25  --size_z 25 \
  --thread 8000  --search_depth 32  --cpu 8 \
  --gpu_id 0,1 \
  --opencl_binary_path /path/to/Vina-GPU-2.1
```

---

## 5. 模式 D — Config 檔（推薦長期使用）

```bash
./AutoDock-Vina-GPU-2-1 --config config.txt --gpu_id 0,1
```

`config.txt` 內容：
```ini
# 受體與配體
receptor           = /abs/path/receptor.pdbqt
ligand_directory   = /abs/path/ligands
output_directory   = /abs/path/output

# OpenCL 快取（第一次跑 ~30s 編譯，之後直接讀）
opencl_binary_path = /abs/path/ocl_cache

# 口袋位置
center_x = 10.5
center_y = 14.2
center_z = 14.5
size_x   = 25
size_y   = 25
size_z   = 25

# GPU 設定
thread       = 8000
search_depth = 32
cpu          = 8
```

範本在 `example/` 目錄：
| 檔案 | 用途 |
|------|------|
| `example/config_screen.txt` | 大規模篩選（sd=1，快） |
| `example/config_balanced.txt` | 平衡模式（sd=8） |
| `example/config_accurate.txt` | 精準模式（sd=32） |

---

## 6. 模式 E — 雙配體共同 Docking

```bash
./AutoDock-Vina-GPU-2-1 \
  --receptor   receptor.pdbqt \
  --ligand     ligand_A.pdbqt \
  --ligand2    ligand_B.pdbqt \
  --out        result_A.pdbqt \
  --out2       result_B.pdbqt \
  --ref        crystal_ligand.pdbqt \
  --thread 8000  --search_depth 32  --cpu 8 \
  --gpu_id 0,1 \
  --opencl_binary_path /path/to/Vina-GPU-2.1
```

- A 和 B 同時在口袋內 dock，互相考慮碰撞
- 兩張 GPU 各跑不同 seed，結果合併

---

## 7. 所有參數一覽

| 參數 | 預設 | 說明 |
|------|------|------|
| `--receptor` | 必填 | 受體 PDBQT（保留金屬，移除共晶配體） |
| `--ligand` | — | 單一配體 PDBQT |
| `--ligand_directory` | — | 配體資料夾（虛擬篩選用） |
| `--ligand2` | — | 第二配體（啟動雙配體模式） |
| `--ref` | — | 共晶/參考配體 → 自動算 box |
| `--center_x/y/z` | — | 口袋中心坐標（手動） |
| `--size_x/y/z` | — | 搜尋盒大小 Å（手動） |
| `--out` | 自動命名 | 單配體輸出 PDBQT |
| `--out2` | — | 第二配體輸出 PDBQT |
| `--output_directory` | — | 資料夾模式的輸出目錄（自動建立） |
| `--thread` | `8000` | GPU work-items，RTX/A-series 固定 8000 |
| `--search_depth` | `32` | MC 步數（1=快速篩選，32=精準） |
| `--cpu` | `1` | 平行解析配體的 CPU 執行緒數 |
| `--gpu_id` | `0` | GPU 選擇：`0`、`0,1`、`all` |
| `--opencl_binary_path` | `.` | OCL 核心快取目錄 |
| `--config` | — | 把上面所有選項寫進一個檔案 |
| `--ad4zn` | off | 使用 AutoDock4Zn Zn 參數 |

---

## 8. --ref 自動口袋說明

```
crystal_ligand.pdbqt
      │
      ▼ 讀取全部 ATOM/HETATM 坐標
      │
      ├─ center = 所有原子的幾何重心
      └─ size   = 每軸 (max - min + 10 Å)，最小 20 Å

例：11 個原子的配體 (extent ~8Å × 7Å × 6Å)
  → size = 18 × 17 × 16 Å
```

也可以 `--ref` 配 `--size_x/y/z` 手動覆蓋 size：
```bash
--ref crystal.pdbqt --size_x 30 --size_y 30 --size_z 30
```
*(尚未支援，size 目前由 --ref 自動決定)*

---

## 9. 速度速查（sd=32，thread=8000，cpu=8）

| 配體數 | 1 GPU | 2 GPU（`--gpu_id 0,1`） |
|-------:|------:|----------------------:|
| 100 | 1.2 分鐘 | **38 秒** |
| 1,000 | 12.4 分鐘 | **6.3 分鐘** |
| 10,000 | 2.1 小時 | **1.1 小時** |
| 100,000 | 20.8 小時 | **10.6 小時** |

> 基準：2× NVIDIA RTX 6000 Ada，實測 **2.62 mol/s**（2 GPU）

---

## 10. 典型工作流

```bash
# 1. 準備受體（保留 Zn/Mg/Fe 等金屬，移除共晶配體）
prepare_receptor4.py -r protein.pdb -o receptor.pdbqt -A hydrogens

# 2. 準備配體資料夾
for f in compounds/*.mol2; do
  prepare_ligand4.py -l "$f" -o ligands/$(basename "${f%.mol2}").pdbqt
done

# 3. 直接跑（--ref 自動算口袋）
./AutoDock-Vina-GPU-2-1 \
  --receptor         receptor.pdbqt \
  --ligand_directory ./ligands/ \
  --output_directory ./output/ \
  --ref              co_crystal.pdbqt \
  --thread 8000  --search_depth 32  --cpu 8 \
  --gpu_id 0,1 \
  --opencl_binary_path /path/to/Vina-GPU-2.1

# 4. 按分數排序最佳命中
grep "VINA RESULT" output/*_out.pdbqt | sort -k4 -n | head -20
```
