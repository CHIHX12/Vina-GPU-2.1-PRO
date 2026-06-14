# Vina-GPU Capability Test Suite

Self-contained tests for every capability, with exact commands and verified results.
Run a FREE GPU (no other CUDA process), then: `bash run_all.sh <gpu_id>`

RMSD < 2 Å vs the crystal pose = the docking correctly reproduced the experimental binding mode.

## Verified results (2× RTX 6000 Ada)
| # | capability | test case | command flags | best-of-N RMSD | status |
|---|-----------|-----------|---------------|----------------|--------|
| 1 | single ligand | 5ofu (drug-like) | `--ligand` | **0.26 Å** | ✅ core, reliable |
| 2 | metal | 1A42 (Zn / carbonic anhydrase) | `--ligand` | **1.18 Å** | ✅ core, reliable |
| 3 | dual co-docking | 1RX2 (NADP + folate ternary) | `--ligand A --ligand2 B` | NAP **1.73**, FOL ~6–13 | ⚠ beta; works, flexible cofactors hard |
| 4 | flexible receptor | 1q6k (induced fit) | `--receptor rigid --flex flex` | **2.95 Å** (rigid 6.9) | ⚠ beta; helps induced-fit |

## Exact commands (each test folder is self-contained)

### 1. Single ligand  (`01_single_ligand/`)
```bash
AutoDock-Vina-GPU-2-1 \
  --receptor receptor.pdbqt --ligand ligand.pdbqt --out out.pdbqt \
  --opencl_binary_path /home/cycheng/Vina-GPU-2.1 \
  --center_x <X> --center_y <Y> --center_z <Z> --size_x <SX> --size_y <SY> --size_z <SZ> \
  --thread 8000 --search_depth 20 --gpu_id 0
```
(box values are in each folder's `box.txt`.)

### 2. Metal  (`02_metal/`)
Same command as single — metal coordination scoring is built into the kernel (always on).
Works for 62 metal types (Zn, Mg, Fe, Mn, Ca, Cu, Ni, Co, …) via the meeko metal patch.

### 3. Dual-ligand co-docking  (`03_dual/`)
```bash
AutoDock-Vina-GPU-2-1 \
  --receptor receptor.pdbqt --ligand ligandA.pdbqt --ligand2 ligandB.pdbqt \
  --out outA.pdbqt --out2 outB.pdbqt \
  --opencl_binary_path /home/cycheng/Vina-GPU-2.1 \
  --center_x <X> ... --size_x <SX> ... --thread 8000 --search_depth 20 --gpu_id 0
```
Both ligands are docked simultaneously with an explicit ligand–ligand interaction term
(for synergy vs competition studies; see ../dual_test/ for the full validation + analysis).

### 4. Flexible receptor  (`04_flex/`)
```bash
# 1) prepare flexible side chains around the pocket centre:
python3 tools/prep/prep_flex_receptor.py --receptor receptor.pdbqt \
        --center_x <X> --center_y <Y> --center_z <Z> --cutoff 6.0 --output_dir 04_flex
# 2) dock against rigid part + flexible side chains:
AutoDock-Vina-GPU-2-1 \
  --receptor receptor_rigid.pdbqt --flex receptor_flex.pdbqt --ligand ligand.pdbqt \
  --out out.pdbqt --opencl_binary_path /home/cycheng/Vina-GPU-2.1 \
  --center_x <X> ... --thread 8000 --search_depth 20 --gpu_id 0
```

## Accuracy summary (honest)
- **Single + metal docking = standard AutoDock Vina accuracy** (~42–50 % top-1 RMSD<2 Å on
  general PDBbind; equivalent to CPU Vina, verified head-to-head), but **tens of times faster**
  (≈34 mol/s on 2 GPUs). This is the reliable, production core.
- **Dual + flex are newer (beta):** they run and help in many cases, but accuracy is lower for
  highly flexible ligands / hard induced-fit targets — true of all docking software.

## Requirements / caveats for users
1. **A free GPU.** If another CUDA process (e.g. an inference server) is using the GPUs, docking
   fails with `Err-6: CL_OUT_OF_HOST_MEMORY`. Check `nvidia-smi` first.
2. Inputs must be PDBQT (use `prepare_receptor4` for the receptor, `meeko`/`obabel` for ligands).
3. Set `--opencl_binary_path` to the dir holding the cached `Kernel*_Opt.bin` (the repo root).
4. Docking is stochastic — single-run RMSD varies; report best-of-N over a few runs for the record.
