# Vina-GPU 2.1 PRO

A performance-optimized fork of [DeltaGroupNJUPT/Vina-GPU-2.1](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1)
with two key additions: **metal coordination scoring** for metalloenzyme targets and
**multi-GPU work-stealing dispatch** for large-scale virtual screening.

> For general AutoDock Vina background, theory, and upstream usage docs, see the
> [upstream repository](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1).

---

## What's New in PRO

| Feature | Upstream | PRO |
|---------|----------|-----|
| Metal coordination scoring (Zn, Mg, Fe, Mn, Ca) | ✗ | ✅ |
| **Receptor prep: full periodic table** (62 metals, no recompile) | ✗ | ✅ |
| Multi-GPU work-stealing dispatch | ✗ | ✅ |
| PDBbind self-docking — Best RMSD < 2 Å (1000 targets) | — | **67.1 % (12,291/18,320)** |
| PDBbind self-docking — Best RMSD < 1 Å (1000 targets) | — | **44.6 % (8,179/18,320)** |
| Metalloenzyme self-docking Best RMSD < 2 Å (19 targets) | 61 % (11/18) | **95 % (18/19)** |
| Throughput (1 GPU, sd=32, thread=8000) | 0.50 mol/s | **1.34 mol/s** |
| Throughput (2 GPU, sd=32, thread=8000) | — | **2.59 mol/s** |
| Throughput (2 GPU, sd=1,  thread=8000) | — | **27 mol/s** (with `--cpu 4`) |
| Parallel ligand parsing (`--cpu N`) | ✗ | ✅ +39 % at sd=1 |

---

## Quick Start (Singularity / HPC)

**Prerequisites:** Docker, Singularity/Apptainer, NVIDIA GPU with CUDA drivers.

### Step 1 — Build the SIF (one-time, ~5 min)

```bash
git clone https://github.com/CHIHX12/Vina-GPU-2.1-PRO.git
cd Vina-GPU-2.1-PRO
docker build -t autodock-vina-gpu .
singularity build autodock-vina-gpu.sif docker-daemon://autodock-vina-gpu:latest
```

### Step 2 — Patch meeko (one-time, any metal in the periodic table)

Standard meeko only recognises Zn/Fe/Mg/Ca/Mn.  
Run this **once** after installing meeko to enable all 62 metals (Cu, Ru, Gd, Ac, Th …):

```bash
# activate the environment where meeko is installed
conda activate jp_214
PYTHONNOUSERSITE=1 python3 tools/patch_meeko_metals.py
```

> **No Vina recompilation needed.**  
> Metal support lives entirely in meeko (receptor/ligand `.pdbqt` preparation).  
> Vina-GPU reads only the atom types in the PDBQT — it does not care which element produced them.  
> To undo: `python3 tools/patch_meeko_metals.py --restore`

### Step 3 — Prepare inputs

- **Receptor:** `mk_prepare_receptor.py --read_pdb receptor.pdb -o rec -p`  
  Keep the catalytic metal atom; remove the co-crystallised ligand before prep.
- **Ligands:** one `.pdbqt` per compound in a single directory
- **Reference ligand** (for auto-box): the co-crystal ligand saved as PDBQT

```bash
# Example: Ru-containing receptor
PYTHONNOUSERSITE=1 mk_prepare_receptor.py --read_pdb 1RU.pdb -o 1RU_rec -p
```

> See `metal_validation/prepare_all.py` for a batch preparation reference.

### Step 4 — Run

```bash
./dock.sh \
  --receptor receptor.pdbqt \
  --ligands  ./ligands/ \
  --ref      co_crystal_ligand.pdbqt \
  --out      ./output/
```

`dock.sh` auto-computes the binding box from the reference ligand and handles all
Singularity bind mounts internally.  No config file needed.

Output: `./output/<ligand_name>_out.pdbqt` — 9 poses per ligand, ranked by affinity.

> **First run** compiles OpenCL kernels (~30 s) and caches them.  Subsequent runs are instant.

#### dock.sh options

| Option | Default | Description |
|--------|---------|-------------|
| `--receptor` / `-r` | required | Receptor PDBQT file |
| `--ligands` / `-l` | required | Directory of ligand PDBQT files |
| `--ref` | — | Co-crystal ligand → auto-compute box center + size |
| `--box "X Y Z"` | — | Box center coordinates (required if no `--ref`) |
| `--size N` | auto | Box size in Å; one value or `"SX SY SZ"` |
| `--out` / `-o` | `./output` | Output directory |
| `--depth` | `4` | search_depth (see guide below) |
| `--gpu` | `all` | GPU selection: `all`, `0`, `1`, `0,1` |
| `--threads` | `8000` | OpenCL thread count |
| `--sif` | `./autodock-vina-gpu.sif` | Path to SIF image |

When `--ref` is given, box center = ligand centroid and box size = ligand extent + 10 Å margin.
When no co-crystal structure is available, specify the binding site manually:

```bash
./dock.sh \
  --receptor receptor.pdbqt \
  --ligands  ./ligands/ \
  --box "10.5 20.3 -5.1" --size 25 \
  --out ./output/
```

#### search_depth guide

All numbers measured on 2× RTX 6000 Ada, `thread=8000`.

| sd | 1 GPU | 2 GPU | Accuracy | Best `--cpu` | Use for |
|:--:|------:|------:|:--------:|:------------:|---------|
|  1 | ~10 mol/s | ~27 mol/s | coarse | **4** | Stage-1 library screening |
|  8 |  ~0.5 mol/s | ~1.0 mol/s | good | **2** | Stage-2 hit re-scoring |
| 32 | **1.34 mol/s** | **2.59 mol/s** | publication | **1** | Final re-dock / validation |

> At `sd=32` the GPU takes ~387 ms/ligand; parsing overhead (~14 ms/lig) is only 3.6% —
> `--cpu 1` is the optimal choice and extra threads give no measurable benefit.  
> At `sd=1` parsing is 38% of wall time — `--cpu 4` recovers that cost entirely.

#### Advanced — raw Singularity

For HPC environments where `dock.sh` is not suitable, copy `example/config.txt`,
edit the binding site coordinates, then:

```bash
singularity run --nv \
  -B /path/to/work:/work \
  autodock-vina-gpu.sif --config /work/config.txt --gpu_id all
```

---

## Virtual Screening Pipeline (1 B ligands → final hits)

A recommended three-stage funnel that balances throughput against accuracy.
Ready-made config files for each stage are in `example/`.

```
1,000,000,000 compounds
        │
        ▼ Stage 0 — CPU pre-filter (RDKit)
        │   MW ≤ 500, rotbonds ≤ 10, Lipinski/PAINS filter
        │   ~hours on CPU alone
        ▼
   10,000,000 compounds
        │
        ▼ Stage 1 — Fast GPU screen   (example/config_screen.txt)
        │   thread=8000  sd=1  --cpu 4  --gpu_id 0,1
        │   27 mol/s (2 GPU) → ~4.3 days
        │   Keep top 1 % by Vina score
        ▼
      100,000 compounds
        │
        ▼ Stage 2 — Balanced re-score  (example/config_balanced.txt)
        │   thread=8000  sd=8  --cpu 2  --gpu_id 0,1
        │   ~1 mol/s (2 GPU) → ~28 hours
        │   Keep top 1 % by Vina score
        ▼
        1,000 compounds
        │
        ▼ Stage 3 — Accurate re-dock   (example/config_accurate.txt)
        │   thread=8000  sd=32  --cpu 1  --gpu_id 0,1
        │   2.59 mol/s (2 GPU) → ~6.5 minutes
        ▼
     Final ranked list  (pose files + Vina scores)
```

### Per-stage capacity (sd=32, thread=8000, cpu=8)

If you prefer a **uniform accurate setting** throughout (`sd=32, cpu=8, thread=8000`):

| GPUs | Throughput | Per day | 1 K ligands | 10 K ligands |
|:----:|:----------:|--------:|:-----------:|:------------:|
| 1 | 1.34 mol/s | ~116 K | 12 min | 2.1 h |
| 2 | 2.59 mol/s | ~224 K |  6 min | 1.1 h |
| 4 | ~5.2 mol/s | ~448 K |  3 min | 32 min |

### Multi-GPU distribution

Pass a comma-separated list or `all` — the engine distributes ligands
across GPUs automatically using a work-stealing atomic counter:

```bash
# Two specific GPUs
./AutoDock-Vina-GPU-2-1 --config config_accurate.txt --gpu_id 0,1

# All available GPUs
./AutoDock-Vina-GPU-2-1 --config config_accurate.txt --gpu_id all
```

Each GPU receives an independent OCL context and a unique random seed
(`seed + gpu_index × 10000`), so results are reproducible and poses from
all GPUs are merged and re-ranked before output.

To split a large library manually across machines or containers, divide
the ligand directory into N subdirectories and run one job per GPU server:

```bash
# Machine A — first half
./AutoDock-Vina-GPU-2-1 --config cfg.txt --ligand_directory ./ligs_part1 --gpu_id 0

# Machine B — second half
./AutoDock-Vina-GPU-2-1 --config cfg.txt --ligand_directory ./ligs_part2 --gpu_id 0
```

Then merge and sort the output `.pdbqt` files by the first-line REMARK score.

---

## Compile from Source (Linux)

For compile-from-source instructions, see the
[upstream README](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1#compiling-and-running).
All PRO changes are in `src/AutoDock-Vina-GPU-2.1/` — the modified files are:

| File | Change |
|------|--------|
| `src/AutoDock-Vina-GPU-2.1/OpenCL/src/kernels/kernel1.cl` | Metal coordination Gaussian scoring (Zn/Mg/Fe/Mo …) |
| `src/AutoDock-Vina-GPU-2.1/main/main.cpp` | Parallel ligand parsing (`--cpu N`), multi-GPU dual-ligand dispatch |
| `src/AutoDock-Vina-GPU-2.1/lib/main_procedure_cl.cpp` | Per-GPU OCL context map, work-stealing dispatch |
| `src/AutoDock-Vina-GPU-2.1/OpenCL/inc/wrapcl.h / wrapcl.cpp` | Multi-GPU queue management |

---

## Metal Receptor Preparation (Full Periodic Table)

### Why no recompilation is needed

Vina-GPU reads receptors and ligands as **PDBQT** files.  
The PDBQT format carries only atom coordinates + AD4 atom-type strings (e.g. `Zn`, `Fe`, `Cu`).  
Vina never sees element numbers — it only sees those type strings.

Metal support therefore lives entirely in **meeko** (the preparation tool), not in Vina.  
Adding a new metal = patching meeko's data files so it writes the correct atom-type string into the PDBQT.  
**No C++ / OpenCL recompilation required.**

### One-time setup

```bash
conda activate jp_214
PYTHONNOUSERSITE=1 python3 tools/patch_meeko_metals.py
```

Output confirms 62 metals patched:

```
覆蓋 62 種金屬（基礎 12 + 擴充 50）
  ✓ autodock4_atom_types_elements.py: 62 種金屬全部存在
  ✓ ad4_types.json: 62 種金屬類型全部存在
  ✓ residue_chem_templates.json: 62 種金屬模板存在
  ✓ residue_chem_templates.json: ambiguous 無衝突
  ✓ rdkitutils.py: 62 種金屬 covalent_radius 全部存在
  ✓ utils.py mini_periodic_table: 62 種金屬原子序全部存在
  所有 62 種金屬 patch 已正確套用 ✓
```

### Supported metals (62 total)

| Group | Elements |
|-------|---------|
| Core (meeko default) | Zn, Fe, Mg, Ca, Mn |
| 3d transition | V, Cr, Co, Ni, Cu |
| 4d transition | Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd |
| 5d transition | Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg |
| Post-transition | Al, Ga, Ge, As, Se, In, Sn, Sb, Te, Tl, Pb, Bi |
| Alkali / alkaline earth | Li, Na, K, Rb, Cs, Be, Sr, Ba |
| Lanthanides | La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu |
| Actinides | Ac, Th |

### Usage

```bash
# Receptor with any metal — same command regardless of element
PYTHONNOUSERSITE=1 mk_prepare_receptor.py --read_pdb receptor.pdb -o rec -p

# Examples
PYTHONNOUSERSITE=1 mk_prepare_receptor.py --read_pdb 1RU_ru.pdb  -o ru_rec  -p   # Ru
PYTHONNOUSERSITE=1 mk_prepare_receptor.py --read_pdb gd_mri.pdb  -o gd_rec  -p   # Gd (MRI contrast)
PYTHONNOUSERSITE=1 mk_prepare_receptor.py --read_pdb lu_dotatate.pdb -o lu_rec -p # 177Lu therapy
```

To restore the original meeko files: `python3 tools/patch_meeko_metals.py --restore`

### Self-docking redocking validation (crystal structure overlap)

20 metalloprotein crystal structures redocked; RMSD measured against co-crystal pose:

| PDB | Metal | Enzyme | Top-1 RMSD (Å) | Best RMSD (Å) | Status |
|-----|-------|--------|---------------|--------------|--------|
| 1A42 | Zn | CAII | 1.509 | **1.130** | PASS |
| 1BNN | Zn | CAII | 3.942 | **1.266** | RECOV |
| 1G52 | Zn | CAII | 4.577 | **1.285** | RECOV |
| 1GKC | Zn | MMP-2 | 1.201 | **0.776** | PASS |
| 1JAQ | Zn | MMP-8 | 3.656 | **0.773** | RECOV |
| 1MMQ | Zn | MMP-1 | 0.685 | **0.685** | PASS |
| 1O86 | Zn | ACE | 1.216 | **0.913** | PASS |
| 1OQ5 | Zn | CAII | 1.793 | **1.244** | PASS |
| 1UZE | Zn | ACE | 1.459 | **0.472** | PASS |
| 1YDB | Zn | CAII | 1.259 | **1.259** | PASS |
| 2C6N | Zn | ACE | 1.419 | **0.866** | PASS |
| 2G1M | Fe | PHD2 | 4.864 | 3.143 | FAIL |
| 2OVX | Zn | MMP-9 | 1.081 | **1.081** | PASS |
| 2W0D | Zn | MMP-9 | 1.226 | **1.226** | PASS |
| 3HS4 | Zn | CAII | 2.387 | **1.612** | RECOV |
| 3L2U | Mg | HIV-IN | 1.935 | **1.264** | PASS |
| 3NRZ | Mo | Xanthine oxidase | 2.922 | **0.640** | RECOV |
| 3P5A | Zn | CAII | 3.690 | **1.038** | RECOV |
| 3S3M | Mg | HIV-IN | 1.355 | **0.490** | PASS |

**PASS** = top-1 pose RMSD < 2 Å; **RECOV** = best pose < 2 Å (ranking needs LigandScope re-scoring); **FAIL** = best > 2 Å (2G1M: Fe-PHD2 unusual coordination geometry).

Summary: **18/19 best RMSD < 2 Å** (95 %) — redocked poses overlap with crystal structures.

---

## Validation

### LigandScope Self-Docking (12 targets)

Standard pose-recovery benchmark across diverse protein classes.

| Metric | GPU Vina (PRO) | CPU Vina |
|--------|---------------|----------|
| Top-1 RMSD < 2 Å | **12/12 (100 %)** | 12/12 (100 %) |
| Speed per target | ~2 s | ~15–50 s |

### Metalloenzyme Self-Docking (19 targets)

19 metalloprotein crystal structures — Zn, Mg, Fe, Mo across CAII, MMP, ACE, HIV-IN, PHD2, XO.

| Metric | Baseline (standard Vina) | **PRO** |
|--------|--------------------------|---------|
| Best RMSD < 2 Å | 11/18 = 61 % | **18/19 (95 %)** |
| Vina rank-1 < 2 Å | — | **12/19 (63 %)** |
| LigandScope ETr=1 re-ranking PASS | — | **14/18 (78 %)** |
| Mg / Fe / Mo ETr=1 PASS | 0/3 (0 %) | **2/3 (67 %)** |

> **Failure analysis (2G1M):** Fe-PHD2 with 4HG inhibitor — all 9 poses > 3 Å.
> Unusual Fe²⁺ coordination geometry (distorted octahedral with bulky 4HG) results in no viable binding mode.
>
> **LigandScope re-ranking note:** ETr=1 worsened 1OQ5 (1.244 → 4.419 Å) and 2C6N (1.419 → 8.642 Å).
> Vina default rank-1 already achieves < 2 Å for these two targets — re-ranking not universally beneficial.

#### Results table (sd=32, LigandScope ETr=1 re-ranking, 18 of 19 targets)

| PDB | Metal | Ki (nM) | Vina R1 Å | ETr=1 Å | Best Å | Enzyme |
|-----|-------|---------|-----------|---------|--------|--------|
| 1A42 | Zn |   15 | 1.509 | **1.509** | 1.130 | CAII |
| 1BNN | Zn |  380 | 3.942 | **1.266** | 1.266 | CAII |
| 1G52‡| Zn |  100 | 4.577 | 4.577 | 1.285 | CAII |
| 1GKC | Zn |   27 | 1.201 | **1.201** | 0.776 | MMP-2 |
| 1JAQ | Zn |   10 | 3.656 | **1.332** | 0.773 | MMP-8 |
| 1MMQ | Zn |  670 | 1.127 | **0.685** | 0.685 | MMP-1 |
| 1O86 | Zn |  1.7 | 1.216 | **1.236** | 0.913 | ACE |
| 1OQ5*| Zn |    9 | 1.244 | 4.419 | 1.244 | CAII |
| 1UZE | Zn |  2.2 | 1.459 | **1.459** | 0.472 | ACE |
| 1YDB | Zn |    2 | 1.259 | **1.259** | 1.259 | CAII |
| 2C6N*| Zn | 0.27 | 1.419 | 8.642 | 0.866 | ACE |
| 2G1M | Fe |   50 | 3.143 | 3.911 | 3.143 | PHD2 |
| 2OVX | Zn |   26 | 1.163 | **1.081** | 1.081 | MMP-9 |
| 2W0D | Zn |  1.0 | 4.935 | **1.701** | 1.226 | MMP-9 |
| 3HS4 | Zn | 8300 | 2.387 | **1.859** | 1.612 | CAII |
| 3L2U | Mg |  2.0 | 1.935 | **1.935** | 1.264 | HIV-IN |
| 3NRZ | Mo |20000 | 2.922 | — | 0.640 | XO |
| 3P5A | Zn |  200 | 3.872 | **1.038** | 1.038 | CAII |
| 3S3M | Mg |  3.0 | 1.355 | **1.355** | 0.490 | HIV-IN |

**Bold ETr=1** = LigandScope re-ranking PASS (< 2 Å).  
‡1G52: best=1.285Å pose has flipped sulfonyl (O→Zn instead of crystal N→Zn); correct binding mode not sampled.  
*1OQ5, 2C6N: Vina rank-1 already < 2 Å; ETr=1 re-ranking selected a suboptimal pose for these targets.  
3NRZ: LigandScope re-ranking not yet run; Vina best pose = 0.640 Å (RECOV).

| Metric | sd=32 |
|--------|-------|
| Best dist < 2 Å (all poses) | **18/19 (95 %)** |
| Vina rank-1 PASS | 11–12/19 (58–63 %)† |
| LigandScope ETr=1 PASS | **14/18 (78 %)** |

†Vina rank-1 varies by run (stochastic search); canonical run (metal_results.tsv): 12/19; LigandScope pipeline run: 11/19.

To reproduce: `cd metal_validation && bash reproduce.sh --depth 32`

---

## Metal Coordination Parameters

| Metal | r_ideal (Å) | σ (Å) | weight | Donor types |
|-------|------------|-------|--------|-------------|
| Zn²⁺  | 2.10 (N/O) · 2.30 (S) | 0.35 | 4.0 · 3.5 | N, O, S |
| Mg²⁺  | 2.05 | 0.35 | 10.0 | N, O |
| Fe²⁺  | 2.05 | 0.40 |  8.0 | N, O |
| Mn²⁺  | 2.20 | 0.40 |  7.0 | N, O |
| Ca²⁺  | 2.40 | 0.40 |  5.0 | N, O |

Distances from CSD medians. Implemented as a Gaussian energy well in `kernel1.cl`:
`E = −weight × exp(−(r − r_ideal)² / (2σ²))`

---

## Repository Structure

```
Vina-GPU-2.1/
├── AutoDock-Vina-GPU-2-1   ← compiled binary (RTX 6000 Ada / CUDA)
├── Kernel1_Opt.bin          ← pre-compiled OpenCL kernel (grid)
├── Kernel2_Opt.bin          ← pre-compiled OpenCL kernel (search)
├── dock.sh                  ← one-shot docking wrapper
├── benchmark/               ← throughput & docking benchmark scripts
│   ├── 10lig/               ← 10-ligand throughput test
│   ├── 10dual/              ← dual-ligand mode test
│   └── run_*.sh
├── example/                 ← config templates (screen / balanced / accurate)
├── src/                     ← C++ / OpenCL source
│   └── AutoDock-Vina-GPU-2.1/
│       ├── lib/             ← core scoring, main_procedure_cl
│       ├── main/main.cpp    ← entry point, --cpu, multi-GPU dispatch
│       └── OpenCL/src/kernels/
│           ├── kernel1.cl   ← grid + metal coordination Gaussian
│           └── kernel2.cl   ← BFGS search
├── tools/
│   ├── prep/                ← receptor/ligand preparation
│   │   ├── patch_meeko_metals.py  ← 62-metal meeko patch (one-time)
│   │   ├── smiles_to_pdbqt.py
│   │   └── setup_env.sh
│   ├── diagnostics/         ← pose analysis & scoring debug
│   └── tests/               ← unit tests
├── README.md
├── QUICKSTART.md
└── USAGE.md
```

**LigandScope** (companion scoring engine, `~/LigandScope/`):
```
LigandScope/
├── data/
│   ├── Metal_enzymes/       ← 19 metalloprotein targets (Zn/Mg/Fe/Mo)
│   │   ├── 1A42/            ← {PDB}_ligand.pdbqt, _receptor.pdbqt
│   │   ├── 1JAQ/            ← tripeptide hydroxamate (MMP-8)
│   │   └── ...  (19 targets total)
│   ├── Kinases/             ← 3 kinase targets
│   ├── GPCR/                ← 3 GPCR targets
│   ├── Ion_channels/        ← 3 ion channel targets
│   └── Nuclear_receptors/   ← 3 nuclear receptor targets
├── scripts/
│   ├── core/                ← rank_poses.py, batch_redock_pdbbind.py, ...
│   ├── analysis/            ← calibrate_weights.py, gpu_batch_scoring.py, ...
│   ├── tools/               ← fix_altloc.py, make_pairs_from_pdbqt.py, ...
│   ├── legacy/              ← ML training, old diagnostic scripts
│   └── ligandscope/         ← Python package (energy, structs, output)
├── validation/
│   ├── pdbbind/             ← 18,320 completed PDBbind self-dockings
│   ├── metal/               ← 19 metalloenzyme validation runs
│   └── xo/                  ← xanthine oxidase (Mo) benchmark
└── results/
    ├── pdbbind_results.tsv       ← 19,037 targets, RMSD + LigandScope scores
    └── benchmark_1000.tsv        ← 1,000-target paper benchmark subset
```

---

## Large-Scale Benchmark (PDBbind, 18,320 targets)

Self-docking pose recovery across PDBbind (all years, sd=32, RTX 6000 Ada):

| Metric | Result |
|--------|--------|
| Total targets completed | 18,320 / 19,037 (96.2 %) |
| Best RMSD < **1 Å** | **8,179 / 18,320 = 44.6 %** |
| Best RMSD < **2 Å** | **12,291 / 18,320 = 67.1 %** |
| Median Best RMSD | 1.117 Å |
| Mean Best RMSD | 2.116 Å |

> These statistics come from existing runs in `LigandScope/results/pdbbind_results.tsv`.  
> A curated 1,000-target benchmark subset is available at `LigandScope/results/benchmark_1000.tsv`.

---

## Citation

If you use this fork, please also cite the upstream papers:

- Tang et al. *IEEE/ACM TCBB* (2024). [DOI](https://doi.org/10.1109/TCBB.2024.3378017)
- Ding et al. *J. Chem. Inf. Model.* 63.7 (2023). [DOI](https://doi.org/10.1021/acs.jcim.2c01504)
- Tang et al. *Molecules* 27.9 (2022). [DOI](https://doi.org/10.3390/molecules27093041)
- Trott & Olson. *J. Comput. Chem.* 31 (2010). [DOI](https://doi.org/10.1002/jcc.21334)
