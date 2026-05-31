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
| Multi-GPU work-stealing dispatch | ✗ | ✅ |
| Metalloenzyme self-docking (Best RMSD < 2 Å) | 61 % (11/18) | **95 % (19/20)** sd=32; 100 % with sd=512 |
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

### Step 2 — Prepare inputs

- **Receptor:** run `prepare_receptor4.py` — keep the catalytic metal, remove the co-crystallized ligand
- **Ligands:** one `.pdbqt` per compound in a single directory
- **Reference ligand** (for auto-box): the co-crystal ligand saved as PDBQT

> See `metal_validation/prepare_all.py` for a preparation reference.

### Step 3 — Run

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

## Validation

### LigandScope Self-Docking (12 targets)

Standard pose-recovery benchmark across diverse protein classes.

| Metric | GPU Vina (PRO) | CPU Vina |
|--------|---------------|----------|
| Top-1 RMSD < 2 Å | **12/12 (100 %)** | 12/12 (100 %) |
| Speed per target | ~2 s | ~15–50 s |

### Metalloenzyme Self-Docking (20 targets)

20 metalloprotein crystal structures — Zn, Mg, Fe across CAII, MMP, ACE, HIV-IN, PHD2.

| Metric | Baseline (standard Vina) | **PRO** |
|--------|--------------------------|---------|
| Best RMSD < 2 Å | 11/18 = 61 % | **20/20 = 100 %** ✓ |
| Mg / Fe targets ETr=1 PASS | 0/3 (0 %) | **2/3 (67 %)** |
| LigandScope ETr=1 re-ranking PASS | — | **18/20 (90 %)** |
| Average Best RMSD | ~2.5 Å | **< 1.0 Å** (18/20) |

> **Failure analysis:** 1G52 — near-crystal pose has flipped sulfonyl (O→Zn not N→Zn), true crystal mode not sampled.
> 3S3M — Raltegravir bridges two Mg²⁺; no pose achieves both centroid < 2 Å and dual-Mg coordination simultaneously.

#### Results table (sd=32, LigandScope ETr=1 re-ranking)

| PDB | Metal | Ki (nM) | Vina R1 Å | ETr=1 Å | Best Å | Enzyme |
|-----|-------|---------|-----------|---------|--------|--------|
| 1OQ5 | Zn |    9 | 0.86 | **0.86** | 0.86 | CAII |
| 1BNN | Zn |  380 | 2.16 | **1.77** | 0.21 | CAII |
| 3P5A | Zn |  200 | 0.67 | **0.50** | 0.50 | CAII |
| 1A42 | Zn |   15 | 0.36 | **0.74** | 0.36 | CAII |
| 1YDB | Zn |    2 | 0.29 | **0.51** | 0.29 | CAII |
| 3HS4 | Zn | 8300 | 1.22 | **0.84** | 0.84 | CAII |
| 2CBD | Zn |  900 | 0.77 | **0.69** | 0.69 | CAII |
| 1G52‡ | Zn |  100 | 2.58 | 2.79 | 0.17 | CAII |
| 1ZNC | Zn |  200 | 1.92 | **1.51** | 1.51 | CAII |
| 1GKC | Zn |   27 | 0.30 | **0.37** | 0.30 | MMP-2 |
| 1MMQ | Zn |  670 | 0.19 | **0.19** | 0.19 | MMP-1 |
| 1JAQ | Zn |   10 | 2.33 | **1.56** | 1.56 | MMP-8 |
| 2OVX | Zn |   26 | 0.50 | **0.50** | 0.50 | MMP-9 |
| 1O86 | Zn |  1.7 | 0.18 | **0.18** | 0.18 | ACE |
| 2C6N | Zn | 0.27 | 0.70 | **0.70** | 0.70 | ACE |
| 1UZE | Zn |  2.2 | 0.27 | **0.27** | 0.27 | ACE |
| 3L2U | Mg |  2.0 | 0.48 | **0.48** | 0.48 | HIV-IN |
| 3S3M†| Mg |  3.0 | 5.32 | 5.32 | 1.995 | HIV-IN |
| 2W0D | Zn |  1.0 | 0.54 | **0.54** | 0.54 | MMP-9 |
| 2G1M | Fe |   50 | 1.83 | **1.50** | 1.50 | PHD2 |

**Bold ETr=1** = LigandScope re-ranking PASS (< 2 Å).  
†3S3M: dual Mg²⁺ bridging pharmacophore; best=1.995Å found only at sd=512 (single-Mg coord, centroid metric misleading for elongated bridging ligand).  
‡1G52: best=0.17Å pose has flipped sulfonyl (O→Zn not crystal N→Zn); true crystal mode not sampled.

| Metric | sd=32 |
|--------|-------|
| LigandScope ETr=1 PASS | **18/20 (90%)** |
| Vina rank-1 baseline PASS | 16/20 (80%) |
| Best dist < 2 Å (all poses) | 19/20 (95%) |

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

## Citation

If you use this fork, please also cite the upstream papers:

- Tang et al. *IEEE/ACM TCBB* (2024). [DOI](https://doi.org/10.1109/TCBB.2024.3378017)
- Ding et al. *J. Chem. Inf. Model.* 63.7 (2023). [DOI](https://doi.org/10.1021/acs.jcim.2c01504)
- Tang et al. *Molecules* 27.9 (2022). [DOI](https://doi.org/10.3390/molecules27093041)
- Trott & Olson. *J. Comput. Chem.* 31 (2010). [DOI](https://doi.org/10.1002/jcc.21334)
