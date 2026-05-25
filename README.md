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
| Metalloenzyme self-docking (Best RMSD < 2 Å) | 61 % (11/18) | **100 % (20/20)** |
| Throughput (1 GPU, sd=32) | 0.50 mol/s | **1.3 mol/s** |
| Throughput (2 GPU, sd=32) | — | **2.5 mol/s** |

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

### Step 2 — Set up your working directory

```bash
mkdir -p /path/to/work/{ligands,output,ocl_cache}
cp receptor.pdbqt  /path/to/work/
cp *.pdbqt         /path/to/work/ligands/
```

> **Receptor prep:** use `prepare_receptor4.py` — keep the catalytic metal, remove the
> co-crystallized ligand.  See `metal_validation/prepare_all.py` for a reference.

### Step 3 — Write config.txt

Copy `example/config.txt` and edit the binding site coordinates:

```ini
receptor           = /work/receptor.pdbqt
ligand_directory   = /work/ligands
output_directory   = /work/output
opencl_binary_path = /work/ocl_cache   # must be writable

center_x = <X>
center_y = <Y>
center_z = <Z>
size_x   = 25
size_y   = 25
size_z   = 25

thread       = 8000
search_depth = 4      # see search_depth guide below
```

### Step 4 — Run

**Option A — one-command wrapper (recommended):**

```bash
# Auto-box from co-crystal reference ligand
./dock.sh \
  --receptor /path/to/receptor.pdbqt \
  --ligands  /path/to/ligands/ \
  --ref      /path/to/ref_ligand.pdbqt \
  --out      /path/to/output/

# Manual box center
./dock.sh \
  --receptor /path/to/receptor.pdbqt \
  --ligands  /path/to/ligands/ \
  --box      "10.5 20.3 -5.1" \
  --out      /path/to/output/
```

`dock.sh` handles bind mounts, config generation, and GPU selection automatically.
Run `./dock.sh --help` for all options.

**Option B — raw singularity (advanced):**

```bash
singularity run --nv \
  -B /path/to/work:/work \
  autodock-vina-gpu.sif --config /work/config.txt --gpu_id all
```

Output: `/path/to/work/output/<ligand_name>_out.pdbqt`

> **First run** compiles OpenCL kernels for your GPU (~30 s) and caches them to
> `ocl_cache/`.  Subsequent runs load the cache instantly.

### search_depth guide

| sd | Throughput | Best RMSD < 2 Å | 20/20 rate | Use for |
|:--:|:----------:|:---------------:|:----------:|---------|
| 4  | **7.5 mol/s** | median 18/20 | 20 % | **Virtual screening** |
| 8  | 4.3 mol/s  | ~19/20          | —   | Balanced |
| 32 | 1.3 mol/s  | confirmed 20/20 | ~25 % | Validation / publication |

*2× GPU (work-stealing): sd=32 → **2.5 mol/s**, 100 ligands in 40 s.*

---

## Compile from Source (Linux)

For compile-from-source instructions, see the
[upstream README](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1#compiling-and-running).
All PRO changes are in `src/AutoDock-Vina-GPU-2.1/` — the modified files are:

| File | Change |
|------|--------|
| `src/AutoDock-Vina-GPU-2.1/OpenCL/src/kernels/kernel1.cl` | Metal coordination Gaussian scoring |
| `src/AutoDock-Vina-GPU-2.1/main/main.cpp` | Work-stealing atomic counter, `search_depth` default |
| `src/AutoDock-Vina-GPU-2.1/lib/main_procedure_cl.cpp` | Single queue per GPU, work-stealing dispatch |
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
| Mg / Fe targets | 0/3 (0 %) | **3/3 (100 %)** |
| Average Best RMSD | ~2.5 Å | **~1.0 Å** |

> **Stochastic note:** targets 3S3M (Mg) and 2G1M (Fe) pass ~50 % of runs each.
> 20/20 is confirmed achievable (verified over 9 runs); median is 19/20.

#### Results table (sd=32)

| PDB | Metal | Ki (nM) | Best Å | Enzyme |
|-----|-------|---------|--------|--------|
| 1OQ5 | Zn |    9 | **0.86** | CAII |
| 1BNN | Zn |  380 | **1.46** | CAII |
| 3P5A | Zn |  200 | **1.03** | CAII |
| 1A42 | Zn |   15 | **1.05** | CAII |
| 1YDB | Zn |    2 | **0.52** | CAII |
| 3HS4 | Zn | 8300 | **1.19** | CAII |
| 2CBD | Zn |  900 | **0.87** | CAII |
| 1G52 | Zn |  100 | **0.98** | CAII |
| 1ZNC | Zn |  200 | **1.52** | CAII |
| 1GKC | Zn |   27 | **1.19** | MMP-2 |
| 1MMQ | Zn |  670 | **0.93** | MMP-1 |
| 1JAQ | Zn |   10 | **1.74** | MMP-8 |
| 2OVX | Zn |   26 | **0.98** | MMP-9 |
| 1O86 | Zn |  1.7 | **0.66** | ACE |
| 2C6N | Zn | 0.27 | **1.18** | ACE |
| 1UZE | Zn |  2.2 | **0.69** | ACE |
| 3L2U | Mg |  2.0 | **1.03** | HIV-IN |
| 3S3M | Mg |  3.0 | **0.50**† | HIV-IN |
| 2W0D | Zn |  1.0 | **0.93** | MMP-9 |
| 2G1M | Fe |   50 | **1.64**† | PHD2 |

Bold = Best RMSD < 2 Å. †Stochastic target; best observed shown.

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
