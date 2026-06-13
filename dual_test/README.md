# Dual-Ligand Co-Docking Test Suite

Validate the GPU dual-ligand co-docking (synergy vs competition studies) against
**real two-ligand co-crystal structures** and against the **AutoDock Vina 1.2.7**
native multi-ligand reference.

## Background
- **Vina-GPU dual co-docking** (`--ligand A --ligand2 B --out .. --out2 ..`) docks
  two ligands simultaneously, scoring with an explicit ligand–ligand interaction
  term (`eval_lig_lig_cl`). It was a non-functional stub (CL_OUT_OF_HOST_MEMORY)
  until the private-memory fix (commit 226ee69); now it runs.
- **CPU AutoDock Vina 1.1.2** (in env `jp_214`) does NOT support multi-ligand.
  **Vina 1.2.7** (installed at `~/.local/bin/vina12`) does — used here as the
  reference for "true" simultaneous co-docking.

## Files
| file | purpose |
|------|---------|
| `targets.txt` | candidate two-ligand PDB IDs |
| `prepare_target.py` | download a PDB, auto-find the 2 closest drug-like ligands, prepare `rec.pdbqt` + `ligA/B.pdbqt` + crystal refs + `box.txt` into `<ID>/` |
| `run_dual.sh <ID>` | run Vina-GPU dual **and** Vina 1.2.7 native multi-ligand |
| `analyze.py <ID>` | best-of-N RMSD vs crystal (GPU and CPU) + synergy/competition verdict |
| `run_all.sh` | run+analyze all prepared targets → `summary.tsv` |
| `<ID>/` | per-target prepared files, docked outputs, `result.txt` |

## Usage
```bash
cd dual_test
python3 prepare_target.py 1HVY        # or no args = all of targets.txt
bash run_dual.sh 1HVY                  # GPU dual + Vina 1.2.7
python3 analyze.py 1HVY                # RMSD vs crystal + verdict
```

## Synergy vs competition interpretation
The two ligands are co-docked. From the best poses:
- **COMPETITIVE** — poses overlap/clash (many heavy-atom pairs <2 Å): the two
  ligands want the same sub-site, cannot co-bind.
- **SYNERGY candidate** — adjacent, non-clashing, in contact (<4 Å): possible
  cooperative co-binding.
- **INDEPENDENT** — separate regions, little interaction.
(The verdict is only as good as the docked poses — flexible ligands that dock
poorly can give a misleading clash.)

## Validation status (see each `<ID>/result.txt` and `summary.tsv`)
The realistic two-ligand pockets are enzyme **ternary complexes** (cofactor +
substrate/inhibitor), and cofactors (NADP/ADP/antifolates, 18–34 torsions) are
intrinsically very flexible — hard for any docker.

Example (1HVY, thymidylate synthase + antifolate D16 + dUMP):
| ligand | GPU dual best | Vina 1.2.7 best |
|--------|------|------|
| UMP (10 tors, drug-like) | 1.31 Å | 1.10 Å |
| D16 (18 tors, flexible)  | 9.82 Å | 4.40 Å |

Takeaway: **GPU dual is competitive with the CPU reference for low-torsion,
drug-like ligands; weaker for highly flexible ones** (the GPU's many-shallow-
trajectory search under-samples high-torsion conformers). For drug-like two-ligand
studies the accuracy is good; for flexible cofactors prefer Vina 1.2.7 or raise
`--search_depth`.
