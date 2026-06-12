# GPU Flexible-Receptor Docking — Implementation Spec

**Goal:** Implement flexible-receptor (induced-fit) docking on the GPU kernel. This is the one
physically-justified lever for the ~32% of PDBbind targets that fail rigid docking (their crystal
pose clashes with the rigid receptor — proven by tiny-box diagnostics showing +30 to +76 kcal/mol
steric clashes at the correct pocket center).

## Why (evidence, 2026-06-11)
Systematic experiments ruled out every other cause of the redocking gap:
- Ranking is fine: lowest-energy top-1 = **94%** on well-sampled targets.
- Search depth: heavy sampling stochastic; 92% of failures stay failures.
- Funnel-density re-ranking: falsified (33% vs 94%).
- Metropolis RNG decorrelation: measured neutral.
- Box tightening: 1/20 rescued.
- **Hard failures = the crystal pose does not fit the RIGID receptor (clash).** Only receptor
  flexibility can fix this.

## Current state — flex is a STUB
- `main_procedure_cl.cpp:827` → `throw "m.other_pairs is not supported!"`.
- `quasi_newton.cl:626` comment → `// Only support one ligand, no flex !`.
- `m_cl`/`output_type_cl` already carry `flex_torsion[MAX_NUM_OF_FLEX_TORSION=12]`; `mutate_conf.cl`
  can perturb flex torsions — but the kernel never turns them into sidechain coords, and no
  flex interaction energy is computed.
- CPU model (`model.h`) HAS full flex support: `ligands` + `flex` heterotrees, `other_pairs`,
  forward-kinematics. `prep_flex_receptor.py` produces correct rigid+flex PDBQT.
- So: the entire missing work is **mirroring the CPU flex machinery onto the GPU**.

## Forward-kinematics reference (already implemented for the ligand)
`quasi_newton.cl set()` (line 391) walks `rigid_cl` (lib `rigid_cl`: parent[], children_map[],
origin[], axis[], relative_axis[], relative_origin[], orientation_q/m[], atom_range[]):
- root node: origin = conf.position, orientation = conf.orientation_q → atom coords via local_to_lab.
- each child (DFS): origin/axis from parent via local_to_lab(_direction); apply torsion via
  angle_to_quaternion2 + multiply parent q; atom coords via local_to_lab.
Sidechains are the SAME structure but: anchored at a FIXED backbone point (no global position/
orientation from conf), driven only by `flex_torsion`.

## Staged plan (each stage independently verifiable)

### Stage 1 — Sidechain forward-kinematics (foundation)
- Add `rigid_cl flex_rigid;` + `int flex_num_movable_atoms;` to `m_cl` (kernel2.h). Flex atom coords
  live in `m_coords` AFTER the ligand movable atoms (indices [lig_movable, lig_movable+flex_movable)).
- CPU (`main_procedure_cl.cpp`, model-build loop ~line 861): populate `flex_rigid` from the CPU
  `flex` heterotrees (mirror how the ligand rigid tree is populated; anchor origins are the fixed
  backbone atom positions). Upload.
- Kernel: add `set_flex()` (clone of `set()` branch loop, but root anchors are fixed, torsions from
  `x->flex_torsion`, atom_range over flex atoms). Call after `set()` each step.
- **Verify:** flex_torsion=0 ⇒ flex coords == input sidechain coords (within fp eps); no NaN.

### Stage 2 — `other_pairs` energy (no gradient yet)
- Add a 2nd pair list `other_pairs_cl` (same layout as `lig_pairs_cl`: a[], b[], type_pair_index[])
  to the global `mg[]` buffer (NOT private — keep private copy small).
- CPU: remove the throw; build other_pairs (ligand↔flex + flex↔flex) from the model; upload.
- Kernel: in the energy path, call `eval_interacting_pairs_deriv` over other_pairs, accumulate into e.
- **Verify:** total energy of a known flex pose matches a CPU reference within eps.

### Stage 3 — Gradients for BFGS (the hard part)
- `eval_interacting_pairs_deriv` already returns per-atom forces (`minus_forces`). Add the flex-torsion
  derivative accumulation: for each flex torsion, sum the force projected on its rotation axis over
  the atoms it moves (mirror the ligand torsion-gradient code in `quasi_newton.cl`).
- Extend the change/conf vectors so BFGS optimizes flex torsions too (the `change_cl`/conf already
  have flex_torsion fields — wire them into the BFGS update + line search).
- **Verify:** numerical-gradient check vs analytic for flex torsions; flex docking on 1bjr/2aoe/3jzh
  recovers near-crystal (obrms < 2) where rigid failed.

### Stage 4 — Validation
- Re-run the induced-fit failure set (fail40 subset with clashes) flex vs rigid, obrms.
- Confirm no regression on the well-sampled set (must stay ~94%).
- Watch GPU memory: flex tree + other_pairs add to `m_cl`; keep `m_cl_private` (kernel2.h) free of the
  large pair arrays (access via `__global`), as already done for ligand pairs.

## Test targets (induced-fit failures, rigid best >2Å, clash at pocket center)
1bjr, 2aoe, 3jzh, 6ipi, 1q6k, 1q91, 2ajd (centers in their `validation/pdbbind/<id>/config.txt`).
`prep_flex_receptor.py --cutoff 6` to generate rigid+flex PDBQT.

## Risks / notes
- Memory: `m_cl` grows; verify `CL_MEM_OBJECT_ALLOCATION_FAILURE` doesn't recur (keep thread ≤ 8000).
- The hardest stage is 3 (flex torsion gradients); without it BFGS won't drive sidechains and flex
  adds cost without benefit. Do NOT ship stages 1–2 as "flex support" — it must include gradients.
- Keep the rigid path unchanged when `flex_torsion_size == 0` (zero overhead for normal docking).
