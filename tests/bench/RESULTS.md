# Cavity-init validation on a clean PDBbind redocking set

`run_clean_redock.sh` prepares each ligand pdbqt FROM `<id>_ligand.sdf` (obabel) and uses that
SAME sdf as the obrms reference — so obrms's symmetry-corrected graph matching is reliable. (The
earlier `dual_test` failure was mismatched prep: input ligand and crystal reference were different
molecules, which breaks every RMSD tool. spyrmsd-via-order also fails because obabel reorders atoms.)

## Result (best-of-N obrms, baseline vs VINA_CAVITY_INIT=1)
| target | rot | baseline | cav-init |
|--------|-----|----------|----------|
| 1bra   | 1   | 0.31     | 0.31     |
| 1add   | 6   | 0.57     | 0.57     |
| 1tng   | 2   | 0.58     | 0.57     |
| 1stp   | 5   | 0.74     | 0.75     |
| 1cps   | 5   | 1.29     | 1.31     |
| 1bcd   | 2   | 1.49     | 1.50     |
| 1ppc   | 9   | 1.48     | 1.53     |

## Conclusion (honest)
- The clean-set + obrms measurement is RELIABLE (all sensible sub-2A redocks).
- cavity-init is DO-NO-HARM here (changes within +/-0.05 A = docking noise).
- BUT these are all EASY targets (baseline already redocks <1.5 A), so this set does NOT exercise
  cavity-init's purpose (rescuing LARGE/floppy ligands that baseline throws into solvent).
- To actually demonstrate the rescue benefit, the set needs HARD targets: >=12-15 rotatable bonds
  where baseline best-of-N RMSD is large. That is the next data-prep step.
