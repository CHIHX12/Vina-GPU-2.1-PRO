#pragma once
#include "model.h"
#include "precalculate.h"
#include "non_cache.h"
#include "conf.h"    // output_type, output_container
#include "coords.h"  // add_to_output_container
#include "random.h"

// CPU-only dual-ligand co-docking Monte Carlo search.
//
// Each MC step randomly selects one ligand to mutate (50/50), applies a
// random mutation, runs independent protein-only BFGS minimization on the
// mutated ligand, then evaluates a combined score:
//   E_combined = E(protein-A) + E(protein-B) + E(A-B)
// The Metropolis criterion uses E_combined for accept/reject.
//
// This is a first-order approximation: lig-lig coupling guides global
// placement (MC acceptance) but not local geometry (protein-only BFGS).
//
// Parameters:
//   ma, mb          - protein+ligand models (protein in grid_atoms)
//   p               - precalculated Vina scoring function (shared)
//   nc_a, nc_b      - non-cache evaluators for protein-ligand A/B
//   corner1/corner2 - search box corners
//   num_steps       - MC steps per run
//   num_runs        - independent restarts (= exhaustiveness)
//   temperature     - Metropolis temperature (default 1.2 kcal/mol)
//   mutation_amplitude - max perturbation per step
//   min_rmsd        - RMSD threshold for deduplication of saved poses
//   num_saved_mins  - max poses saved per run
//   out_a, out_b    - ranked output poses for ligand A and B respectively
void dual_mc_search(
    model& ma, model& mb,
    const precalculate& p,
    const non_cache& nc_a, const non_cache& nc_b,
    const vec& corner1, const vec& corner2,
    sz num_steps, sz num_runs,
    fl temperature, fl mutation_amplitude,
    fl min_rmsd, sz num_saved_mins,
    output_container& out_a, output_container& out_b,
    rng& generator);
