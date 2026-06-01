/*
 * residue_energy.h
 *
 * Per-residue energy decomposition for Vina-GPU PRO.
 *
 * After docking, computes how much each receptor residue contributes
 * to the total Vina/Vinardo interaction energy for a given pose.
 *
 * Usage (in main.cpp):
 *   auto labels = parse_receptor_residue_labels(rigid_name);
 *   auto contrib = compute_residue_energy(m, prec, labels);
 *   write_residue_energy(contrib, output_path, pose_index);
 */

#ifndef VINA_RESIDUE_ENERGY_H
#define VINA_RESIDUE_ENERGY_H

#include <string>
#include <vector>
#include <map>
#include "model.h"
#include "precalculate.h"

// One entry per receptor PDBQT atom (same order as grid_atoms / r.atoms).
// Format: "CHAIN:RESNAME:RESNUM"  e.g. "A:GLU:231"
using ResidueLabels = std::vector<std::string>;

// Cumulative energy per residue label (kcal/mol-equivalent Vina units).
using ResidueEnergy = std::map<std::string, fl>;

// Parse residue labels from receptor PDBQT (same ATOM order as parse_pdbqt_rigid).
// Returns one label per accepted ATOM/HETATM line.
ResidueLabels parse_receptor_residue_labels(const std::string& receptor_path);

// Compute per-residue contribution for a single pose already stored in m.coords.
// Uses the same precalculate table that was used during docking.
ResidueEnergy compute_residue_energy(const model& m,
                                     const precalculate& prec,
                                     const ResidueLabels& labels);

// Append per-residue table to an output stream (TSV, pose_index 1-based).
void write_residue_energy(const ResidueEnergy& contrib,
                          std::ostream& out,
                          int pose_index);

// Convenience wrapper: write to a file path, appending if it already exists.
void write_residue_energy_file(const ResidueEnergy& contrib,
                               const std::string& path,
                               int pose_index);

#endif
