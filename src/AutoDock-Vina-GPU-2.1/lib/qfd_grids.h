#pragma once
// QFD field grids (pipi & cavity buildable directly from the receptor). Extracted from
// main_procedure_cl.cpp for clarity.
#include <string>
#include "model.h"
#include "kernel2.h"   // grid_cl + MAX_NUM_OF_GRID_* + GRID_IDX_*

// Load a precomputed QFD grid binary into `slot`. Returns false (slot untouched) if the file is absent.
bool load_qfd_grid_file(const std::string& path, grid_cl* slot);
// Build the pi-pi aromatic-proximity grid from receptor aromatic carbons (ref_grid = box reference).
bool compute_pipi_grid_from_receptor(const model& m, const grid& ref_grid, grid_cl* slot);
// Build the cavity CONTAINMENT (exposure) grid via buriedness ray-casting (0=buried pocket, 1=solvent).
bool compute_cavity_grid_from_receptor(const model& m, const grid& ref_grid, grid_cl* slot);
