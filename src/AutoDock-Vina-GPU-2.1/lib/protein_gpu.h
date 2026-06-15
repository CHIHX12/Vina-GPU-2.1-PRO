#pragma once
// Protein/grid GPU setup, extracted from main_procedure_cl.cpp: build host-side protein/precalc/grid
// structures, create+upload device buffers, run kernel1 (atom-type grids), free intermediates.
// Returns only the buffers kernel2 needs. (Vina headers BEFORE commonMacros.h — matches the include
// order in main_procedure_cl.cpp; kernel2.h mis-parses if a vina macro isn't defined first.)
#include "cache.h"
#include "precalculate.h"
#include "szv_grid.h"
#include "parallel_mc.h"
#include "commonMacros.h"   // cl_* types + kernel2.h (grids_cl, mis_cl, ...)

struct ProteinGpuBuffers { cl_mem pre_gpu, grids_gpu, mis_gpu; mis_cl* mis_ptr; };

ProteinGpuBuffers protein_gpu_setup(
    cache& c, const model& m, const precalculate& p, szv_grid& ig, grid& g,
    const szv& needed, const parallel_mc& par, cl_context context, cl_command_queue queue,
    cl_kernel kernel1_k, const size_t* max_wi_size, const vec& authentic_v, fl cutoff_sqr,
    bool use_ad4zn, int gpu_id, int num_ligands);
