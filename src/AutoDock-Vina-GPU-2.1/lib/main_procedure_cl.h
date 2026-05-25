#pragma once
#include <atomic>
// ms: full ligand list (all GPUs share the same vector)
// global_next_lig: shared atomic work counter — each GPU grabs the next available index
// Results are written directly into out[lig_idx] with no offset needed
void main_procedure_cl(cache& c, const std::vector<model>& ms, const precalculate& p, const parallel_mc par,
	const vec& corner1, const vec& corner2, const int seed, std::vector<output_container>& out, std::string opencl_binary_path,
	const std::vector<std::vector<std::string>> ligand_names, const int rilc_bfgs,
	const int gpu_id, const int cpu_threads, std::atomic<int>& global_next_lig);
