#pragma once
// Shared between main_procedure_cl.cpp (single-ligand) and main_procedure_dual.cpp (dual):
// the docking-status enum, the progress-printer thread, and the GPU-result -> Vina converter.
#include <atomic>
#include <vector>
#include <thread>
#include <random>
#include "conf.h"      // output_type
#include "kernel2.h"   // output_type_cl, ligand_atom_coords_cl

// Fix: RAII guard ensures console_thread is always joined even when exceptions are thrown
struct ThreadGuard {
	std::thread& t;
	explicit ThreadGuard(std::thread& t_) : t(t_) {}
	~ThreadGuard() { if (t.joinable()) t.join(); }
	ThreadGuard(const ThreadGuard&) = delete;
	ThreadGuard& operator=(const ThreadGuard&) = delete;
};

enum class DockingStatus : int { Finish = 0, Docking = 1, Abort = 2 };
void print_process(std::atomic<int>* status_ptr);
std::vector<output_type> cl_to_vina(output_type_cl result_ptr[], ligand_atom_coords_cl result_coords_ptr[],
                                    int thread, int lig_torsion_size,
                                    const std::vector<int>& flex_tors_counts = std::vector<int>());

// Epoch-level exchange moves (single-ligand search), defined in cl_common.cpp.
void apply_re_swaps(std::vector<output_type_cl*>& result_ptrs, int num_ligands, int thread,
                    int epoch, std::mt19937& rng_re);
void apply_de_exchange(std::vector<output_type_cl*>& result_ptrs, const std::vector<int>& torsion_sizes,
                       int num_ligands, int thread, std::mt19937& rng);
