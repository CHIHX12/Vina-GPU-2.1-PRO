#include "cache.h"
#include "wrapcl.h"
#include "monte_carlo.h"
#include "coords.h"
#include "mutate.h"
#include "quasi_newton.h"
#include "parallel_mc.h"
#include "szv_grid.h"
#include <thread>
#include <atomic>
#include <future>
#include <fstream>
#include <mutex>
#include <chrono>
#include <random>
#include <cmath>
#include <boost/progress.hpp>

#include "commonMacros.h"
#include "wrapcl.h"
#include "random.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <stdio.h>
#include <map>
#include "qfd_grids.h"
#include "ls_metal.h"
#include "cl_common.h"
#include "protein_gpu.h"
#include "ligand_prep.h"
#include "ocl_setup.h"

using namespace std;

// print_process / cl_to_vina / apply_re_swaps / apply_de_exchange moved to cl_common.cpp

void main_procedure_cl(cache& c, const std::vector<model>& ms,  const precalculate& p, const parallel_mc par,
	const vec& corner1, const vec& corner2, const int seed, std::vector<output_container>& outs, std::string opencl_binary_path,
	const std::vector<std::vector<std::string>> ligand_names, const int rilc_bfgs,
	const int gpu_id, const int cpu_threads, std::atomic<int>& global_next_lig,
	const bool use_ad4zn) {

/**************************************************************************/
/***************************    OpenCL Init    ****************************/
/**************************************************************************/

	auto _fn_t0 = std::chrono::steady_clock::now();
	printf("DIAG GPU%d: main_procedure_cl started  num_ligands=%d\n", gpu_id, (int)ms.size()); fflush(stdout);

	// Replica Exchange rounds (VINA_RE_ROUNDS env var; default 1 = disabled)
	int re_rounds = 1;
	const char* re_env = std::getenv("VINA_RE_ROUNDS");
	if (re_env) re_rounds = std::max(1, std::atoi(re_env));
	if (re_rounds > 1) printf("DIAG GPU%d: Replica Exchange enabled — %d rounds\n", gpu_id, re_rounds);
	std::mt19937 rng_re(static_cast<std::mt19937::result_type>(seed ^ 0xDEADBEEF));

	cl_int err;
	printf("\nUsing random seed: %d", seed);
	OclSession ocl = ocl_setup(gpu_id, opencl_binary_path);
	cl_platform_id* platforms = ocl.platforms;
	cl_device_id*   devices   = ocl.devices;
	cl_context      context   = ocl.context;
	cl_command_queue queue    = ocl.queue;
	cl_int gpu_platform_id    = ocl.gpu_platform_id;
	cl_program* programs      = ocl.programs;
	cl_kernel*  kernels       = ocl.kernels;
	size_t max_wg_size        = ocl.max_wg_size;
	size_t* max_wi_size       = ocl.max_wi_size;

	/**************************************************************************/
	/************************    Original Vina code    ************************/
	/**************************************************************************/
	sz nat = num_atom_types(c.atu);

	// Fix: protect lazy cache init with a mutex — multiple GPU threads share the same
	// cache& c reference. Without a lock, concurrent c.grids[i].init() calls corrupt
	// the grid, and an empty `needed` makes needed.front() undefined behaviour.
	szv needed;
	{
		static std::mutex grid_init_mutex;
		std::lock_guard<std::mutex> lock(grid_init_mutex);
		for (sz i = 0; i < nat; i++) {
			if (!c.grids[i].initialized()) {
				needed.push_back(i);
				c.grids[i].init(c.gd);
			}
		}
	}
	// If another GPU thread already initialised all grids, needed is empty here.
	// Fall back to the full list so needed.front() is always valid.
	if (needed.empty()) {
		for (sz i = 0; i < nat; i++) needed.push_back(i);
	}

	flv affinities(needed.size());

	grid& g = c.grids[needed.front()];

	const fl cutoff_sqr = p.cutoff_sqr();         

	grid_dims gd_reduced = szv_grid_dims(c.gd);     
	szv_grid ig(ms[0], gd_reduced, cutoff_sqr); // use ms[0]           
	//szv_grid ig2(ms[1], gd_reduced, cutoff_sqr); // use ms[0]  

	for (int i = 0; i < 3; i++) {
		if (ig.m_init[i] != g.m_init[i]) {
			printf("m_init not equal!");
			exit(-1);
		}
		if (ig.m_range[i] != g.m_range[i]) {
			printf("m_range not equal!");
			exit(-1);
		}
	}

	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl

	std::vector<conf_size> s;
	std::vector<output_type> tmps;
	for (int i = 0; i < ms.size(); i++) {
		s.push_back(ms[i].get_size());
		output_type tmp(s[i], 0);
		tmps.push_back(tmp);
	}
	//change g(s);
	
	//quasi_newton quasi_newton_par; const int quasi_newton_par_max_steps = par.mc.ssd_par.evals;
	/**************************************************************************/
	/************************    Kernel1    ***********************/
	/**************************************************************************/
	rng generator(static_cast<rng::result_type>(seed));
	model m = ms[0];

	// LS metal rescoring: extract receptor metal positions once (shared for all ligands in VS batch)
	auto receptor_metals = collect_receptor_metals(m);
	float ls_metal_weight = 0.30f;
	{
		const char* env = std::getenv("VINA_LS_METAL_WEIGHT");
		if (env) ls_metal_weight = (float)std::atof(env);
	}
	if (!receptor_metals.empty())
		printf("QFD LS: %zu receptor metal(s) found (weight=%.3f)\n",
		       receptor_metals.size(), ls_metal_weight);

	// Fix: per-invocation status — each GPU thread has its own, so they don't
	// interfere when running concurrently (e.g., one GPU finishing doesn't stop
	// the other GPU's progress display).
	std::atomic<int> local_status{static_cast<int>(DockingStatus::Finish)};
	local_status.store(static_cast<int>(DockingStatus::Docking), std::memory_order_release);
# ifdef NDEBUG
	std::thread console_thread(print_process, &local_status);
	ThreadGuard console_guard(console_thread); // Fix: RAII ensures join on any exit path
# endif

	ProteinGpuBuffers __pg = protein_gpu_setup(c, m, p, ig, g, needed, par, context, queue,
	        kernels[0], max_wi_size, authentic_v, cutoff_sqr, use_ad4zn, gpu_id, (int)ms.size());
	cl_mem pre_gpu = __pg.pre_gpu; cl_mem grids_gpu = __pg.grids_gpu; cl_mem mis_gpu = __pg.mis_gpu;
	mis_cl* mis_ptr = __pg.mis_ptr;
/**************************************************************************/
/************************    Kernel2    ***********************/
/**************************************************************************/
	int num_ligands = ms.size();
	std::vector<random_maps*>			rand_maps_ptrs		(num_ligands, nullptr);
	std::vector<output_type_cl*>		ric_ptrs			(num_ligands, nullptr);
	std::vector<m_cl*>					m_ptrs				(num_ligands, nullptr);
	std::vector<ligand_atom_coords_cl*> result_coords_ptrs	(num_ligands, nullptr);
	std::vector<output_type_cl*>		result_ptrs			(num_ligands, nullptr);
	std::vector<int>					torsion_sizes		(num_ligands, 0);
	std::vector<output_type_cl*>		ric_for_re			(num_ligands, nullptr);  // RE: stores result for next epoch

	// Batch-dispatch state — initialized on re_epoch==0, reused across RE epochs
	int batch_n = 0;
	cl_kernel k2 = nullptr;
	cl_mem ric_gpu_k2 = nullptr, m_gpu_k2 = nullptr, rand_gpu_k2 = nullptr;
	cl_mem result_gpu_k2 = nullptr, result_coords_gpu_k2 = nullptr;
	cl_mem torsion_gpu_k2 = nullptr, search_depth_gpu_k2 = nullptr, bfgs_steps_gpu_k2 = nullptr;
	size_t kernel2_global_size[2] = { 512, 32 };
	size_t kernel2_local_size[2]  = { 16, 2 };
	double t_upload = 0, t_kernel = 0, t_download = 0, t_cl2vina = 0, t_add2out = 0;
	int lig_count = 0;
	std::vector<int> batch_torsion, batch_depth, batch_bfgs;

	for (int re_epoch = 0; re_epoch < re_rounds; re_epoch++) {

        if (re_epoch > 0) {
            // For subsequent epochs: use RE-swapped result as new ric
            for (int i = 0; i < num_ligands; i++) {
                if (ric_for_re[i]) {
                    ric_ptrs[i] = ric_for_re[i];
                    ric_for_re[i] = nullptr;
                }
            }
        }

	// Multi-CPU: pre-compute random maps and initial conformations in parallel
	// This is pure CPU work and independent per ligand — safe to parallelize
	if (re_epoch == 0) {
	printf("\nPre-computing ligand data using %d CPU thread(s)...\n", cpu_threads);
	{
		const int actual_threads = std::max(1, std::min(cpu_threads, num_ligands));
		std::vector<std::future<void>> prep_futures;
		prep_futures.reserve(actual_threads);

		// Divide ligands into chunks for each thread
		auto prep_ligand = [&](int start, int end) {
			rng local_gen(static_cast<rng::result_type>(seed + start));
			for (int i = start; i < end; i++) {
				try {
					model mi = ms[i];
					if (mi.atoms.size() >= MAX_NUM_OF_ATOMS) { throw std::runtime_error("Ligand too large! The maximum number of atoms is " +
					 std::to_string(MAX_NUM_OF_ATOMS) + " and the ligand has " + std::to_string(mi.atoms.size()) + " atoms."); }
					if (mi.ligands.size() != 1) { throw std::runtime_error("Only one ligand supported!"); }
					// Flexible-receptor docking: GPU forward-kinematics (set_flex) is implemented and
					// Flexible-receptor docking is supported on GPU (set_flex + other_pairs energy +
					// flex-torsion gradients, validated 2026-06-12). Rigid docking has num_other_pairs==0,
					// so this is a no-op there. Validated to rescue induced-fit targets (e.g. 1q6k 6.9→2.9Å).
					if (mi.num_other_pairs() != 0)
						printf("Flexible receptor: %d ligand-flex/flex-flex interaction pairs\n", (int)mi.num_other_pairs());

					output_type tmp = tmps[i];
					torsion_sizes[i] = tmp.c.ligands[0].torsions.size();
					if (tmp.c.ligands[0].torsions.size() >= MAX_NUM_OF_LIG_TORSION) { throw std::runtime_error("Ligand too large! The maximum number of torsions is " +
					 std::to_string(MAX_NUM_OF_LIG_TORSION) + " and the ligand has " + std::to_string(tmp.c.ligands[0].torsions.size()) + " torsions."); }

					// Random maps
					rand_maps_ptrs[i] = (random_maps*)malloc(sizeof(random_maps));
					random_maps* rmp = rand_maps_ptrs[i];
					for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
						rmp->int_map[k] = random_int(0, int(tmp.c.ligands[0].torsions.size()), local_gen);
						rmp->pi_map[k]  = random_fl(-pi, pi, local_gen);
					}
					for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
						vec rc = random_inside_sphere(local_gen);
						for (int j = 0; j < 3; j++) rmp->sphere_map[k][j] = rc[j];
					}

					// Initial conformations (ric)
					ric_ptrs[i] = (output_type_cl*)malloc(par.mc.thread * sizeof(output_type_cl));
					output_type_cl* rp = ric_ptrs[i];
					for (int k = 0; k < par.mc.thread; k++) {
						tmp.c.randomize(corner1, corner2, local_gen);
						for (int j = 0; j < 3; j++) rp[k].position[j] = tmp.c.ligands[0].rigid.position[j];
						rp[k].orientation[0] = tmp.c.ligands[0].rigid.orientation.R_component_1();
						rp[k].orientation[1] = tmp.c.ligands[0].rigid.orientation.R_component_2();
						rp[k].orientation[2] = tmp.c.ligands[0].rigid.orientation.R_component_3();
						rp[k].orientation[3] = tmp.c.ligands[0].rigid.orientation.R_component_4();
						for (int j = 0; j < (int)tmp.c.ligands[0].torsions.size(); j++)
							rp[k].lig_torsion[j] = tmp.c.ligands[0].torsions[j];
					}

					// m_cl struct
					m_ptrs[i] = (m_cl*)malloc(sizeof(m_cl));
					m_cl* m_ptr = m_ptrs[i];
					fill_m_cl(mi, m_ptr);
					// Funnel-density experiment: dump crystal (input) heavy coords in the SAME
					// movable-atom order the kernel's get_heavy_atom_movable_coords uses, so the
					// endpoint dump and this reference are index-aligned for valid RMSD.
					if (const char* dmp = getenv("VINA_DUMP_ENDPOINTS")) {
						char fn[1024];
						snprintf(fn, sizeof(fn), "%s_ref_lig%d.txt", dmp, i);
						FILE* rf = fopen(fn, "w");
						if (rf) {
							int nmov = (int)mi.num_movable_atoms();
							for (int ai = 0; ai < nmov; ai++) {
								if (mi.atoms[ai].el == EL_TYPE_H) continue;
								fprintf(rf, "%.3f %.3f %.3f\n",
								    mi.coords[ai].data[0], mi.coords[ai].data[1], mi.coords[ai].data[2]);
							}
							fclose(rf);
						}
					}

					// Result buffers
					result_coords_ptrs[i] = (ligand_atom_coords_cl*)malloc(par.mc.thread * sizeof(ligand_atom_coords_cl));
					result_ptrs[i]        = (output_type_cl*)malloc(par.mc.thread * sizeof(output_type_cl));

				} catch (const std::exception& ex) {
					fprintf(stderr, "CPU prep error ligand %d: %s\n", i, ex.what());
					// Free any partially-allocated buffers for this ligand
					if (rand_maps_ptrs[i])     { free(rand_maps_ptrs[i]);     rand_maps_ptrs[i] = nullptr; }
					if (ric_ptrs[i])           { free(ric_ptrs[i]);           ric_ptrs[i] = nullptr; }
					if (m_ptrs[i])             { free(m_ptrs[i]);             m_ptrs[i] = nullptr; }
					if (result_coords_ptrs[i]) { free(result_coords_ptrs[i]); result_coords_ptrs[i] = nullptr; }
					if (result_ptrs[i])        { free(result_ptrs[i]);        result_ptrs[i] = nullptr; }
				}
			}
		};

		int chunk = (num_ligands + actual_threads - 1) / actual_threads;
		for (int t = 0; t < actual_threads; t++) {
			int s = t * chunk;
			int e = std::min(s + chunk, num_ligands);
			if (s >= num_ligands) break;
			prep_futures.push_back(std::async(std::launch::async, prep_ligand, s, e));
		}
		for (auto& f : prep_futures) f.get(); // Wait for all CPU prep to finish
	}
	{
		auto _cpu_t1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: CPU pre-computation done in %.2fs\n", gpu_id,
		       std::chrono::duration<double>(_cpu_t1 - _fn_t0).count()); fflush(stdout);
	}

	// Buffer size diagnostics — helps identify memory bottlenecks
	printf("DIAG GPU%d: thread=%d  sizeof(output_type_cl)=%zu  sizeof(ligand_atom_coords_cl)=%zu  sizeof(random_maps)=%zu\n",
	       gpu_id, par.mc.thread,
	       sizeof(output_type_cl), sizeof(ligand_atom_coords_cl), sizeof(random_maps)); fflush(stdout);
	printf("DIAG GPU%d: per-ligand upload=%.1fKB  download=%.1fKB\n",
	       gpu_id,
	       (par.mc.thread * sizeof(output_type_cl) + sizeof(m_cl) + sizeof(random_maps)) / 1024.0,
	       (par.mc.thread * (sizeof(output_type_cl) + sizeof(ligand_atom_coords_cl))) / 1024.0); fflush(stdout);
	} // end if (re_epoch == 0) for CPU prep

/**************************************************************************/
/***************    Batched work-stealing kernel2 dispatch  ***************/
/**************************************************************************/
	// Fix 1: single queue eliminates multi-queue DMA serialization (upload 116s→0.08s).
	// Fix 2: work-stealing loop — both GPUs grab ligands from shared global_next_lig.
	// Fix 3: multi-ligand batching — pack batch_n ligands per kernel dispatch so all
	//         GPU work items are active (RTX 6000 Ada: batch_n=2 → ~97% utilization).

	// Auto-detect batch_n: fill the GPU's work items with multiple ligands.
	// Each ligand needs par.mc.thread work items; GPU has CU * 128 cores (NVIDIA).
	// Clamp to num_ligands so single-ligand runs still work.
	if (re_epoch == 0) {
		cl_uint compute_units = 0;
		err = clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_COMPUTE_UNITS,
		                      sizeof(cl_uint), &compute_units, NULL); checkErr(err);
		batch_n = std::max(1, (int)(compute_units * 128) / par.mc.thread);
		batch_n = std::min(batch_n, num_ligands);
		printf("DIAG GPU%d: compute_units=%u  auto batch_n=%d  (thread=%d)\n",
		       gpu_id, compute_units, batch_n, par.mc.thread); fflush(stdout);

		k2 = clCreateKernel(programs[1], "kernel2", &err); checkErr(err);

		// DIAGNOSTIC: print the runtime kernel arg count so we can detect source/binary mismatch
		{
			cl_uint k2_nargs = 0;
			clGetKernelInfo(k2, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &k2_nargs, NULL);
			printf("DIAG GPU%d: kernel2 CL_KERNEL_NUM_ARGS=%u (expect 13)\n", gpu_id, k2_nargs); fflush(stdout);
		}

		// GPU buffers sized for batch_n ligands
		const size_t ric_size_           = (size_t)batch_n * par.mc.thread * sizeof(output_type_cl);
		const size_t m_size_             = (size_t)batch_n * sizeof(m_cl);
		const size_t rand_maps_size_     = (size_t)batch_n * sizeof(random_maps);
		const size_t result_size_        = (size_t)batch_n * par.mc.thread * sizeof(output_type_cl);
		const size_t result_coords_size_ = (size_t)batch_n * par.mc.thread * sizeof(ligand_atom_coords_cl);
		const size_t int_arr_size_       = (size_t)batch_n * sizeof(int);

		CreateDeviceBuffer(&ric_gpu_k2,           CL_MEM_READ_ONLY,  ric_size_,           context);
		CreateDeviceBuffer(&m_gpu_k2,             CL_MEM_READ_WRITE, m_size_,             context);
		CreateDeviceBuffer(&rand_gpu_k2,          CL_MEM_READ_ONLY,  rand_maps_size_,     context);
		CreateDeviceBuffer(&result_gpu_k2,        CL_MEM_WRITE_ONLY, result_size_,        context);
		CreateDeviceBuffer(&result_coords_gpu_k2, CL_MEM_WRITE_ONLY, result_coords_size_, context);
		CreateDeviceBuffer(&torsion_gpu_k2,       CL_MEM_READ_ONLY,  int_arr_size_,       context);
		CreateDeviceBuffer(&search_depth_gpu_k2,  CL_MEM_READ_ONLY,  int_arr_size_,       context);
		CreateDeviceBuffer(&bfgs_steps_gpu_k2,    CL_MEM_READ_ONLY,  int_arr_size_,       context);

		// New kernel arg layout (matches updated kernel2.cl):
		//   0: ric           1: mg            2: pre (const)   3: grids (const)
		//   4: rand_maps_arr 5: coords        6: results       7: mis (const)
		//   8: torsion_sizes 9: search_depths 10: bfgs_steps   11: rilc_bfgs_enable
		//   12: batch_n  (set per-dispatch because last batch may be smaller)
		SetKernelArg(k2, 0,  sizeof(cl_mem), &ric_gpu_k2);
		SetKernelArg(k2, 1,  sizeof(cl_mem), &m_gpu_k2);
		SetKernelArg(k2, 2,  sizeof(cl_mem), &pre_gpu);
		SetKernelArg(k2, 3,  sizeof(cl_mem), &grids_gpu);
		SetKernelArg(k2, 4,  sizeof(cl_mem), &rand_gpu_k2);
		SetKernelArg(k2, 5,  sizeof(cl_mem), &result_coords_gpu_k2);
		SetKernelArg(k2, 6,  sizeof(cl_mem), &result_gpu_k2);
		SetKernelArg(k2, 7,  sizeof(cl_mem), &mis_gpu);
		SetKernelArg(k2, 8,  sizeof(cl_mem), &torsion_gpu_k2);
		SetKernelArg(k2, 9,  sizeof(cl_mem), &search_depth_gpu_k2);
		SetKernelArg(k2, 10, sizeof(cl_mem), &bfgs_steps_gpu_k2);
		SetKernelArg(k2, 11, sizeof(int),    &rilc_bfgs);
		// arg 12 (batch_n) set per-dispatch below

		batch_torsion.assign(batch_n, 0);
		batch_depth.assign(batch_n, 0);
		batch_bfgs.assign(batch_n, 0);

		printf("\nBatched work-stealing kernel2: GPU%d  total_ligands=%d  batch_n=%d\n",
		       gpu_id, num_ligands, batch_n); fflush(stdout);
	} // end if (re_epoch == 0) for batch setup

	// RE epoch > 0: each GPU iterates its own ligands (those with non-null ric_ptrs)
	// using a per-thread local counter (global_next_lig is exhausted after epoch 0).
	std::atomic<int> gpu_local_next{0};
	int batch_start;
	while ((batch_start = (re_epoch == 0
	                       ? global_next_lig.fetch_add(batch_n, std::memory_order_relaxed)
	                       : gpu_local_next  .fetch_add(batch_n, std::memory_order_relaxed))) < num_ligands) {
		int actual_batch = std::min(batch_n, num_ligands - batch_start);

		// Verify CPU prep succeeded for all ligands in this batch
		bool all_ready = true;
		for (int b = 0; b < actual_batch; b++) {
			int li = batch_start + b;
			if (!rand_maps_ptrs[li] || !ric_ptrs[li] || !m_ptrs[li] ||
			    !result_ptrs[li] || !result_coords_ptrs[li]) {
				std::cerr << "Skipping batch starting at " << batch_start
				          << ": CPU prep failed for ligand " << li << ".\n";
				all_ready = false;
				break;
			}
		}
		if (!all_ready) continue;

		auto _t_up0 = std::chrono::steady_clock::now();

		// Upload actual_batch ligands at their stride offsets
		for (int b = 0; b < actual_batch; b++) {
			int li = batch_start + b;
			size_t ric_off  = (size_t)b * par.mc.thread * sizeof(output_type_cl);
			size_t m_off    = (size_t)b * sizeof(m_cl);
			size_t rand_off = (size_t)b * sizeof(random_maps);
			err = clEnqueueWriteBuffer(queue, ric_gpu_k2,  CL_FALSE, ric_off,  par.mc.thread * sizeof(output_type_cl), ric_ptrs[li],       0, NULL, NULL); checkErr(err);
			err = clEnqueueWriteBuffer(queue, m_gpu_k2,    CL_FALSE, m_off,    sizeof(m_cl),                           m_ptrs[li],         0, NULL, NULL); checkErr(err);
			err = clEnqueueWriteBuffer(queue, rand_gpu_k2, CL_FALSE, rand_off, sizeof(random_maps),                    rand_maps_ptrs[li], 0, NULL, NULL); checkErr(err);

			batch_torsion[b] = torsion_sizes[li];
			batch_depth[b]   = par.mc.search_depth[li];
			batch_bfgs[b]    = par.mc.ssd_par.bfgs_steps[li];
		}

		// Upload per-ligand scalar arrays
		err = clEnqueueWriteBuffer(queue, torsion_gpu_k2,      CL_FALSE, 0, (size_t)actual_batch * sizeof(int), batch_torsion.data(), 0, NULL, NULL); checkErr(err);
		err = clEnqueueWriteBuffer(queue, search_depth_gpu_k2, CL_FALSE, 0, (size_t)actual_batch * sizeof(int), batch_depth.data(),   0, NULL, NULL); checkErr(err);
		err = clEnqueueWriteBuffer(queue, bfgs_steps_gpu_k2,   CL_FALSE, 0, (size_t)actual_batch * sizeof(int), batch_bfgs.data(),    0, NULL, NULL); checkErr(err);

		// clFinish ensures all non-blocking writes are complete before freeing CPU buffers.
		// OpenCL spec: host_ptr must not be freed until the transfer command completes.
		clFinish(queue);

		// Safe to free CPU upload buffers now that DMA transfers are complete
		for (int b = 0; b < actual_batch; b++) {
			int li = batch_start + b;
			free(rand_maps_ptrs[li]); rand_maps_ptrs[li] = nullptr;
			free(ric_ptrs[li]);       ric_ptrs[li] = nullptr;
			// m_ptrs hold atom topology (static across RE epochs) — keep alive until last epoch
			if (re_epoch == re_rounds - 1) {
				free(m_ptrs[li]); m_ptrs[li] = nullptr;
			}
		}
		auto _t_kn0 = std::chrono::steady_clock::now();
		t_upload += std::chrono::duration<double>(_t_kn0 - _t_up0).count();

		// Set actual_batch as kernel arg (handles last batch smaller than batch_n)
		SetKernelArg(k2, 12, sizeof(int), &actual_batch);

		// Kernel launch + wait
		cl_event done_event;
		err = clEnqueueNDRangeKernel(queue, k2, 2, 0,
		    kernel2_global_size, kernel2_local_size,
		    0, NULL, &done_event); checkErr(err);
		clFlush(queue);
		clWaitForEvents(1, &done_event);
		clReleaseEvent(done_event);
		auto _t_dn0 = std::chrono::steady_clock::now();
		t_kernel += std::chrono::duration<double>(_t_dn0 - _t_kn0).count();

		// Download actual_batch results (non-blocking per ligand, then sync)
		for (int b = 0; b < actual_batch; b++) {
			int li = batch_start + b;
			size_t res_off   = (size_t)b * par.mc.thread * sizeof(output_type_cl);
			size_t coord_off = (size_t)b * par.mc.thread * sizeof(ligand_atom_coords_cl);
			err = clEnqueueReadBuffer(queue, result_gpu_k2,        CL_FALSE, res_off,   par.mc.thread * sizeof(output_type_cl),        result_ptrs[li],        0, NULL, NULL); checkErr(err);
			err = clEnqueueReadBuffer(queue, result_coords_gpu_k2, CL_FALSE, coord_off, par.mc.thread * sizeof(ligand_atom_coords_cl), result_coords_ptrs[li], 0, NULL, NULL); checkErr(err);
		}
		clFinish(queue);
		auto _t_cv0 = std::chrono::steady_clock::now();
		t_download += std::chrono::duration<double>(_t_cv0 - _t_dn0).count();

		// Convert and store results for each ligand in the batch
		for (int b = 0; b < actual_batch; b++) {
			int li = batch_start + b;

			auto _t_a0 = std::chrono::steady_clock::now();
			// Stage 1.5: per-residue flex torsion counts so cl_to_vina rebuilds conf.flex correctly.
			std::vector<int> flex_counts_li;
			{
				conf ic_li = ms[li].get_initial_conf();
				for (size_t r = 0; r < ic_li.flex.size(); r++)
					flex_counts_li.push_back((int)ic_li.flex[r].torsions.size());
			}
			std::vector<output_type> result_vina = cl_to_vina(
			    result_ptrs[li], result_coords_ptrs[li],
			    par.mc.thread, torsion_sizes[li], flex_counts_li);
			// LS metal rescoring: adjust .e before add_to_output_container sorts
			apply_ls_metal_scores(result_vina, receptor_metals, ls_metal_weight);
			auto _t_a1 = std::chrono::steady_clock::now();
			t_cl2vina += std::chrono::duration<double>(_t_a1 - _t_a0).count();

			if (re_epoch == re_rounds - 1) {
				// --- Funnel-density experiment: dump ALL trajectory endpoints ---
				// Gated by VINA_DUMP_ENDPOINTS=<prefix>. One line/pose: e then x y z per heavy atom.
				// Lets us test offline whether the densest basin (most-converged cluster)
				// predicts the crystal pose better than the lowest-energy pose.
				if (const char* dmp = getenv("VINA_DUMP_ENDPOINTS")) {
					char fn[1024];
					snprintf(fn, sizeof(fn), "%s_lig%d.txt", dmp, li);
					FILE* f = fopen(fn, "w");
					if (f) {
						for (int i = 0; i < par.mc.thread; i++) {
							fprintf(f, "%.4f", result_vina[i].e);
							for (const auto& cc : result_vina[i].coords)
								fprintf(f, " %.3f %.3f %.3f", cc[0], cc[1], cc[2]);
							fprintf(f, "\n");
						}
						fclose(f);
						printf("DUMP: wrote %d endpoints to %s\n", par.mc.thread, fn);
					}
				}
				// Final epoch: commit to output
				if (result_vina.empty()) {
					std::cerr << "Warning: no results for ligand " << li << " — skipping.\n";
				} else {
					for (int i = 0; i < par.mc.thread; i++) {
						add_to_output_container(outs[li], result_vina[i],
						    par.mc.min_rmsd, par.mc.num_saved_mins);
					}
				}
				free(result_ptrs[li]);        result_ptrs[li] = nullptr;
			} else {
				// Intermediate RE epoch: transfer result_ptrs ownership to ric_for_re
				// for RE swap; result_coords not needed beyond this epoch
				ric_for_re[li] = result_ptrs[li];
				result_ptrs[li] = nullptr;
			}
			auto _t_a2 = std::chrono::steady_clock::now();
			t_add2out += std::chrono::duration<double>(_t_a2 - _t_a1).count();

			free(result_coords_ptrs[li]); result_coords_ptrs[li] = nullptr;
		}
		lig_count += actual_batch;
	}

	printf("DIAG GPU%d: RE epoch %d done — %d ligands  upload=%.2fs  kernel=%.2fs  dl=%.2fs  conv=%.2fs\n",
	       gpu_id, re_epoch, lig_count, t_upload, t_kernel, t_download, t_cl2vina); fflush(stdout);

	// RE post-epoch: apply Metropolis swaps, regenerate rand_maps + result buffers
	if (re_epoch < re_rounds - 1) {
		apply_re_swaps(ric_for_re, num_ligands, par.mc.thread, re_epoch, rng_re);
		// Algorithm A: differential-evolution inter-trajectory exchange (gated by VINA_DE)
		if (std::getenv("VINA_DE")) {
			static std::mt19937 de_rng(static_cast<unsigned>(seed + 90001));
			apply_de_exchange(ric_for_re, torsion_sizes, num_ligands, par.mc.thread, de_rng);
			if (re_epoch == 0) printf("DIAG GPU%d: DE inter-trajectory exchange enabled\n", gpu_id);
		}

		rng local_re_gen(static_cast<rng::result_type>(seed + (re_epoch + 1) * 31337));
		for (int i = 0; i < num_ligands; i++) {
			if (!ric_for_re[i]) continue;  // not this GPU's ligand
			// Fresh random maps for diversity in next epoch
			rand_maps_ptrs[i] = (random_maps*)malloc(sizeof(random_maps));
			random_maps* rmp = rand_maps_ptrs[i];
			for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
				rmp->int_map[k] = random_int(0, std::max(1, (int)torsion_sizes[i]), local_re_gen);
				rmp->pi_map[k]  = random_fl(-pi, pi, local_re_gen);
			}
			for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
				vec rc = random_inside_sphere(local_re_gen);
				for (int j = 0; j < 3; j++) rmp->sphere_map[k][j] = rc[j];
			}
			// Fresh result buffers
			result_coords_ptrs[i] = (ligand_atom_coords_cl*)malloc(par.mc.thread * sizeof(ligand_atom_coords_cl));
			result_ptrs[i]        = (output_type_cl*)malloc(par.mc.thread * sizeof(output_type_cl));
		}
		gpu_local_next.store(0, std::memory_order_relaxed);
		t_upload = 0; t_kernel = 0; t_download = 0; t_cl2vina = 0; t_add2out = 0;
		lig_count = 0;
	}

	} // end for (re_epoch)

	// Free any CPU buffers for ligands that were stolen by the other GPU (or leaked).
	for (int i = 0; i < num_ligands; i++) {
		if (rand_maps_ptrs[i])     { free(rand_maps_ptrs[i]);     rand_maps_ptrs[i] = nullptr; }
		if (ric_ptrs[i])           { free(ric_ptrs[i]);           ric_ptrs[i] = nullptr; }
		if (m_ptrs[i])             { free(m_ptrs[i]);             m_ptrs[i] = nullptr; }
		if (result_ptrs[i])        { free(result_ptrs[i]);        result_ptrs[i] = nullptr; }
		if (result_coords_ptrs[i]) { free(result_coords_ptrs[i]); result_coords_ptrs[i] = nullptr; }
		if (ric_for_re[i])         { free(ric_for_re[i]);         ric_for_re[i] = nullptr; }
	}

	// Cleanup batched-dispatch resources
	err = clReleaseKernel(k2);                        checkErr(err);
	err = clReleaseMemObject(ric_gpu_k2);             checkErr(err);
	err = clReleaseMemObject(m_gpu_k2);               checkErr(err);
	err = clReleaseMemObject(rand_gpu_k2);            checkErr(err);
	err = clReleaseMemObject(result_gpu_k2);          checkErr(err);
	err = clReleaseMemObject(result_coords_gpu_k2);   checkErr(err);
	err = clReleaseMemObject(torsion_gpu_k2);         checkErr(err);
	err = clReleaseMemObject(search_depth_gpu_k2);    checkErr(err);
	err = clReleaseMemObject(bfgs_steps_gpu_k2);      checkErr(err);


	free(mis_ptr);
	err = clReleaseMemObject(mis_gpu);		checkErr(err);
	err = clReleaseMemObject(grids_gpu);	checkErr(err);
	err = clReleaseMemObject(pre_gpu);		checkErr(err);

	local_status.store(static_cast<int>(DockingStatus::Finish), std::memory_order_release);
	// Fix: ThreadGuard destructor joins console_thread automatically here
	{
		auto _fn_t1 = std::chrono::steady_clock::now();
		double _fn_s = std::chrono::duration<double>(_fn_t1 - _fn_t0).count();
		printf("DIAG GPU%d: main_procedure_cl done in %.2fs  (%d ligands = %.3f mol/s)\n",
		       gpu_id, _fn_s, (int)ms.size(), ms.size() / _fn_s); fflush(stdout);
	}
#ifdef OPENCL_TIME_ANALYSIS
		cout << setiosflags(ios::fixed);
		std::ofstream file("gpu_runtime.log");
		if (file.is_open())
		{
			file << "GPU grid cache runtime = " << (kernel1_total_time / 1000000000.0) << " s" << std::endl;
			file << "GPU monte carlo runtime = " << (kernel2_total_time / 1000000000.0) << " s" << std::endl;
			file.close();
		}
#endif
}

/*****************************************************************************
 * GPU dual-ligand co-docking: kernel2_dual
 * Docks two ligands simultaneously using a combined E(A-prot)+E(B-prot)+E(A-B)
 * objective. Uses the same protein affinity grid as single-ligand mode.
 *****************************************************************************/

// main_procedure_cl_dual moved to main_procedure_dual.cpp (see cl_common.h for shared decls)
