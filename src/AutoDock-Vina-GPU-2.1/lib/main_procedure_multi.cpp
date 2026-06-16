// Native multi-ligand co-docking host launch (P1).
// Mirrors main_procedure_dual.cpp but drives the kernel2_multi path: ONE model holding N
// ligands (built host-side via parse_bundle()/model::append()), jointly optimised on the GPU.
// Lean standard-Vina scoring (no QFD), to validate against CPU AutoDock Vina 1.2.7.
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
#include <chrono>
#include <random>
#include <cmath>
#include <vector>

#include "commonMacros.h"
#include "random.h"
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <cstdlib>
#include "qfd_grids.h"
#include "ls_metal.h"
#include "cl_common.h"
#include "protein_gpu.h"
#include "ligand_prep.h"
#include "ocl_setup.h"
#include <cstddef>   // offsetof — host/device layout probe

using namespace std;

// out receives the best joint pose(s); the caller writes all N ligands via write_all_output
// (m.set(out[0].c) updates every ligand's coords, then the whole model is written to one file).
void main_procedure_multi(cache& c, model& m, const precalculate& p, const parallel_mc par,
	const vec& corner1, const vec& corner2, const int seed, output_container& out,
	std::string opencl_binary_path, const int rilc_bfgs, const int gpu_id, const bool use_ad4zn) {

	auto _fn_t0 = std::chrono::steady_clock::now();
	const int N = (int)m.ligands.size();
	printf("DIAG GPU%d: main_procedure_multi started  num_ligands=%d\n", gpu_id, N); fflush(stdout);
	if (N < 1) throw std::runtime_error("multi: model has no ligands");
	if (N > MAX_NUM_OF_LIGANDS)
		throw std::runtime_error("multi: " + std::to_string(N) + " ligands exceeds MAX_NUM_OF_LIGANDS=" + std::to_string(MAX_NUM_OF_LIGANDS));

	cl_int err;
	OclSession ocl = ocl_setup(gpu_id, opencl_binary_path);
	cl_context       context  = ocl.context;
	cl_command_queue queue    = ocl.queue;
	cl_device_id*    devices  = ocl.devices;
	cl_program*      programs = ocl.programs;
	cl_kernel*       kernels  = ocl.kernels;
	size_t*          max_wi_size = ocl.max_wi_size;

	// ───── DEBUG: host vs device struct-layout probe (cap-bug hunt). VINA_MULTI_PROBE=1 ─────
	if (getenv("VINA_MULTI_PROBE")) {
		cl_kernel probe = clCreateKernel(programs[1], "multi_layout_probe", &err); checkErr(err);
		cl_mem buf; CreateDeviceBuffer(&buf, CL_MEM_WRITE_ONLY, 16 * sizeof(cl_ulong), context);
		SetKernelArg(probe, 0, sizeof(cl_mem), &buf);
		size_t gw1 = 1, lw1 = 1;
		err = clEnqueueNDRangeKernel(queue, probe, 1, 0, &gw1, &lw1, 0, NULL, NULL); checkErr(err);
		clFinish(queue);
		cl_ulong d[16];
		err = clEnqueueReadBuffer(queue, buf, true, 0, 16 * sizeof(cl_ulong), d, 0, NULL, NULL); checkErr(err);
		auto cmp = [](const char* name, size_t h, cl_ulong dev) {
			printf("  %-32s host=%-8zu dev=%-8lu %s\n", name, h, (unsigned long)dev,
			       (h == (size_t)dev) ? "OK" : "<<<<< MISMATCH");
		};
		printf("\n=== MULTI LAYOUT PROBE (MAX_NUM_OF_LIGANDS=%d) ===\n", (int)d[15]);
		cmp("sizeof(output_type_multi_cl)", sizeof(output_type_multi_cl), d[0]);
		cmp("sizeof(change_multi_cl)",      sizeof(change_multi_cl),      d[1]);
		cmp("sizeof(m_multi_cl_private)",   sizeof(m_multi_cl_private),   d[2]);
		cmp("sizeof(m_multi_cl)",           sizeof(m_multi_cl),           d[3]);
		cmp("sizeof(ligand_multi_cl)",      sizeof(ligand_multi_cl),      d[4]);
		cmp("sizeof(rigid_cl)",             sizeof(rigid_cl),             d[5]);
		cmp("sizeof(m_coords_multi_cl)",    sizeof(m_coords_multi_cl),    d[6]);
		cmp("sizeof(lig_pairs_multi_cl)",   sizeof(lig_pairs_multi_cl),   d[7]);
		cmp("off output.orientation",       offsetof(output_type_multi_cl, orientation), d[8]);
		cmp("off output.lig_torsion",       offsetof(output_type_multi_cl, lig_torsion), d[9]);
		cmp("off change.orientation",       offsetof(change_multi_cl, orientation), d[10]);
		cmp("off change.lig_torsion",       offsetof(change_multi_cl, lig_torsion), d[11]);
		cmp("off m.ligands[0]",             offsetof(m_multi_cl_private, ligands), d[12]);
		printf("  %-32s host=%-8zu dev=%-8lu %s\n", "ligand stride (ligands[1]-[0])",
		       sizeof(ligand_multi_cl), (unsigned long)(d[13] - d[12]),
		       (sizeof(ligand_multi_cl) == (size_t)(d[13] - d[12])) ? "OK" : "<<<<< MISMATCH");
		cmp("off m.m_coords",               offsetof(m_multi_cl_private, m_coords), d[14]);
		printf("=== probe done (no docking performed) ===\n"); fflush(stdout);
		clReleaseMemObject(buf); clReleaseKernel(probe);
		return;
	}

	/**** Vina grid setup (reused from single/dual path) ****/
	sz nat = num_atom_types(c.atu);
	szv needed;
	{
		static std::mutex grid_init_mutex;
		std::lock_guard<std::mutex> lock(grid_init_mutex);
		for (sz i = 0; i < nat; i++)
			if (!c.grids[i].initialized()) { needed.push_back(i); c.grids[i].init(c.gd); }
	}
	if (needed.empty()) for (sz i = 0; i < nat; i++) needed.push_back(i);

	flv affinities(needed.size());
	grid& g = c.grids[needed.front()];
	const fl cutoff_sqr = p.cutoff_sqr();
	grid_dims gd_reduced = szv_grid_dims(c.gd);
	szv_grid ig(m, gd_reduced, cutoff_sqr);
	for (int i = 0; i < 3; i++) {
		if (ig.m_init[i]  != g.m_init[i])  { printf("m_init not equal!");  exit(-1); }
		if (ig.m_range[i] != g.m_range[i]) { printf("m_range not equal!"); exit(-1); }
	}
	vec authentic_v(1000, 1000, 1000);

	std::atomic<int> local_status{static_cast<int>(DockingStatus::Docking)};
#ifdef NDEBUG
	std::thread console_thread(print_process, &local_status);
	ThreadGuard console_guard(console_thread);
#endif

	ProteinGpuBuffers __pg = protein_gpu_setup(c, m, p, ig, g, needed, par, context, queue,
	        kernels[0], max_wi_size, authentic_v, cutoff_sqr, use_ad4zn, gpu_id, 1);
	cl_mem pre_gpu = __pg.pre_gpu, grids_gpu = __pg.grids_gpu, mis_gpu = __pg.mis_gpu;
	mis_cl* mis_ptr = __pg.mis_ptr;

	/**** Trajectory / work-item counts (multi state is large → fewer than single) ****/
	int n_traj = par.mc.thread;
	if (const char* e = getenv("VINA_MULTI_TRAJ")) { int v = atoi(e); if (v >= 1) n_traj = v; }
	size_t multi_wi = 512;
	if (const char* e = getenv("VINA_MULTI_WI")) { int v = atoi(e); if (v >= 128) multi_wi = ((v + 127) / 128) * 128; }
	int search_depth = par.mc.search_depth.empty() ? 64 : par.mc.search_depth[0];
	int bfgs_steps   = par.mc.ssd_par.bfgs_steps.empty() ? 300 : par.mc.ssd_par.bfgs_steps[0];

	/**** Per-ligand torsion partition (must match fill_m_cl_multi ordering) ****/
	conf_size s = m.get_size();
	std::vector<int> tors_off(N, 0);
	int total_torsions = 0, total_choices = 0;
	for (int k = 0; k < N; k++) {
		tors_off[k] = total_torsions;
		total_torsions += (int)s.ligands[k];
		total_choices  += 2 + (int)s.ligands[k];   // mutate: 1 translate + 1 rotate + ntors
	}
	if (total_torsions > MAX_NUM_OF_LIG_TORSION)
		throw std::runtime_error("multi: total torsions " + std::to_string(total_torsions) +
			" exceeds MAX_NUM_OF_LIG_TORSION=" + std::to_string(MAX_NUM_OF_LIG_TORSION));
	int n_dof = 6 * N + total_torsions;
	if (n_dof + 2 > MAX_MULTI_CONF_DIM)
		throw std::runtime_error("multi: DOF " + std::to_string(n_dof) + " exceeds MAX_MULTI_CONF_DIM");

	printf("DIAG GPU%d: N=%d  total_torsions=%d  DOF=%d  n_traj=%d  work_items=%zu  sd=%d  bfgs=%d\n",
	       gpu_id, N, total_torsions, n_dof, n_traj, multi_wi, search_depth, bfgs_steps); fflush(stdout);

	/**** CPU prep: model buffer, random maps, initial confs ****/
	rng generator(static_cast<rng::result_type>(seed));

	m_multi_cl* mg = (m_multi_cl*)malloc(sizeof(m_multi_cl));
	fill_m_cl_multi(m, mg);

	random_maps* rmap = (random_maps*)malloc(sizeof(random_maps));
	for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
		rmap->int_map[k] = random_int(0, std::max(1, total_choices), generator);
		rmap->pi_map[k]  = random_fl(-pi, pi, generator);
	}
	for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
		vec rc = random_inside_sphere(generator);
		for (int j = 0; j < 3; j++) rmap->sphere_map[k][j] = rc[j];
	}

	// DIAGNOSTIC: VINA_MULTI_CRYSTAL=1 starts every trajectory from the INPUT (crystal) pose —
	// each ligand's parsed root frame + torsions=0 — a non-clashing pose so BFGS actually
	// optimises (vs the random box pose, which clashes and makes BFGS a no-op).
	const bool use_crystal = (getenv("VINA_MULTI_CRYSTAL") != nullptr);
	output_type tmp(s, 0);  // conf with N ligand_confs
	output_type_multi_cl* ric = (output_type_multi_cl*)malloc((size_t)n_traj * sizeof(output_type_multi_cl));
	for (int t = 0; t < n_traj; t++) {
		if (!use_crystal) tmp.c.randomize(corner1, corner2, generator);
		ric[t].num_ligands = N;
		for (int k = 0; k < N; k++) {
			if (use_crystal) {
				vec o = m.ligands[k].node.get_origin();
				for (int j = 0; j < 3; j++) ric[t].position[k][j] = (float)o[j];
				ric[t].orientation[k][0] = m.ligands[k].node.orientation().R_component_1();
				ric[t].orientation[k][1] = m.ligands[k].node.orientation().R_component_2();
				ric[t].orientation[k][2] = m.ligands[k].node.orientation().R_component_3();
				ric[t].orientation[k][3] = m.ligands[k].node.orientation().R_component_4();
				for (int j = 0; j < (int)s.ligands[k]; j++) ric[t].lig_torsion[tors_off[k] + j] = 0.0f;
			} else {
				for (int j = 0; j < 3; j++) ric[t].position[k][j] = tmp.c.ligands[k].rigid.position[j];
				ric[t].orientation[k][0] = tmp.c.ligands[k].rigid.orientation.R_component_1();
				ric[t].orientation[k][1] = tmp.c.ligands[k].rigid.orientation.R_component_2();
				ric[t].orientation[k][2] = tmp.c.ligands[k].rigid.orientation.R_component_3();
				ric[t].orientation[k][3] = tmp.c.ligands[k].rigid.orientation.R_component_4();
				for (int j = 0; j < (int)s.ligands[k]; j++)
					ric[t].lig_torsion[tors_off[k] + j] = tmp.c.ligands[k].torsions[j];
			}
		}
	}

	output_type_multi_cl* results    = (output_type_multi_cl*)malloc((size_t)n_traj * sizeof(output_type_multi_cl));
	m_coords_multi_cl*    out_coords = (m_coords_multi_cl*)   malloc((size_t)n_traj * sizeof(m_coords_multi_cl));

	/**** GPU buffers ****/
	const size_t ric_sz    = (size_t)n_traj * sizeof(output_type_multi_cl);
	const size_t res_sz    = ric_sz;
	const size_t coords_sz = (size_t)n_traj * sizeof(m_coords_multi_cl);
	cl_mem ric_gpu, mg_gpu, rand_gpu, results_gpu, coords_gpu;
	CreateDeviceBuffer(&ric_gpu,     CL_MEM_READ_ONLY,  ric_sz,            context);
	CreateDeviceBuffer(&mg_gpu,      CL_MEM_READ_WRITE, sizeof(m_multi_cl),context);
	CreateDeviceBuffer(&rand_gpu,    CL_MEM_READ_ONLY,  sizeof(random_maps),context);
	CreateDeviceBuffer(&results_gpu, CL_MEM_WRITE_ONLY, res_sz,            context);
	CreateDeviceBuffer(&coords_gpu,  CL_MEM_WRITE_ONLY, coords_sz,         context);

	err = clEnqueueWriteBuffer(queue, ric_gpu,  false, 0, ric_sz,             ric,  0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, mg_gpu,   false, 0, sizeof(m_multi_cl), mg,   0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, rand_gpu, false, 0, sizeof(random_maps),rmap, 0, NULL, NULL); checkErr(err);
	clFinish(queue);
	free(ric); free(mg); free(rmap);

	// ───── DEBUG: score ric[0] once (no MC/BFGS) for cap-bug bisection. VINA_MULTI_SCORE1=1 ─────
	// Run with a FIXED --seed at N=8 and N=16; if E or per-ligand gradnorm differ, the multi
	// eval diverges with the cap (and gradnorm_ligA vs ligB shows which ligand is mis-evaluated).
	if (getenv("VINA_MULTI_SCORE1")) {
		cl_kernel sc = clCreateKernel(programs[1], "kernel2_multi_score1", &err); checkErr(err);
		cl_mem ebuf; CreateDeviceBuffer(&ebuf, CL_MEM_WRITE_ONLY, 3 * sizeof(float), context);
		SetKernelArg(sc, 0, sizeof(cl_mem), &ric_gpu);   // conf_in = ric[0]
		SetKernelArg(sc, 1, sizeof(cl_mem), &mg_gpu);
		SetKernelArg(sc, 2, sizeof(cl_mem), &pre_gpu);
		SetKernelArg(sc, 3, sizeof(cl_mem), &grids_gpu);
		SetKernelArg(sc, 4, sizeof(cl_mem), &mis_gpu);
		SetKernelArg(sc, 5, sizeof(int),    &N);
		SetKernelArg(sc, 6, sizeof(int),    &total_torsions);
		SetKernelArg(sc, 7, sizeof(cl_mem), &ebuf);
		size_t g1 = 1, l1 = 1;
		err = clEnqueueNDRangeKernel(queue, sc, 1, 0, &g1, &l1, 0, NULL, NULL); checkErr(err);
		clFinish(queue);
		float ev[3];
		err = clEnqueueReadBuffer(queue, ebuf, true, 0, 3 * sizeof(float), ev, 0, NULL, NULL); checkErr(err);
		printf("\n=== SCORE1 (MAX_NUM_OF_LIGANDS=%d) ===\n", MAX_NUM_OF_LIGANDS);
		printf("  E(ric[0]) = %.6f   gradnorm_ligA = %.6f   gradnorm_ligB = %.6f\n", ev[0], ev[1], ev[2]);
		printf("=== score1 done (no docking) ===\n"); fflush(stdout);
		clReleaseMemObject(ebuf); clReleaseKernel(sc);
		return;
	}

	// ───── DEBUG: run bfgs_multi ONCE on ric[0] (isolates the optimizer). VINA_MULTI_BFGS1=1 ─────
	if (getenv("VINA_MULTI_BFGS1")) {
		cl_kernel bk = clCreateKernel(programs[1], "kernel2_multi_bfgs1", &err); checkErr(err);
		cl_mem obuf; CreateDeviceBuffer(&obuf, CL_MEM_WRITE_ONLY, 8 * sizeof(float), context);
		SetKernelArg(bk, 0, sizeof(cl_mem), &ric_gpu);
		SetKernelArg(bk, 1, sizeof(cl_mem), &mg_gpu);
		SetKernelArg(bk, 2, sizeof(cl_mem), &pre_gpu);
		SetKernelArg(bk, 3, sizeof(cl_mem), &grids_gpu);
		SetKernelArg(bk, 4, sizeof(cl_mem), &mis_gpu);
		SetKernelArg(bk, 5, sizeof(int),    &N);
		SetKernelArg(bk, 6, sizeof(int),    &total_torsions);
		SetKernelArg(bk, 7, sizeof(int),    &bfgs_steps);
		SetKernelArg(bk, 8, sizeof(cl_mem), &obuf);
		cl_ulong bpriv = 0;
		clGetKernelWorkGroupInfo(bk, devices[gpu_id], CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &bpriv, NULL);
		printf("DIAG: bfgs1 private_mem=%llu bytes/work-item\n", (unsigned long long)bpriv); fflush(stdout);
		size_t g1 = 64, l1 = 64;   // match docking work-group (local=1 over-allocates local-mem backing)
		err = clEnqueueNDRangeKernel(queue, bk, 1, 0, &g1, &l1, 0, NULL, NULL); checkErr(err);
		clFinish(queue);
		float ov[8];
		err = clEnqueueReadBuffer(queue, obuf, true, 0, 8 * sizeof(float), ov, 0, NULL, NULL); checkErr(err);
		printf("\n=== BFGS1 (MAX_NUM_OF_LIGANDS=%d, bfgs_steps=%d) ===\n", MAX_NUM_OF_LIGANDS, bfgs_steps);
		printf("  post-BFGS E = %.6f\n", ov[0]);
		printf("  ligA root pos = (%.5f, %.5f, %.5f)\n", ov[1], ov[2], ov[3]);
		printf("  ligB root pos = (%.5f, %.5f, %.5f)\n", ov[4], ov[5], ov[6]);
		printf("=== bfgs1 done (no MC) ===\n"); fflush(stdout);
		clReleaseMemObject(obuf); clReleaseKernel(bk);
		return;
	}

	/**** Launch kernel2_multi ****/
	cl_kernel k2m = clCreateKernel(programs[1], "kernel2_multi", &err); checkErr(err);
	{
		cl_uint nargs = 0; clGetKernelInfo(k2m, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &nargs, NULL);
		printf("DIAG GPU%d: kernel2_multi CL_KERNEL_NUM_ARGS=%u (expect 13)\n", gpu_id, nargs); fflush(stdout);
		cl_ulong priv = 0; cl_device_id dev = devices[gpu_id];
		clGetKernelWorkGroupInfo(k2m, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &priv, NULL);
		printf("DIAG GPU%d: kernel2_multi private_mem=%llu bytes/work-item\n", gpu_id, (unsigned long long)priv); fflush(stdout);
	}
	SetKernelArg(k2m,  0, sizeof(cl_mem), &ric_gpu);
	SetKernelArg(k2m,  1, sizeof(cl_mem), &mg_gpu);
	SetKernelArg(k2m,  2, sizeof(cl_mem), &pre_gpu);
	SetKernelArg(k2m,  3, sizeof(cl_mem), &grids_gpu);
	SetKernelArg(k2m,  4, sizeof(cl_mem), &rand_gpu);
	SetKernelArg(k2m,  5, sizeof(cl_mem), &results_gpu);
	SetKernelArg(k2m,  6, sizeof(cl_mem), &coords_gpu);
	SetKernelArg(k2m,  7, sizeof(cl_mem), &mis_gpu);
	SetKernelArg(k2m,  8, sizeof(int),    &N);
	SetKernelArg(k2m,  9, sizeof(int),    &total_torsions);
	SetKernelArg(k2m, 10, sizeof(int),    &search_depth);
	SetKernelArg(k2m, 11, sizeof(int),    &bfgs_steps);
	SetKernelArg(k2m, 12, sizeof(int),    &n_traj);

	// Clamp the work-group size to what the kernel actually supports (large private memory
	// shrinks the max work-group), and round the global size up to a multiple of it.
	size_t k2m_max_wg = 64;
	clGetKernelWorkGroupInfo(k2m, devices[gpu_id], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &k2m_max_wg, NULL);
	size_t local_wg = k2m_max_wg < 64 ? k2m_max_wg : 64;
	if (local_wg < 1) local_wg = 1;
	if (const char* e = getenv("VINA_MULTI_LWS")) { int v = atoi(e); if (v >= 1) local_wg = (size_t)v; }
	size_t gw = ((multi_wi + local_wg - 1) / local_wg) * local_wg;
	printf("DIAG GPU%d: kernel2_multi max_wg=%zu  local=%zu  global=%zu\n", gpu_id, k2m_max_wg, local_wg, gw); fflush(stdout);
	const size_t gws[2] = { gw, 1 };
	const size_t lws[2] = { local_wg, 1 };
	{
		auto _k0 = std::chrono::steady_clock::now();
		cl_event ev;
		err = clEnqueueNDRangeKernel(queue, k2m, 2, 0, gws, lws, 0, NULL, &ev); checkErr(err);
		clFlush(queue);
		clWaitForEvents(1, &ev);
		clReleaseEvent(ev);
		auto _k1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: kernel2_multi = %.1fms\n", gpu_id,
		       std::chrono::duration<double, std::milli>(_k1 - _k0).count()); fflush(stdout);
	}

	err = clEnqueueReadBuffer(queue, results_gpu, false, 0, res_sz,    results,    0, NULL, NULL); checkErr(err);
	err = clEnqueueReadBuffer(queue, coords_gpu,  false, 0, coords_sz, out_coords, 0, NULL, NULL); checkErr(err);
	clFinish(queue);

	/**** Pick best trajectory; reconstruct the joint conf for output ****/
	int best = 0; float best_e = results[0].e;
	for (int t = 1; t < n_traj; t++) if (results[t].e < best_e) { best_e = results[t].e; best = t; }
	printf("DIAG GPU%d: best trajectory %d  combined E = %.3f kcal/mol\n", gpu_id, best, best_e); fflush(stdout);

	output_type best_ot(s, best_e);
	for (int k = 0; k < N; k++) {
		best_ot.c.ligands[k].rigid.position = vec(results[best].position[k][0],
		                                          results[best].position[k][1],
		                                          results[best].position[k][2]);
		best_ot.c.ligands[k].rigid.orientation = qt(results[best].orientation[k][0],
		                                            results[best].orientation[k][1],
		                                            results[best].orientation[k][2],
		                                            results[best].orientation[k][3]);
		for (int j = 0; j < (int)s.ligands[k]; j++)
			best_ot.c.ligands[k].torsions[j] = results[best].lig_torsion[tors_off[k] + j];
	}
	best_ot.e = best_e;
	out.push_back(new output_type(best_ot));

	free(results); free(out_coords); free(mis_ptr);
	err = clReleaseKernel(k2m);            checkErr(err);
	err = clReleaseMemObject(ric_gpu);     checkErr(err);
	err = clReleaseMemObject(mg_gpu);      checkErr(err);
	err = clReleaseMemObject(rand_gpu);    checkErr(err);
	err = clReleaseMemObject(results_gpu); checkErr(err);
	err = clReleaseMemObject(coords_gpu);  checkErr(err);
	err = clReleaseMemObject(mis_gpu);     checkErr(err);
	err = clReleaseMemObject(grids_gpu);   checkErr(err);
	err = clReleaseMemObject(pre_gpu);     checkErr(err);

	local_status.store(static_cast<int>(DockingStatus::Finish), std::memory_order_release);
	{
		auto _fn_t1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: main_procedure_multi done in %.2fs\n", gpu_id,
		       std::chrono::duration<double>(_fn_t1 - _fn_t0).count()); fflush(stdout);
	}
}
