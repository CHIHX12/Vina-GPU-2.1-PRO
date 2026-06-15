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
#include "ligand_prep.h"
#include "protein_gpu.h"
#include "main_procedure_cl.h"
#include <numeric>
#include <algorithm>

using namespace std;

void main_procedure_cl_dual(
	cache& c,
	model& ma, model& mb,
	const precalculate& p,
	const parallel_mc par,
	const vec& corner1, const vec& corner2,
	const int seed,
	output_container& out_a, output_container& out_b,
	std::string opencl_binary_path,
	const int rilc_bfgs,
	const int gpu_id,
	const bool use_ad4zn)
{
	auto _fn_t0 = std::chrono::steady_clock::now();

	cl_int err;

	// Lazy singleton: create OCL context+queue once per GPU ID and reuse.
	// Eliminates ~220ms driver-init overhead on every dual-ligand call.
	struct OclCtx {
		cl_platform_id* platforms = nullptr;
		cl_device_id*   devices   = nullptr;
		cl_context      context   = nullptr;
		cl_command_queue queue    = nullptr;
		cl_program       progs[2] = { nullptr, nullptr };
		int              gpu_id   = -1;
		std::string      bin_path;
	};
	// Per-GPU singleton map: one OclCtx per GPU ID so multiple GPUs can run in parallel.
	static std::mutex                  g_ocl_mutex;
	static std::map<int, OclCtx>       g_ocl_map;

	cl_platform_id* platforms;
	cl_device_id*   devices;
	cl_context      context;
	cl_command_queue queue;
	cl_int gpu_platform_id = 0;
	cl_program programs[2];

	{
		std::lock_guard<std::mutex> _lock(g_ocl_mutex);
		OclCtx& g_ocl = g_ocl_map[gpu_id];
		bool need_init  = (g_ocl.context == nullptr || g_ocl.gpu_id != gpu_id);
		bool need_progs = need_init || g_ocl.bin_path != opencl_binary_path || g_ocl.progs[0] == nullptr;

		if (need_init) {
			auto _ti = std::chrono::steady_clock::now();
			SetupPlatform(&g_ocl.platforms, &gpu_platform_id);
			SetupDevice(g_ocl.platforms, &g_ocl.devices, gpu_platform_id);
			SetupContext(g_ocl.platforms, g_ocl.devices + gpu_id, &g_ocl.context, 1, gpu_platform_id);
			SetupQueue(&g_ocl.queue, g_ocl.context, g_ocl.devices, gpu_id);
			g_ocl.gpu_id = gpu_id;
			auto _tj = std::chrono::steady_clock::now();
			printf("DIAG GPU%d: OCL_init=%.0fms (first call)\n", gpu_id,
			       std::chrono::duration<double,std::milli>(_tj-_ti).count()); fflush(stdout);
		} else {
			printf("DIAG GPU%d: OCL_init=0ms (cached)\n", gpu_id); fflush(stdout);
		}
		platforms = g_ocl.platforms;
		devices   = g_ocl.devices;
		context   = g_ocl.context;
		queue     = g_ocl.queue;
	}

	printf("\nUsing random seed: %d", seed);

#ifdef BUILD_KERNEL_FROM_SOURCE
{
	static std::mutex kernel_compile_mutex;
	std::lock_guard<std::mutex> compile_lock(kernel_compile_mutex);

	const std::string bin_out_path = opencl_binary_path.empty() ? "." : opencl_binary_path;
	const std::string bin1 = bin_out_path + "/Kernel1_Opt.bin";
	const std::string bin2 = bin_out_path + "/Kernel2_Opt.bin";

	auto file_exists = [](const std::string& fp) { std::ifstream f(fp); return f.good(); };

	if (!file_exists(bin1) || !file_exists(bin2)) {
		const char* vina_home_env = std::getenv("VINA_GPU_HOME");
		const std::string kernel_src_path = vina_home_env ? std::string(vina_home_env) : ".";
		const std::string include_path = kernel_src_path + "/OpenCL/inc";
		const std::string addtion = "";

		printf("\nCompiling GPU kernels (one-time, caching to %s)...\n", bin_out_path.c_str()); fflush(stdout);

		char* program1_file_n[NUM_OF_FILES_KERNEL_1];
		size_t program1_size_n[NUM_OF_FILES_KERNEL_1];
		std::string file1_paths[NUM_OF_FILES_KERNEL_1] = {
			kernel_src_path + "/OpenCL/src/kernels/code_head.cl",
			kernel_src_path + "/OpenCL/src/kernels/kernel1.cl"
		};
		read_n_file(program1_file_n, program1_size_n, file1_paths, NUM_OF_FILES_KERNEL_1);
		std::string final_file;
		size_t final_size = NUM_OF_FILES_KERNEL_1 - 1;
		for (int i = 0; i < NUM_OF_FILES_KERNEL_1; i++) {
			final_file = (i == 0) ? program1_file_n[0] : final_file + '\n' + (std::string)program1_file_n[i];
			final_size += program1_size_n[i];
		}
		const char* final_files1_char = final_file.data();
		programs[0] = clCreateProgramWithSource(context, 1, (const char**)&final_files1_char, &final_size, &err); checkErr(err);
		SetupBuildProgramWithSource(programs[0], NULL, devices + gpu_id, include_path, addtion);
		SaveProgramToBinary(programs[0], bin1.c_str());

		char* program2_file_n[NUM_OF_FILES_KERNEL_2];
		size_t program2_size_n[NUM_OF_FILES_KERNEL_2];
		std::string file2_paths[NUM_OF_FILES_KERNEL_2] = {
			kernel_src_path + "/OpenCL/src/kernels/code_head.cl",
			kernel_src_path + "/OpenCL/src/kernels/mutate_conf.cl",
			kernel_src_path + "/OpenCL/src/kernels/matrix.cl",
			kernel_src_path + "/OpenCL/src/kernels/quasi_newton.cl",
			kernel_src_path + "/OpenCL/src/kernels/kernel2.cl"
		};
		read_n_file(program2_file_n, program2_size_n, file2_paths, NUM_OF_FILES_KERNEL_2);
		final_size = NUM_OF_FILES_KERNEL_2 - 1;
		for (int i = 0; i < NUM_OF_FILES_KERNEL_2; i++) {
			final_file = (i == 0) ? program2_file_n[0] : final_file + '\n' + (std::string)program2_file_n[i];
			final_size += program2_size_n[i];
		}
		const char* final_files2_char = final_file.data();
		programs[1] = clCreateProgramWithSource(context, 1, (const char**)&final_files2_char, &final_size, &err); checkErr(err);
		SetupBuildProgramWithSource(programs[1], NULL, devices + gpu_id, include_path, addtion);
		SaveProgramToBinary(programs[1], bin2.c_str());
		printf("Kernel compilation done.\n"); fflush(stdout);
	}
}
#endif
	{
		std::lock_guard<std::mutex> _lock(g_ocl_mutex);
		OclCtx& g_ocl = g_ocl_map[gpu_id];
		bool need_progs = (g_ocl.progs[0] == nullptr || g_ocl.bin_path != opencl_binary_path);
		if (need_progs) {
			auto _t = std::chrono::steady_clock::now();
			g_ocl.progs[0] = SetupBuildProgramWithBinary(context, devices + gpu_id,
			    (opencl_binary_path + std::string("/Kernel1_Opt.bin")).c_str());
			auto _t1 = std::chrono::steady_clock::now();
			g_ocl.progs[1] = SetupBuildProgramWithBinary(context, devices + gpu_id,
			    (opencl_binary_path + std::string("/Kernel2_Opt.bin")).c_str());
			auto _t2 = std::chrono::steady_clock::now();
			g_ocl.bin_path = opencl_binary_path;
			printf("DIAG GPU%d: LoadBin1=%.0fms  LoadBin2=%.0fms (first load)\n", gpu_id,
			       std::chrono::duration<double,std::milli>(_t1-_t).count(),
			       std::chrono::duration<double,std::milli>(_t2-_t1).count()); fflush(stdout);
		} else {
			printf("DIAG GPU%d: LoadBin=0ms (cached)\n", gpu_id); fflush(stdout);
		}
		programs[0] = g_ocl.progs[0];
		programs[1] = g_ocl.progs[1];
	}

	// clUnloadPlatformCompiler only needed after source compilation (not binary load)
	cl_kernel kernels[2];
	char kernel_name[][50] = { "kernel1", "kernel2" };
	SetupKernel(kernels, programs, 2, kernel_name);

	size_t max_wg_size;
	size_t max_wi_size[3];
	err = clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL); checkErr(err);
	err = clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(size_t), &max_wi_size, NULL); checkErr(err);

	/**** Original Vina grid setup (reused from single-ligand path) ****/
	sz nat = num_atom_types(c.atu);

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
	if (needed.empty()) {
		for (sz i = 0; i < nat; i++) needed.push_back(i);
	}

	flv affinities(needed.size());
	grid& g = c.grids[needed.front()];
	const fl cutoff_sqr = p.cutoff_sqr();
	grid_dims gd_reduced = szv_grid_dims(c.gd);
	szv_grid ig(ma, gd_reduced, cutoff_sqr);

	for (int i = 0; i < 3; i++) {
		if (ig.m_init[i] != g.m_init[i]) { printf("m_init not equal!"); exit(-1); }
		if (ig.m_range[i] != g.m_range[i]) { printf("m_range not equal!"); exit(-1); }
	}

	vec authentic_v(1000, 1000, 1000);

	std::atomic<int> local_status{static_cast<int>(DockingStatus::Docking)};
#ifdef NDEBUG
	std::thread console_thread(print_process, &local_status);
	ThreadGuard console_guard(console_thread);
#endif

	ProteinGpuBuffers __pg = protein_gpu_setup(c, ma, p, ig, g, needed, par, context, queue,
	        kernels[0], max_wi_size, authentic_v, cutoff_sqr, use_ad4zn, gpu_id, 2);
	cl_mem pre_gpu = __pg.pre_gpu, grids_gpu = __pg.grids_gpu, mis_gpu = __pg.mis_gpu;
	mis_cl* mis_ptr = __pg.mis_ptr;

	// LS metal rescoring for dual (receptor metals shared by both ligands)
	auto dual_receptor_metals = collect_receptor_metals(ma);
	float dual_ls_metal_weight = 0.30f;
	{ const char* env = std::getenv("VINA_LS_METAL_WEIGHT"); if (env) dual_ls_metal_weight = (float)std::atof(env); }
	if (!dual_receptor_metals.empty())
		printf("QFD LS (dual): %zu receptor metal(s) found (weight=%.3f)\n",
		       dual_receptor_metals.size(), dual_ls_metal_weight);

	/**** Prepare ligand data (CPU, for both ligands) ****/
	const int thread = par.mc.thread;

	// Helper lambda: populate one m_cl from a model
	// fill_m_cl now shared from ligand_prep.h (was a local lambda)

	rng generator(static_cast<rng::result_type>(seed));
	auto _tlig0 = std::chrono::steady_clock::now();

	// Ligand A
	conf_size sa = ma.get_size(); output_type tmp_a(sa, 0);
	int torsion_a = (int)tmp_a.c.ligands[0].torsions.size();
	if (torsion_a >= MAX_NUM_OF_LIG_TORSION) throw std::runtime_error("Ligand A too many torsions");

	random_maps* rmaps_a = (random_maps*)malloc(sizeof(random_maps));
	for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
		rmaps_a->int_map[k] = random_int(0, torsion_a, generator);
		rmaps_a->pi_map[k]  = random_fl(-pi, pi, generator);
	}
	for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
		vec rc = random_inside_sphere(generator);
		for (int j = 0; j < 3; j++) rmaps_a->sphere_map[k][j] = rc[j];
	}
	output_type_cl* ric_a = (output_type_cl*)malloc(thread * sizeof(output_type_cl));
	for (int k = 0; k < thread; k++) {
		tmp_a.c.randomize(corner1, corner2, generator);
		for (int j = 0; j < 3; j++) ric_a[k].position[j] = tmp_a.c.ligands[0].rigid.position[j];
		ric_a[k].orientation[0] = tmp_a.c.ligands[0].rigid.orientation.R_component_1();
		ric_a[k].orientation[1] = tmp_a.c.ligands[0].rigid.orientation.R_component_2();
		ric_a[k].orientation[2] = tmp_a.c.ligands[0].rigid.orientation.R_component_3();
		ric_a[k].orientation[3] = tmp_a.c.ligands[0].rigid.orientation.R_component_4();
		for (int j = 0; j < torsion_a; j++) ric_a[k].lig_torsion[j] = tmp_a.c.ligands[0].torsions[j];
	}
	m_cl* mg_a = (m_cl*)malloc(sizeof(m_cl));
	fill_m_cl(ma, mg_a);

	// Ligand B
	conf_size sb = mb.get_size(); output_type tmp_b(sb, 0);
	int torsion_b = (int)tmp_b.c.ligands[0].torsions.size();
	if (torsion_b >= MAX_NUM_OF_LIG_TORSION) throw std::runtime_error("Ligand B too many torsions");

	random_maps* rmaps_b = (random_maps*)malloc(sizeof(random_maps));
	for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
		rmaps_b->int_map[k] = random_int(0, torsion_b, generator);
		rmaps_b->pi_map[k]  = random_fl(-pi, pi, generator);
	}
	for (int k = 0; k < MAX_NUM_OF_RANDOM_MAP; k++) {
		vec rc = random_inside_sphere(generator);
		for (int j = 0; j < 3; j++) rmaps_b->sphere_map[k][j] = rc[j];
	}
	output_type_cl* ric_b = (output_type_cl*)malloc(thread * sizeof(output_type_cl));
	for (int k = 0; k < thread; k++) {
		tmp_b.c.randomize(corner1, corner2, generator);
		for (int j = 0; j < 3; j++) ric_b[k].position[j] = tmp_b.c.ligands[0].rigid.position[j];
		ric_b[k].orientation[0] = tmp_b.c.ligands[0].rigid.orientation.R_component_1();
		ric_b[k].orientation[1] = tmp_b.c.ligands[0].rigid.orientation.R_component_2();
		ric_b[k].orientation[2] = tmp_b.c.ligands[0].rigid.orientation.R_component_3();
		ric_b[k].orientation[3] = tmp_b.c.ligands[0].rigid.orientation.R_component_4();
		for (int j = 0; j < torsion_b; j++) ric_b[k].lig_torsion[j] = tmp_b.c.ligands[0].torsions[j];
	}
	m_cl* mg_b = (m_cl*)malloc(sizeof(m_cl));
	fill_m_cl(mb, mg_b);

	// ───── Dual coupling fix (2026-06-14): make the two ligands feel each other ─────
	// Bug: kernel2_dual optimised A and B with INDEPENDENT BFGS, so both collapsed onto
	// the same site and clashed (the ligand–ligand term was only in the Metropolis accept,
	// not in the gradient). Fix: append the PARTNER ligand's movable atoms into each
	// ligand's coord array (beyond its movable range, so set() never moves them) and build
	// inter-ligand interaction pairs in `other_pairs`. m_eval_deriv already evaluates
	// other_pairs with FORCES, so the A↔B repulsion now enters each ligand's BFGS gradient
	// (block-coordinate descent: A optimised vs fixed B, B vs fixed A, alternating).
	{
		const atom_type::t atu_t = p.atom_typing_used();
		const sz n_at = num_atom_types(atu_t);
		const int na = (int)ma.num_movable_atoms();
		const int nb = (int)mb.num_movable_atoms();
		if (na + nb > MAX_NUM_OF_ATOMS)
			throw std::runtime_error("dual: na+nb exceeds MAX_NUM_OF_ATOMS (raise it)");
		if ((long long)na * nb > MAX_NUM_OF_OTHER_PAIRS)
			throw std::runtime_error("dual: too many inter-ligand pairs (raise MAX_NUM_OF_OTHER_PAIRS)");

		// Append B's movable atoms into A's arrays at [na, na+nb); A's into B's at [nb, nb+na).
		// (Types are static; coords get refreshed every BFGS step by the kernel's SYNC.)
		for (int j = 0; j < nb; j++) {
			mg_a->atoms[na + j] = mg_b->atoms[j];
			for (int d = 0; d < 3; d++) mg_a->m_coords.coords[na + j][d] = mg_b->m_coords.coords[j][d];
		}
		for (int i = 0; i < na; i++) {
			mg_b->atoms[nb + i] = mg_a->atoms[i];
			for (int d = 0; d < 3; d++) mg_b->m_coords.coords[nb + i][d] = mg_a->m_coords.coords[i][d];
		}

		// Inter-ligand pairs (skip atoms unassigned in this typing, mirroring eval_lig_lig).
		int np = 0;
		for (int i = 0; i < na; i++) {
			sz t1 = ma.atoms[i].get(atu_t);
			if (t1 >= n_at) continue;
			for (int j = 0; j < nb; j++) {
				sz t2 = mb.atoms[j].get(atu_t);
				if (t2 >= n_at) continue;
				int tpi = (int)get_type_pair_index(atu_t, ma.atoms[i], mb.atoms[j]);
				mg_a->other_pairs.type_pair_index[np] = tpi;  // A view: A atom i ↔ appended B atom (na+j)
				mg_a->other_pairs.a[np] = i;
				mg_a->other_pairs.b[np] = na + j;
				mg_b->other_pairs.type_pair_index[np] = tpi;  // B view: B atom j ↔ appended A atom (nb+i)
				mg_b->other_pairs.a[np] = j;
				mg_b->other_pairs.b[np] = nb + i;
				np++;
			}
		}
		mg_a->other_pairs.num_pairs = np;
		mg_b->other_pairs.num_pairs = np;
		printf("Dual coupling: %d inter-ligand pairs (A=%d atoms, B=%d atoms) -> ligands now repel in BFGS\n",
		       np, na, nb);
		fflush(stdout);
	}

	// Result buffers
	output_type_cl*        results_a = (output_type_cl*)malloc(thread * sizeof(output_type_cl));
	output_type_cl*        results_b = (output_type_cl*)malloc(thread * sizeof(output_type_cl));
	ligand_atom_coords_cl* coords_a  = (ligand_atom_coords_cl*)malloc(thread * sizeof(ligand_atom_coords_cl));
	ligand_atom_coords_cl* coords_b  = (ligand_atom_coords_cl*)malloc(thread * sizeof(ligand_atom_coords_cl));
	float*                 combined  = (float*)malloc(thread * sizeof(float));  // per-traj combined E for PAIRED ranking

	/**** Allocate GPU buffers for kernel2_dual ****/
	const size_t ric_sz    = (size_t)thread * sizeof(output_type_cl);
	const size_t m_sz      = sizeof(m_cl);
	const size_t rmap_sz   = sizeof(random_maps);
	const size_t res_sz    = (size_t)thread * sizeof(output_type_cl);
	const size_t coords_sz = (size_t)thread * sizeof(ligand_atom_coords_cl);
	const size_t int_sz    = sizeof(int);

	cl_mem ric_a_gpu, ric_b_gpu;
	cl_mem mg_a_gpu,  mg_b_gpu;
	cl_mem rmaps_a_gpu, rmaps_b_gpu;
	cl_mem results_a_gpu, results_b_gpu;
	cl_mem coords_a_gpu,  coords_b_gpu;
	cl_mem torsion_a_gpu, torsion_b_gpu;
	cl_mem search_depth_gpu, bfgs_steps_gpu;
	cl_mem combined_gpu;

	CreateDeviceBuffer(&combined_gpu,    CL_MEM_WRITE_ONLY, (size_t)thread * sizeof(float), context);
	CreateDeviceBuffer(&ric_a_gpu,       CL_MEM_READ_ONLY,  ric_sz,    context);
	CreateDeviceBuffer(&ric_b_gpu,       CL_MEM_READ_ONLY,  ric_sz,    context);
	CreateDeviceBuffer(&mg_a_gpu,        CL_MEM_READ_WRITE, m_sz,      context);
	CreateDeviceBuffer(&mg_b_gpu,        CL_MEM_READ_WRITE, m_sz,      context);
	CreateDeviceBuffer(&rmaps_a_gpu,     CL_MEM_READ_ONLY,  rmap_sz,   context);
	CreateDeviceBuffer(&rmaps_b_gpu,     CL_MEM_READ_ONLY,  rmap_sz,   context);
	CreateDeviceBuffer(&results_a_gpu,   CL_MEM_WRITE_ONLY, res_sz,    context);
	CreateDeviceBuffer(&results_b_gpu,   CL_MEM_WRITE_ONLY, res_sz,    context);
	CreateDeviceBuffer(&coords_a_gpu,    CL_MEM_WRITE_ONLY, coords_sz, context);
	CreateDeviceBuffer(&coords_b_gpu,    CL_MEM_WRITE_ONLY, coords_sz, context);
	CreateDeviceBuffer(&torsion_a_gpu,   CL_MEM_READ_ONLY,  int_sz,    context);
	CreateDeviceBuffer(&torsion_b_gpu,   CL_MEM_READ_ONLY,  int_sz,    context);
	CreateDeviceBuffer(&search_depth_gpu,CL_MEM_READ_ONLY,  int_sz,    context);
	CreateDeviceBuffer(&bfgs_steps_gpu,  CL_MEM_READ_ONLY,  int_sz,    context);

	// Upload
	err = clEnqueueWriteBuffer(queue, ric_a_gpu,    false, 0, ric_sz,  ric_a,   0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, ric_b_gpu,    false, 0, ric_sz,  ric_b,   0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, mg_a_gpu,     false, 0, m_sz,    mg_a,    0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, mg_b_gpu,     false, 0, m_sz,    mg_b,    0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, rmaps_a_gpu,  false, 0, rmap_sz, rmaps_a, 0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, rmaps_b_gpu,  false, 0, rmap_sz, rmaps_b, 0, NULL, NULL); checkErr(err);

	int sd_val   = par.mc.search_depth.empty() ? 20 : par.mc.search_depth[0];
	int bfgs_val = par.mc.ssd_par.bfgs_steps.empty() ? 300 : par.mc.ssd_par.bfgs_steps[0];
	err = clEnqueueWriteBuffer(queue, torsion_a_gpu,   false, 0, int_sz, &torsion_a, 0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, torsion_b_gpu,   false, 0, int_sz, &torsion_b, 0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, search_depth_gpu,false, 0, int_sz, &sd_val,    0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, bfgs_steps_gpu,  false, 0, int_sz, &bfgs_val,  0, NULL, NULL); checkErr(err);
	clFinish(queue);
	{
		auto _tlig1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: lig_prep+upload=%.0fms\n", gpu_id,
		       std::chrono::duration<double,std::milli>(_tlig1-_tlig0).count()); fflush(stdout);
	}

	free(ric_a); free(ric_b); free(mg_a); free(mg_b); free(rmaps_a); free(rmaps_b);

	/**** Set up and launch kernel2_dual ****/
	cl_kernel k2d = clCreateKernel(programs[1], "kernel2_dual", &err); checkErr(err);
	{
		cl_uint k2d_nargs = 0;
		clGetKernelInfo(k2d, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &k2d_nargs, NULL);
		printf("DIAG GPU%d: kernel2_dual CL_KERNEL_NUM_ARGS=%u (expect 20)\n", gpu_id, k2d_nargs); fflush(stdout);
		size_t k2d_wgs = 0; cl_ulong k2d_priv = 0, k2d_local = 0;
		cl_device_id k2d_dev = NULL;
		clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &k2d_dev, NULL);
		clGetKernelWorkGroupInfo(k2d, k2d_dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &k2d_wgs, NULL);
		clGetKernelWorkGroupInfo(k2d, k2d_dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &k2d_priv, NULL);
		clGetKernelWorkGroupInfo(k2d, k2d_dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &k2d_local, NULL);
		printf("DIAG GPU%d: kernel2_dual max_wgs=%zu  private_mem=%llu bytes  local_mem=%llu bytes\n",
		       gpu_id, k2d_wgs, (unsigned long long)k2d_priv, (unsigned long long)k2d_local); fflush(stdout);
	}

	// kernel2_dual args:
	//  0: ric_a   1: ric_b     2: mg_a     3: mg_b     4: pre       5: grids
	//  6: rmaps_a 7: rmaps_b   8: coords_a 9: coords_b 10: results_a 11: results_b
	// 12: mis     13: torsion_a 14: torsion_b 15: search_depth 16: bfgs_steps
	// 17: rilc_bfgs_enable  18: batch_n (=1 for single pair)
	int batch_n_dual = 1;
	SetKernelArg(k2d,  0, sizeof(cl_mem), &ric_a_gpu);
	SetKernelArg(k2d,  1, sizeof(cl_mem), &ric_b_gpu);
	SetKernelArg(k2d,  2, sizeof(cl_mem), &mg_a_gpu);
	SetKernelArg(k2d,  3, sizeof(cl_mem), &mg_b_gpu);
	SetKernelArg(k2d,  4, sizeof(cl_mem), &pre_gpu);
	SetKernelArg(k2d,  5, sizeof(cl_mem), &grids_gpu);
	SetKernelArg(k2d,  6, sizeof(cl_mem), &rmaps_a_gpu);
	SetKernelArg(k2d,  7, sizeof(cl_mem), &rmaps_b_gpu);
	SetKernelArg(k2d,  8, sizeof(cl_mem), &coords_a_gpu);
	SetKernelArg(k2d,  9, sizeof(cl_mem), &coords_b_gpu);
	SetKernelArg(k2d, 10, sizeof(cl_mem), &results_a_gpu);
	SetKernelArg(k2d, 11, sizeof(cl_mem), &results_b_gpu);
	SetKernelArg(k2d, 12, sizeof(cl_mem), &mis_gpu);
	SetKernelArg(k2d, 13, sizeof(cl_mem), &torsion_a_gpu);
	SetKernelArg(k2d, 14, sizeof(cl_mem), &torsion_b_gpu);
	SetKernelArg(k2d, 15, sizeof(cl_mem), &search_depth_gpu);
	SetKernelArg(k2d, 16, sizeof(cl_mem), &bfgs_steps_gpu);
	SetKernelArg(k2d, 17, sizeof(int),    &rilc_bfgs);
	SetKernelArg(k2d, 18, sizeof(int),    &batch_n_dual);
	SetKernelArg(k2d, 19, sizeof(cl_mem), &combined_gpu);

	// Work-item count is decoupled from trajectory count: the kernel strides
	// (gll += total_wi) over all `thread` trajectories, so each WI handles
	// thread/total_wi of them. The dual kernel holds 2× m_cl_private + 2× Hessian
	// in private memory per WI (~150 KB), so 18176 WIs requested ~2.7 GB of private
	// memory → CL_OUT_OF_HOST_MEMORY. 4096 WIs (32 groups) keeps that ~600 MB and
	// fully covers the trajectories. (Single-ligand kernel is unaffected.)
	size_t dual_wi = 4096;
	if (const char* e = getenv("VINA_DUAL_WI")) { int v = atoi(e); if (v >= 128) dual_wi = ((v + 127) / 128) * 128; }
	const size_t k2d_global[2] = { dual_wi, 1 };
	const size_t k2d_local[2]  = { 128, 1 };

	{
		auto _k2_t0 = std::chrono::steady_clock::now();
		cl_event k2d_ev;
		err = clEnqueueNDRangeKernel(queue, k2d, 2, 0, k2d_global, k2d_local, 0, NULL, &k2d_ev); checkErr(err);
		clFlush(queue);
		clWaitForEvents(1, &k2d_ev);
		clReleaseEvent(k2d_ev);
		auto _k2_t1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: kernel2_dual = %.1fms\n", gpu_id,
		       std::chrono::duration<double, std::milli>(_k2_t1 - _k2_t0).count()); fflush(stdout);
	}

	// Download results
	err = clEnqueueReadBuffer(queue, results_a_gpu, false, 0, res_sz,    results_a, 0, NULL, NULL); checkErr(err);
	err = clEnqueueReadBuffer(queue, results_b_gpu, false, 0, res_sz,    results_b, 0, NULL, NULL); checkErr(err);
	err = clEnqueueReadBuffer(queue, coords_a_gpu,  false, 0, coords_sz, coords_a,  0, NULL, NULL); checkErr(err);
	err = clEnqueueReadBuffer(queue, coords_b_gpu,  false, 0, coords_sz, coords_b,  0, NULL, NULL); checkErr(err);
	err = clEnqueueReadBuffer(queue, combined_gpu,  false, 0, (size_t)thread*sizeof(float), combined, 0, NULL, NULL); checkErr(err);
	clFinish(queue);

	// Convert GPU results → Vina format, then fast top-K by energy (skip O(thread×K×atoms) RMSD loop)
	auto t_cv0 = std::chrono::steady_clock::now();
	std::vector<output_type> vina_a = cl_to_vina(results_a, coords_a, thread, torsion_a);
	std::vector<output_type> vina_b = cl_to_vina(results_b, coords_b, thread, torsion_b);
	// LS metal rescoring: adjust .e before fast_topk sorts
	apply_ls_metal_scores(vina_a, dual_receptor_metals, dual_ls_metal_weight);
	apply_ls_metal_scores(vina_b, dual_receptor_metals, dual_ls_metal_weight);
	auto t_cv1 = std::chrono::steady_clock::now();

	// PAIRED ranking (dual coupling fix): A and B from the SAME trajectory must be output
	// together, ranked by the per-trajectory COMBINED energy. Sorting each ligand independently
	// (the old behaviour) tore the pair apart — best-A and best-B came from different trajectories
	// and overlapped. cl_to_vina preserves trajectory order, so vina_a[i]/vina_b[i] are a pair.
	{
		const int K = par.mc.num_saved_mins;
		std::vector<int> order(thread);
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(),
		          [&](int x, int y){ return combined[x] < combined[y]; });
		int cnt = 0;
		for (int oi : order) {
			if (cnt >= K) break;
			if (!not_max(vina_a[oi].e) || !not_max(vina_b[oi].e)) continue;
			out_a.push_back(new output_type(vina_a[oi]));
			out_b.push_back(new output_type(vina_b[oi]));
			++cnt;
		}
	}
	auto t_cv2 = std::chrono::steady_clock::now();
	printf("DIAG GPU%d: cl_to_vina=%.1fms  topK=%.1fms\n", gpu_id,
	       std::chrono::duration<double,std::milli>(t_cv1-t_cv0).count(),
	       std::chrono::duration<double,std::milli>(t_cv2-t_cv1).count()); fflush(stdout);

	// Cleanup
	free(results_a); free(results_b); free(coords_a); free(coords_b); free(combined);
	free(mis_ptr);

	err = clReleaseKernel(k2d);                    checkErr(err);
	err = clReleaseMemObject(ric_a_gpu);           checkErr(err);
	err = clReleaseMemObject(ric_b_gpu);           checkErr(err);
	err = clReleaseMemObject(mg_a_gpu);            checkErr(err);
	err = clReleaseMemObject(mg_b_gpu);            checkErr(err);
	err = clReleaseMemObject(rmaps_a_gpu);         checkErr(err);
	err = clReleaseMemObject(rmaps_b_gpu);         checkErr(err);
	err = clReleaseMemObject(results_a_gpu);       checkErr(err);
	err = clReleaseMemObject(results_b_gpu);       checkErr(err);
	err = clReleaseMemObject(coords_a_gpu);        checkErr(err);
	err = clReleaseMemObject(coords_b_gpu);        checkErr(err);
	err = clReleaseMemObject(combined_gpu);        checkErr(err);
	err = clReleaseMemObject(torsion_a_gpu);       checkErr(err);
	err = clReleaseMemObject(torsion_b_gpu);       checkErr(err);
	err = clReleaseMemObject(search_depth_gpu);    checkErr(err);
	err = clReleaseMemObject(bfgs_steps_gpu);      checkErr(err);
	err = clReleaseMemObject(mis_gpu);             checkErr(err);
	err = clReleaseMemObject(grids_gpu);           checkErr(err);
	err = clReleaseMemObject(pre_gpu);             checkErr(err);

	local_status.store(static_cast<int>(DockingStatus::Finish), std::memory_order_release);
	{
		auto _fn_t1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: main_procedure_cl_dual done in %.2fs\n", gpu_id,
		       std::chrono::duration<double>(_fn_t1 - _fn_t0).count()); fflush(stdout);
	}
}
