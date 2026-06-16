#include "protein_gpu.h"
#include "wrapcl.h"
#include "qfd_grids.h"
#include "ls_metal.h"
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace std;

ProteinGpuBuffers protein_gpu_setup(
    cache& c, const model& m, const precalculate& p, szv_grid& ig, grid& g,
    const szv& needed, const parallel_mc& par, cl_context context, cl_command_queue queue,
    cl_kernel kernel1_k, const size_t* max_wi_size, const vec& authentic_v, fl cutoff_sqr,
    bool use_ad4zn, int gpu_id, int num_ligands)
{
	cl_int err;
	int nat = num_atom_types(c.atu);

	// Preparing protein atoms related data
	pa_cl* pa_ptr = (pa_cl*)malloc(sizeof(pa_cl));
	if(MAX_NUM_OF_PROTEIN_ATOMS <= m.grid_atoms.size()){throw std::runtime_error("pocket too large!");}
	for (int i = 0; i < m.grid_atoms.size(); i++) {
		pa_ptr->atoms[i].types[0] = m.grid_atoms[i].el;
		pa_ptr->atoms[i].types[1] = m.grid_atoms[i].ad;
		pa_ptr->atoms[i].types[2] = m.grid_atoms[i].xs;
		pa_ptr->atoms[i].types[3] = m.grid_atoms[i].sy;
		for (int j = 0; j < 3; j++)pa_ptr->atoms[i].coords[j] = m.grid_atoms[i].coords.data[j];
	}
	size_t pa_size = sizeof(pa_cl);

	// Preaparing precalculated look up table related data
	pre_cl* pre_ptr = (pre_cl*)malloc(sizeof(pre_cl));
	pre_ptr->m_cutoff_sqr = p.cutoff_sqr();
	pre_ptr->factor = p.factor;
	pre_ptr->n = p.n;
	if(MAX_P_DATA_M_DATA_SIZE <= p.data.m_data.size()){throw std::runtime_error("LUT too large!");}
	for (int i = 0; i < p.data.m_data.size(); i++) {
		pre_ptr->m_data[i].factor = p.data.m_data[i].factor;
		if(FAST_SIZE != p.data.m_data[i].fast.size()){throw std::runtime_error("fast too large!");}
		if(SMOOTH_SIZE != p.data.m_data[i].smooth.size()){throw std::runtime_error("smooth too large!");}

		for (int j = 0; j < FAST_SIZE; j++) {
			pre_ptr->m_data[i].fast[j] = p.data.m_data[i].fast[j];
		}
		for (int j = 0; j < SMOOTH_SIZE; j++) {
			pre_ptr->m_data[i].smooth[j][0] = p.data.m_data[i].smooth[j].first;
			pre_ptr->m_data[i].smooth[j][1] = p.data.m_data[i].smooth[j].second;
		}
	}
	size_t pre_size = sizeof(pre_cl);

	// Preparing grid boundries
	gb_cl* gb_ptr = (gb_cl*)malloc(sizeof(gb_cl));
	gb_ptr->dims[0] = ig.m_data.dim0(); gb_ptr->dims[1] = ig.m_data.dim1(); gb_ptr->dims[2] = ig.m_data.dim2();
	for (int i = 0; i < 3; i++)gb_ptr->init[i] = ig.m_init.data[i];
	for (int i = 0; i < 3; i++)gb_ptr->range[i] = ig.m_range.data[i];
	size_t gb_size = sizeof(gb_cl);

	// Preparing atom relationship
	if(MAX_NUM_OF_AR_CELLS < ig.m_data.m_data.size()){throw std::runtime_error("Relation too large! Define a large box (see readme) would help");}
	// assert(ig.m_data.m_i <= 10); assert(ig.m_data.m_j <= 10); assert(ig.m_data.m_k <= 10);
	ar_cl* ar_ptr = (ar_cl*)malloc(sizeof(ar_cl));
	for (int i = 0; i < ig.m_data.m_data.size(); i++) {
		ar_ptr->relation_size[i] = ig.m_data.m_data[i].size();
		if(MAX_NUM_OF_ATOM_RELATION_COUNT < ar_ptr->relation_size[i]){throw std::runtime_error("Relation too large! Define a large box (see readme) would help");}
		for (int j = 0; j < ar_ptr->relation_size[i]; j++) {
			ar_ptr->relation[i][j] = ig.m_data.m_data[i][j];
		}
	}
	size_t ar_size = sizeof(ar_cl);

	// Preparing grid related data
	// QFD: c.grids holds the standard atom-type grids (typically 17); GRIDS_SIZE=21 adds 4 QFD slots.
	// Allow c.grids.size() <= GRIDS_SIZE; extra QFD slots stay zero-filled (m_i=0 → kernel skips them).
	int base_grids = (int)c.grids.size();
	if (base_grids > GRIDS_SIZE) { throw std::runtime_error("grid_size exceeds GRIDS_SIZE=" + std::to_string(GRIDS_SIZE) + "!"); }
	grids_cl* grids_ptr = (grids_cl*)malloc(sizeof(grids_cl)); if (grids_ptr == nullptr) {throw std::runtime_error("Grid too large! Define a large box (see readme) would help");}
	memset(grids_ptr, 0, sizeof(grids_cl)); // zero-fills all slots including QFD slots 17-20
	grid* tmp_grid_ptr = &c.grids[0];

	grids_ptr->atu = c.atu; // atu
	grids_ptr->slope = c.slope; // slope
	int grids_front = 0;
	for (int i = base_grids - 1; i >= 0; i--) {
		for (int j = 0; j < 3; j++) {
			grids_ptr->grids[i].m_init[j] = tmp_grid_ptr[i].m_init[j];
			grids_ptr->grids[i].m_factor[j] = tmp_grid_ptr[i].m_factor[j];
			grids_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grid_ptr[i].m_dim_fl_minus_1[j];
			grids_ptr->grids[i].m_factor_inv[j] = tmp_grid_ptr[i].m_factor_inv[j];
		}
		if (tmp_grid_ptr[i].m_data.dim0() != 0) {
			grids_ptr->grids[i].m_i = tmp_grid_ptr[i].m_data.dim0();
			grids_ptr->grids[i].m_j = tmp_grid_ptr[i].m_data.dim1();
			grids_ptr->grids[i].m_k = tmp_grid_ptr[i].m_data.dim2();
			if(grids_ptr->grids[i].m_i > MAX_NUM_OF_GRID_DIM) throw std::runtime_error("Grid dim i too large! Define a large box (see readme) would help.");
			if(grids_ptr->grids[i].m_j > MAX_NUM_OF_GRID_DIM) throw std::runtime_error("Grid dim j too large! Define a large box (see readme) would help.");
			if(grids_ptr->grids[i].m_k > MAX_NUM_OF_GRID_DIM) throw std::runtime_error("Grid dim k too large! Define a large box (see readme) would help.");
			{
				long long vol = (long long)grids_ptr->grids[i].m_i * grids_ptr->grids[i].m_j * grids_ptr->grids[i].m_k;
				long long max_vol = (long long)MAX_NUM_OF_GRID_MI * MAX_NUM_OF_GRID_MJ * MAX_NUM_OF_GRID_MK;
				if(vol > max_vol) throw std::runtime_error("Grid volume (mi*mj*mk) too large! Use a smaller docking box.");
			}
			grids_front = i;
		}
		// else: m_i/j/k already 0 from memset
	}
	// QFD: optionally load precomputed field grids from CWD (produced by prep_qfd_grids.py).
	// Each load is frame-validated against the current box (g) so a stale qfd_*.bin from a
	// different receptor is ignored rather than silently applied.
	load_qfd_grid_file("qfd_esp.bin",      &grids_ptr->grids[GRID_IDX_ESP],     g);
	load_qfd_grid_file("qfd_desolv.bin",   &grids_ptr->grids[GRID_IDX_DESOLV],  g);
	load_qfd_grid_file("qfd_infomap.bin",  &grids_ptr->grids[GRID_IDX_INFOMAP], g);
	load_qfd_grid_file("qfd_water.bin",    &grids_ptr->grids[GRID_IDX_WATER],   g);  // Phase 3: xtal water penalty
	// Phase 5: pipi — ALWAYS computed from the receptor for the current box (internalized in
	// commit 190cf87). Building unconditionally is correct-by-construction and immune to a stale
	// qfd_pipi.bin in the CWD; the file path is intentionally not consulted here.
	compute_pipi_grid_from_receptor(m, g, &grids_ptr->grids[GRID_IDX_PIPI]);
	// Phase 6: cavity buriedness grid — opt-in via VINA_CAVITY=1 (A/B test before default-on)
	if (getenv("VINA_CAVITY") && atoi(getenv("VINA_CAVITY")) != 0)
	    compute_cavity_grid_from_receptor(m, g, &grids_ptr->grids[GRID_IDX_CAVITY]);
	size_t grids_size = sizeof(grids_cl);

	// miscellaneous
	mis_cl* mis_ptr = (mis_cl*)malloc(sizeof(mis_cl));
	mis_ptr->needed_size = needed.size();
	mis_ptr->epsilon_fl = epsilon_fl;
	mis_ptr->cutoff_sqr = cutoff_sqr;
	mis_ptr->max_fl = max_fl;
	//mis_ptr->torsion_size = tmp.c.ligands[0].torsions.size();
	//mis_ptr->max_bfgs_steps = quasi_newton_par_max_steps;
	//mis_ptr->search_depth = par.mc.search_depth;
	mis_ptr->mutation_amplitude  = par.mc.mutation_amplitude;
	mis_ptr->use_ad4zn           = use_ad4zn ? 1 : 0;
	mis_ptr->flex_torsion_size   = 0;  // Phase 4: 0 = rigid receptor; set non-zero when flex receptor is used
	mis_ptr->total_wi = max_wi_size[0] * max_wi_size[1];
	mis_ptr->thread = par.mc.thread;
	// DIAG: print key parameters so we can verify they are correct
	printf("\nDIAG GPU%d: max_wi_size={%zu,%zu,%zu}  total_wi=%zu  thread=%d\n",
	       gpu_id, max_wi_size[0], max_wi_size[1], max_wi_size[2],
	       mis_ptr->total_wi, mis_ptr->thread);
	printf("DIAG GPU%d: search_depth[0]=%d  bfgs_steps[0]=%d  num_ligands=%d\n",
	       gpu_id,
	       par.mc.search_depth.empty() ? -1 : par.mc.search_depth[0],
	       par.mc.ssd_par.bfgs_steps.empty() ? -1 : par.mc.ssd_par.bfgs_steps[0],
	       num_ligands); fflush(stdout);
	mis_ptr->ar_mi = ig.m_data.m_i;
	mis_ptr->ar_mj = ig.m_data.m_j;
	mis_ptr->ar_mk = ig.m_data.m_k;
	mis_ptr->grids_front = grids_front;
	for (int i = 0; i < 3; i++) mis_ptr->authentic_v[i] = authentic_v[i];
	for (int i = 0; i < 3; i++) mis_ptr->hunt_cap[i] = par.mc.hunt_cap[i];
	size_t mis_size = sizeof(mis_cl);

	float* needed_ptr = (float*)malloc(mis_ptr->needed_size * sizeof(float));
	for (int i = 0; i < mis_ptr->needed_size; i++)needed_ptr[i] = needed[i];

	cl_mem pre_gpu;
	CreateDeviceBuffer(&pre_gpu, CL_MEM_READ_ONLY, pre_size, context);
	err = clEnqueueWriteBuffer(queue, pre_gpu, false, 0, pre_size, pre_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem pa_gpu;
	CreateDeviceBuffer(&pa_gpu, CL_MEM_READ_ONLY, pa_size, context);
	err = clEnqueueWriteBuffer(queue, pa_gpu, false, 0, pa_size, pa_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem gb_gpu;
	CreateDeviceBuffer(&gb_gpu, CL_MEM_READ_ONLY, gb_size, context);
	err = clEnqueueWriteBuffer(queue, gb_gpu, false, 0, gb_size, gb_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem ar_gpu;
	CreateDeviceBuffer(&ar_gpu, CL_MEM_READ_ONLY, ar_size, context);
	err = clEnqueueWriteBuffer(queue, ar_gpu, false, 0, ar_size, ar_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem grids_gpu;
	CreateDeviceBuffer(&grids_gpu, CL_MEM_READ_WRITE, grids_size, context);
	err = clEnqueueWriteBuffer(queue, grids_gpu, false, 0, grids_size, grids_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem needed_gpu;
	CreateDeviceBuffer(&needed_gpu, CL_MEM_READ_WRITE, mis_ptr->needed_size * sizeof(float), context);
	err = clEnqueueWriteBuffer(queue, needed_gpu, false, 0, mis_ptr->needed_size * sizeof(float), needed_ptr, 0, NULL, NULL); checkErr(err);

	cl_mem mis_gpu;
	CreateDeviceBuffer(&mis_gpu, CL_MEM_READ_ONLY, mis_size, context);
	err = clEnqueueWriteBuffer(queue, mis_gpu, false, 0, mis_size, mis_ptr, 0, NULL, NULL); checkErr(err);
	
	clFinish(queue);

	SetKernelArg(kernel1_k, 0, sizeof(cl_mem), &pre_gpu);
	SetKernelArg(kernel1_k, 1, sizeof(cl_mem), &pa_gpu);
	SetKernelArg(kernel1_k, 2, sizeof(cl_mem), &gb_gpu);
	SetKernelArg(kernel1_k, 3, sizeof(cl_mem), &ar_gpu);
	SetKernelArg(kernel1_k, 4, sizeof(cl_mem), &grids_gpu);
	SetKernelArg(kernel1_k, 5, sizeof(cl_mem), &mis_gpu);
	SetKernelArg(kernel1_k, 6, sizeof(cl_mem), &needed_gpu);
	SetKernelArg(kernel1_k, 7, sizeof(int), &c.atu);
	SetKernelArg(kernel1_k, 8, sizeof(int), &nat);


	cl_event kernel1;
	size_t kernel1_global_size[3] = { 128, 128, 128 };
	size_t kernel1_local_size[3] = { 4,4,4 };

	{
		auto _k1_t0 = std::chrono::steady_clock::now();
		err = clEnqueueNDRangeKernel(queue, kernel1_k, 3, 0, kernel1_global_size, kernel1_local_size, 0, NULL, &kernel1); checkErr(err);
		clWaitForEvents(1, &kernel1);
		auto _k1_t1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: kernel1 (grid setup) = %.1fms  sizeof(grids_cl)=%zu bytes\n",
		       gpu_id,
		       std::chrono::duration<double,std::milli>(_k1_t1 - _k1_t0).count(),
		       sizeof(grids_cl)); fflush(stdout);
	}

	free(pa_ptr);
	free(gb_ptr);
	free(ar_ptr);
	free(needed_ptr);
	free(pre_ptr);
	free(grids_ptr);

	err = clReleaseMemObject(pa_gpu);		checkErr(err);
	err = clReleaseMemObject(gb_gpu);		checkErr(err);
	err = clReleaseMemObject(ar_gpu);		checkErr(err);
	err = clReleaseMemObject(needed_gpu);	checkErr(err);

	err = clReleaseEvent(kernel1);			checkErr(err);

	ProteinGpuBuffers __r; __r.pre_gpu=pre_gpu; __r.grids_gpu=grids_gpu; __r.mis_gpu=mis_gpu; __r.mis_ptr=mis_ptr;
	return __r;
}
