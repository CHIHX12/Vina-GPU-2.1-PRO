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

using namespace std;

// Fix: replaced volatile enum (race condition) with std::atomic for thread-safe access
enum class DockingStatus : int { Finish = 0, Docking = 1, Abort = 2 };

// Fix: status is now per-invocation (passed by pointer) so concurrent GPU threads
// do not interfere with each other's progress display.
void print_process(std::atomic<int>* status_ptr) {
	int count = 0;
	printf("\n");
	do
	{
#ifdef WIN32
		Sleep(100);
#else
		sleep(1);
#endif
		printf("\rPerform docking|");
		for (int i = 0; i < count; i++)printf(" ");
		printf("=======");
		for (int i = 0; i < 30 - count; i++)printf(" ");
		printf("|"); fflush(stdout);

		count++;
		count %= 30;
	} while (status_ptr->load(std::memory_order_acquire) == static_cast<int>(DockingStatus::Docking));

	int final_status = status_ptr->load(std::memory_order_acquire);
	if (final_status == static_cast<int>(DockingStatus::Finish)) {
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)printf("=");
		printf("done");
		for (int i = 0; i < 17; i++)printf("=");
		printf("|\n"); fflush(stdout);
	}
	else if (final_status == static_cast<int>(DockingStatus::Abort)) {
		printf("\rPerform docking|");
		for (int i = 0; i < 16; i++)printf("=");
		printf("error");
		for (int i = 0; i < 16; i++)printf("=");
		printf("|\n"); fflush(stdout);
	}
}

// Fix: RAII guard ensures console_thread is always joined even when exceptions are thrown
struct ThreadGuard {
	std::thread& t;
	explicit ThreadGuard(std::thread& t_) : t(t_) {}
	~ThreadGuard() { if (t.joinable()) t.join(); }
	ThreadGuard(const ThreadGuard&) = delete;
	ThreadGuard& operator=(const ThreadGuard&) = delete;
};

std::vector<output_type> cl_to_vina(output_type_cl result_ptr[],
									ligand_atom_coords_cl result_coords_ptr[],
									int thread, int lig_torsion_size,
									const std::vector<int>& flex_tors_counts = std::vector<int>())
{
	std::vector<output_type> results_vina;
	int num_atoms;
	for (int i = 0; i < thread; i++) {
		output_type_cl tmp = result_ptr[i];
		ligand_atom_coords_cl tmp_coords = result_coords_ptr[i];
		conf tmp_c;
		tmp_c.ligands.resize(1);
		// Position
		for (int j = 0; j < 3; j++)tmp_c.ligands[0].rigid.position[j] = tmp.position[j];
		// Orientation
		qt q(tmp.orientation[0], tmp.orientation[1], tmp.orientation[2], tmp.orientation[3]);
		tmp_c.ligands[0].rigid.orientation = q;
		output_type tmp_vina(tmp_c, tmp.e);
		tmp_vina.e_gpu = tmp.e;  // save GPU total (Vina+QFD) before CPU re-scoring overwrites .e
		// torsion
		for (int j = 0; j < lig_torsion_size; j++)tmp_vina.c.ligands[0].torsions.push_back(tmp.lig_torsion[j]);
		// flex side-chain torsions (Stage 1.5): rebuild conf.flex matching the model's residues
		// so m.set(out.c) during output writing has the correct structure (else out-of-range crash).
		{
			int ft = 0;
			for (int r = 0; r < (int)flex_tors_counts.size(); r++) {
				residue_conf rc;
				for (int t = 0; t < flex_tors_counts[r]; t++)
					rc.torsions.push_back(tmp.flex_torsion[ft++]);
				tmp_vina.c.flex.push_back(rc);
			}
		}
		// coords
		for (int j = 0; j < MAX_NUM_OF_ATOMS; j++) {
			vec v_tmp(tmp_coords.coords[j][0], tmp_coords.coords[j][1], tmp_coords.coords[j][2]);
			if ((v_tmp[0] != 0 || v_tmp[1] != 0) || (v_tmp[2] != 0)) tmp_vina.coords.push_back(v_tmp);
		}
		results_vina.push_back(tmp_vina);
		if (i == 0)num_atoms = tmp_vina.coords.size();
		if(num_atoms != tmp_vina.coords.size()){throw std::runtime_error("atom coords not match!");}
	}
	return results_vina;
}

// QFD grid binary format (written by scripts/prep_qfd_grids.py):
//   int[3]   : m_i, m_j, m_k
//   float[12]: m_init[3], m_factor[3], m_dim_fl_minus_1[3], m_factor_inv[3]
//   float[m_i*m_j*m_k*8]: trilinear interpolation coefficients
// Returns true and fills slot if loaded; returns false and leaves slot zeroed if file absent.
static bool load_qfd_grid_file(const std::string& path, grid_cl* slot) {
	FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	int dims[3];
	float meta[12];
	if (std::fread(dims, sizeof(int), 3, f) != 3 || std::fread(meta, sizeof(float), 12, f) != 12) {
		std::fclose(f); return false;
	}
	int mi = dims[0], mj = dims[1], mk = dims[2];
	if (mi <= 0 || mj <= 0 || mk <= 0 || mi > MAX_NUM_OF_GRID_DIM || mj > MAX_NUM_OF_GRID_DIM || mk > MAX_NUM_OF_GRID_DIM) {
		std::fclose(f); return false;
	}
	size_t ndata = (size_t)mi * mj * mk * 8;
	if (std::fread(slot->m_data, sizeof(float), ndata, f) != ndata) {
		std::fclose(f); return false;
	}
	std::fclose(f);
	slot->m_i = mi; slot->m_j = mj; slot->m_k = mk;
	std::memcpy(slot->m_init,            meta + 0, 3 * sizeof(float));
	std::memcpy(slot->m_factor,          meta + 3, 3 * sizeof(float));
	std::memcpy(slot->m_dim_fl_minus_1,  meta + 6, 3 * sizeof(float));
	std::memcpy(slot->m_factor_inv,      meta + 9, 3 * sizeof(float));
	printf("QFD: loaded %s  dims=%d×%d×%d\n", path.c_str(), mi, mj, mk);
	return true;
}

// LS metal coordination rescoring (Part A).
// Constants mirror kernel2.h — defined here so they're available in C++ code.
static const float LS_METAL_R_OPT_CPP  = 2.10f; // optimal coordination distance [Å]
static const float LS_METAL_SIGMA_CPP  = 0.20f; // tight Gaussian width [Å]
static const float LS_METAL_CUTOFF_CPP = 5.0f;  // per-metal search radius [Å]

// AD type range for metals: Mg=13 .. Hg=31 (see atom_constants.h)
static bool is_receptor_metal(sz ad_type) {
    return ad_type >= 13 && ad_type < 32;
}

// Collect receptor metal atom positions from model's grid_atoms.
static std::vector<std::array<float,3>> collect_receptor_metals(const model& m) {
    std::vector<std::array<float,3>> metals;
    for (sz i = 0; i < m.grid_atoms.size(); i++) {
        if (is_receptor_metal(m.grid_atoms[i].ad)) {
            metals.push_back({(float)m.grid_atoms[i].coords[0],
                              (float)m.grid_atoms[i].coords[1],
                              (float)m.grid_atoms[i].coords[2]});
        }
    }
    return metals;
}

// Apply LS metal coordination bonus to all poses in-place.
// Scoring: sum_metals max_ligAtoms G(dist; r_opt, sigma)
//   — max-per-metal prevents multi-atom accumulation on wrong poses
// pose.e is adjusted: pose.e -= weight * e_ls  (lower e = better, bonus lowers e)
static void apply_ls_metal_scores(
    std::vector<output_type>& poses,
    const std::vector<std::array<float,3>>& metals,
    float weight)
{
    if (metals.empty() || weight == 0.0f) return;
    const float cutoff2   = LS_METAL_CUTOFF_CPP * LS_METAL_CUTOFF_CPP;
    const float inv2sig2  = 1.0f / (2.0f * LS_METAL_SIGMA_CPP * LS_METAL_SIGMA_CPP);
    for (auto& pose : poses) {
        float bonus = 0.0f;
        for (const auto& metal : metals) {
            float mx = metal[0], my = metal[1], mz = metal[2];
            float best_g = 0.0f;   // best single-atom G for this metal
            for (const auto& coord : pose.coords) {
                float dx = (float)coord[0] - mx;
                float dy = (float)coord[1] - my;
                float dz = (float)coord[2] - mz;
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < cutoff2) {
                    float r  = std::sqrt(r2);
                    float dr = r - LS_METAL_R_OPT_CPP;
                    float g  = std::exp(-dr * dr * inv2sig2);
                    if (g > best_g) best_g = g;
                }
            }
            bonus += best_g;   // one coordination bond per metal
        }
        pose.e_ls  = bonus;           // raw coordination score (positive = good)
        pose.e    -= weight * bonus;  // subtract from energy (lower = better)
    }
}

// ── QFD pipi grid: built directly from receptor aromatic carbons ────────────
// Eliminates the need to run prep_qfd_grids.py before docking.
// AD_TYPE_A = 1 (aromatic carbon, see atom_constants.h).
// Matches the Python formula in prep_qfd_grids.py compute_pipi_grid().
static const float PIPI_SIGMA_CPP   = 2.5f;  // Gaussian width [Å]
static const float PIPI_SPACING_CPP = 1.0f;  // grid spacing [Å]

static bool compute_pipi_grid_from_receptor(
    const model& m,
    const grid& ref_grid,   // any initialized Vina atom-type grid (box reference)
    grid_cl* slot)
{
    // Collect aromatic carbon positions (AD_TYPE_A = 1)
    std::vector<std::array<float,3>> aroms;
    for (sz i = 0; i < m.grid_atoms.size(); i++) {
        if (m.grid_atoms[i].ad == 1) {  // AD_TYPE_A
            aroms.push_back({(float)m.grid_atoms[i].coords[0],
                             (float)m.grid_atoms[i].coords[1],
                             (float)m.grid_atoms[i].coords[2]});
        }
    }
    if (aroms.empty()) return false;

    // Derive box centre and size from the reference Vina grid
    const float bs  = (float)ref_grid.m_factor_inv[0];          // base Vina spacing [Å]
    const float bmi = (float)ref_grid.m_data.dim0() - 1.0f;
    const float bmj = (float)ref_grid.m_data.dim1() - 1.0f;
    const float bmk = (float)ref_grid.m_data.dim2() - 1.0f;
    if (bmi <= 0.0f || bmj <= 0.0f || bmk <= 0.0f) return false;

    const float cx = (float)ref_grid.m_init[0] + bmi * 0.5f * bs;
    const float cy = (float)ref_grid.m_init[1] + bmj * 0.5f * bs;
    const float cz = (float)ref_grid.m_init[2] + bmk * 0.5f * bs;
    const float sx = bmi * bs;
    const float sy = bmj * bs;
    const float sz = bmk * bs;

    // QFD pipi grid at PIPI_SPACING_CPP = 1.0 Å
    const float sp = PIPI_SPACING_CPP;
    int mi = (int)std::round(sx / sp) + 1;
    int mj = (int)std::round(sy / sp) + 1;
    int mk = (int)std::round(sz / sp) + 1;
    if (mi > MAX_NUM_OF_GRID_MI) mi = MAX_NUM_OF_GRID_MI;
    if (mj > MAX_NUM_OF_GRID_MJ) mj = MAX_NUM_OF_GRID_MJ;
    if (mk > MAX_NUM_OF_GRID_MK) mk = MAX_NUM_OF_GRID_MK;

    // Grid origin (lower-left corner centred on the box)
    const float ox = cx - (mi - 1) * 0.5f * sp;
    const float oy = cy - (mj - 1) * 0.5f * sp;
    const float oz = cz - (mk - 1) * 0.5f * sp;

    // Raw Gaussian sum: P(r) = Σ_aroms exp(-r²/(2σ²))
    const float inv_2sig2 = 1.0f / (2.0f * PIPI_SIGMA_CPP * PIPI_SIGMA_CPP);
    std::vector<float> raw((size_t)mi * mj * mk, 0.0f);

    for (int i = 0; i < mi; i++) {
        const float gx = ox + i * sp;
        for (int j = 0; j < mj; j++) {
            const float gy = oy + j * sp;
            for (int k = 0; k < mk; k++) {
                const float gz = oz + k * sp;
                float v = 0.0f;
                for (const auto& a : aroms) {
                    const float dx = gx - a[0], dy = gy - a[1], dz = gz - a[2];
                    v += std::exp(-(dx*dx + dy*dy + dz*dz) * inv_2sig2);
                }
                raw[(size_t)(i * mj + j) * mk + k] = v;
            }
        }
    }

    // Normalize to [0, 1]  (QFD_PIPI_WEIGHT controls absolute scale)
    const float vmax = *std::max_element(raw.begin(), raw.end());
    if (vmax <= 0.0f) return false;
    for (auto& v : raw) v /= vmax;

    // Pack into Vina's trilinear corner format: 8 corner values per voxel
    auto vget = [&](int i, int j, int k) -> float {
        i = std::max(0, std::min(mi - 1, i));
        j = std::max(0, std::min(mj - 1, j));
        k = std::max(0, std::min(mk - 1, k));
        return raw[(size_t)(i * mj + j) * mk + k];
    };
    for (int i = 0; i < mi; i++) {
        for (int j = 0; j < mj; j++) {
            for (int k = 0; k < mk; k++) {
                const size_t base = ((size_t)(i * mj + j) * mk + k) * 8;
                slot->m_data[base+0] = vget(i,   j,   k  );
                slot->m_data[base+1] = vget(i+1, j,   k  );
                slot->m_data[base+2] = vget(i,   j+1, k  );
                slot->m_data[base+3] = vget(i+1, j+1, k  );
                slot->m_data[base+4] = vget(i,   j,   k+1);
                slot->m_data[base+5] = vget(i+1, j,   k+1);
                slot->m_data[base+6] = vget(i,   j+1, k+1);
                slot->m_data[base+7] = vget(i+1, j+1, k+1);
            }
        }
    }

    // Fill grid_cl metadata
    slot->m_i = mi; slot->m_j = mj; slot->m_k = mk;
    slot->m_init[0] = ox; slot->m_init[1] = oy; slot->m_init[2] = oz;
    const float inv_sp = 1.0f / sp;
    slot->m_factor[0]         = slot->m_factor[1]         = slot->m_factor[2]         = inv_sp;
    slot->m_dim_fl_minus_1[0] = (float)(mi - 1);
    slot->m_dim_fl_minus_1[1] = (float)(mj - 1);
    slot->m_dim_fl_minus_1[2] = (float)(mk - 1);
    slot->m_factor_inv[0]     = slot->m_factor_inv[1]     = slot->m_factor_inv[2]     = sp;

    printf("QFD pipi: built from %zu aromatic C atoms  dims=%d×%d×%d\n",
           aroms.size(), mi, mj, mk);
    return true;
}

// Replica Exchange: Metropolis swap between adjacent SA replicas.
// Called after each kernel2 epoch. result_ptrs[li] is the result for ligand li,
// with thread entries sorted by traj_id. rep_id = traj_id % QFD_N_REPLICAS.
// T_r = QFD_T_START_MIN * pow(QFD_T_START_MAX/QFD_T_START_MIN, r/(QFD_N_REPLICAS-1))
static void apply_re_swaps(
    std::vector<output_type_cl*>& result_ptrs,
    int num_ligands, int thread, int epoch, std::mt19937& rng_re
) {
    auto T_rep = [](int r) -> float {
        return QFD_T_START_MIN * std::pow(QFD_T_START_MAX / QFD_T_START_MIN,
                                          (float)r / (float)(QFD_N_REPLICAS - 1));
    };
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    // Alternating sweeps: even epoch → pairs (0,1),(2,3),(4,5),(6,7); odd → (1,2),(3,4),(5,6)
    int start_r = (epoch % 2 == 0) ? 0 : 1;
    for (int li = 0; li < num_ligands; li++) {
        if (!result_ptrs[li]) continue;
        output_type_cl* res = result_ptrs[li];
        int n_super = thread / QFD_N_REPLICAS;
        for (int s = 0; s < n_super; s++) {
            for (int r = start_r; r < QFD_N_REPLICAS - 1; r += 2) {
                int traj_a = s * QFD_N_REPLICAS + r;
                int traj_b = s * QFD_N_REPLICAS + r + 1;
                if (traj_a >= thread || traj_b >= thread) continue;
                float E_a = res[traj_a].e;
                float E_b = res[traj_b].e;
                float T_a = T_rep(r);
                float T_b = T_rep(r + 1);
                // RE criterion: Δ = (1/T_b - 1/T_a)(E_a - E_b)
                float delta = (1.0f / T_b - 1.0f / T_a) * (E_a - E_b);
                if (delta >= 0.0f || uni(rng_re) < std::exp(delta)) {
                    std::swap(res[traj_a], res[traj_b]);
                }
            }
        }
    }
}

// Differential-evolution inter-trajectory exchange (algorithm A, gated by VINA_DE).
// Between Replica-Exchange epochs, recombine the population of `thread` confs so good
// torsion sub-solutions found by different trajectories get combined (DE/rand/1/bin on
// position + ligand torsions). The next epoch's MC+BFGS refines the DE-seeded starts.
// Light elitism: the single best trajectory is preserved. Addresses the "8000 independent
// trajectories" waste; helps SAMPLING-limited ligands (not scoring-limited ones).
static void apply_de_exchange(
    std::vector<output_type_cl*>& result_ptrs, const std::vector<int>& torsion_sizes,
    int num_ligands, int thread, std::mt19937& rng
) {
    const float F = 0.6f, CR = 0.9f;
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, thread - 1);
    for (int li = 0; li < num_ligands; li++) {
        output_type_cl* res = result_ptrs[li];
        if (!res || thread < 4) continue;
        int nt = (li < (int)torsion_sizes.size()) ? torsion_sizes[li] : 0;
        int best = 0;
        for (int i = 1; i < thread; i++) if (res[i].e < res[best].e) best = i;
        std::vector<output_type_cl> trial(res, res + thread);
        for (int i = 0; i < thread; i++) {
            if (i == best) continue;                       // elitism: keep global best
            int r1, r2, r3;
            do { r1 = pick(rng); } while (r1 == i);
            do { r2 = pick(rng); } while (r2 == i || r2 == r1);
            do { r3 = pick(rng); } while (r3 == i || r3 == r1 || r3 == r2);
            output_type_cl t = res[i];
            for (int k = 0; k < 3; k++)
                if (uni(rng) < CR) t.position[k] = res[r1].position[k] + F * (res[r2].position[k] - res[r3].position[k]);
            for (int k = 0; k < nt; k++)
                if (uni(rng) < CR) {
                    float v = res[r1].lig_torsion[k] + F * (res[r2].lig_torsion[k] - res[r3].lig_torsion[k]);
                    while (v >  (float)M_PI) v -= 2.0f * (float)M_PI;
                    while (v < -(float)M_PI) v += 2.0f * (float)M_PI;
                    t.lig_torsion[k] = v;
                }
            if (uni(rng) < CR) for (int k = 0; k < 4; k++) t.orientation[k] = res[r1].orientation[k];
            trial[i] = t;
        }
        for (int i = 0; i < thread; i++) res[i] = trial[i];
    }
}

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
	cl_platform_id* platforms;
	cl_device_id* devices;
	cl_context context;
	cl_command_queue queue;
	cl_int gpu_platform_id = 0;
	SetupPlatform(&platforms, &gpu_platform_id);
	SetupDevice(platforms, &devices, gpu_platform_id);
	printf("Using GPU device index: %d\n", gpu_id);
	// Fix: context must be created with the exact target device, not always devices[0]
	SetupContext(platforms, devices + gpu_id, &context, 1, gpu_platform_id);
	SetupQueue(&queue, context, devices, gpu_id);

	cl_program programs[2];

	//printf("\nSearch depth is set to %d", par.mc.search_depth);
	printf("\nUsing random seed: %d", seed);

#ifdef BUILD_KERNEL_FROM_SOURCE
{
	// Fix: serialize kernel compilation across GPU threads — only one thread compiles
	// and saves the .bin files; others wait and then load the cached result.
	static std::mutex kernel_compile_mutex;
	std::lock_guard<std::mutex> compile_lock(kernel_compile_mutex);

	// Compiled .bin files live in opencl_binary_path (writable host dir, default ".")
	const std::string bin_out_path = opencl_binary_path.empty() ? "." : opencl_binary_path;
	const std::string bin1 = bin_out_path + "/Kernel1_Opt.bin";
	const std::string bin2 = bin_out_path + "/Kernel2_Opt.bin";

	// Check if cached .bin files already exist for this machine's GPU
	auto file_exists = [](const std::string& p) {
		std::ifstream f(p); return f.good();
	};

	if (file_exists(bin1) && file_exists(bin2)) {
		printf("\nLoading cached GPU kernels from %s\n", bin_out_path.c_str()); fflush(stdout);
	} else {
		// First run on this GPU: compile from source and cache the .bin files
		// VINA_GPU_HOME is set in the SIF's ENV; falls back to "." for local builds
		const char* vina_home_env = std::getenv("VINA_GPU_HOME");
		const std::string kernel_src_path = vina_home_env ? std::string(vina_home_env) : ".";
		const std::string include_path = kernel_src_path + "/OpenCL/inc";
		const std::string addtion = "";

		printf("\n\nCompiling GPU kernels for this machine (one-time, caching to %s)...\n",
		       bin_out_path.c_str()); fflush(stdout);

		// --- Kernel 1 ---
		printf("  Building kernel 1 from source..."); fflush(stdout);
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
		printf(" done\n"); fflush(stdout);

		// --- Kernel 2 ---
		printf("  Building kernel 2 from source..."); fflush(stdout);
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
		printf(" done\n\n"); fflush(stdout);
	}
}
#endif
	// Fix: pass devices+gpu_id so the program is built for the target GPU, not always devices[0]
	programs[0] = SetupBuildProgramWithBinary(context, devices + gpu_id, (opencl_binary_path + std::string("/Kernel1_Opt.bin")).c_str());

	programs[1] = SetupBuildProgramWithBinary(context, devices + gpu_id, (opencl_binary_path + std::string("/Kernel2_Opt.bin")).c_str());

	err = clUnloadPlatformCompiler(platforms[gpu_platform_id]); checkErr(err);
	//Set kernel arguments
	cl_kernel kernels[2];
	char kernel_name[][50] = { "kernel1","kernel2"};
	SetupKernel(kernels, programs, 2, kernel_name);

	size_t max_wg_size; // max work item within one work group
	size_t max_wi_size[3]; // max work item within each dimension(global)
	// Fix: query the target GPU (devices[gpu_id]), not always devices[0]
	err = clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL); checkErr(err);
	err = clGetDeviceInfo(devices[gpu_id], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(size_t), &max_wi_size, NULL); checkErr(err);

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
	// QFD: optionally load precomputed field grids from CWD (produced by prep_qfd_grids.py)
	load_qfd_grid_file("qfd_esp.bin",      &grids_ptr->grids[GRID_IDX_ESP]);
	load_qfd_grid_file("qfd_desolv.bin",   &grids_ptr->grids[GRID_IDX_DESOLV]);
	load_qfd_grid_file("qfd_infomap.bin",  &grids_ptr->grids[GRID_IDX_INFOMAP]);
	load_qfd_grid_file("qfd_water.bin",    &grids_ptr->grids[GRID_IDX_WATER]);    // Phase 3: xtal water penalty
	// Phase 5: pipi — load from file if available, else compute from receptor (always active)
	if (!load_qfd_grid_file("qfd_pipi.bin", &grids_ptr->grids[GRID_IDX_PIPI]))
	    compute_pipi_grid_from_receptor(m, g, &grids_ptr->grids[GRID_IDX_PIPI]);
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
	       (int)ms.size()); fflush(stdout);
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

	SetKernelArg(kernels[0], 0, sizeof(cl_mem), &pre_gpu);
	SetKernelArg(kernels[0], 1, sizeof(cl_mem), &pa_gpu);
	SetKernelArg(kernels[0], 2, sizeof(cl_mem), &gb_gpu);
	SetKernelArg(kernels[0], 3, sizeof(cl_mem), &ar_gpu);
	SetKernelArg(kernels[0], 4, sizeof(cl_mem), &grids_gpu);
	SetKernelArg(kernels[0], 5, sizeof(cl_mem), &mis_gpu);
	SetKernelArg(kernels[0], 6, sizeof(cl_mem), &needed_gpu);
	SetKernelArg(kernels[0], 7, sizeof(int), &c.atu);
	SetKernelArg(kernels[0], 8, sizeof(int), &nat);

	// Fix: per-invocation status — each GPU thread has its own, so they don't
	// interfere when running concurrently (e.g., one GPU finishing doesn't stop
	// the other GPU's progress display).
	std::atomic<int> local_status{static_cast<int>(DockingStatus::Finish)};
	local_status.store(static_cast<int>(DockingStatus::Docking), std::memory_order_release);
# ifdef NDEBUG
	std::thread console_thread(print_process, &local_status);
	ThreadGuard console_guard(console_thread); // Fix: RAII ensures join on any exit path
# endif

	cl_event kernel1;
	size_t kernel1_global_size[3] = { 128, 128, 128 };
	size_t kernel1_local_size[3] = { 4,4,4 };

	{
		auto _k1_t0 = std::chrono::steady_clock::now();
		err = clEnqueueNDRangeKernel(queue, kernels[0], 3, 0, kernel1_global_size, kernel1_local_size, 0, NULL, &kernel1); checkErr(err);
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
					m_ptr->flex_rigid.num_nodes = 0;   // Stage 1: flex forest empty until populated
					m_ptr->other_pairs.num_pairs = 0;  // Stage 2: ligand↔flex pairs empty until populated
					for (int ai = 0; ai < (int)mi.atoms.size(); ai++) {
						m_ptr->atoms[ai].types[0] = mi.atoms[ai].el;
						m_ptr->atoms[ai].types[1] = mi.atoms[ai].ad;
						m_ptr->atoms[ai].types[2] = mi.atoms[ai].xs;
						m_ptr->atoms[ai].types[3] = mi.atoms[ai].sy;
						for (int j = 0; j < 3; j++) m_ptr->atoms[ai].coords[j] = mi.atoms[ai].coords[j];
						m_ptr->atoms[ai].charge = (float)mi.atoms[ai].charge; // QFD: partial charge [e]
					}
					for (int ci = 0; ci < (int)mi.coords.size(); ci++)
						for (int j = 0; j < 3; j++) m_ptr->m_coords.coords[ci][j] = mi.coords[ci].data[j];
					for (int ci = 0; ci < (int)mi.coords.size(); ci++)
						for (int j = 0; j < 3; j++) m_ptr->minus_forces.coords[ci][j] = mi.minus_forces[ci].data[j];

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

					ligand m_ligand = mi.ligands[0];
					if (m_ligand.end >= MAX_NUM_OF_ATOMS) { throw std::runtime_error("Ligand too large! The maximum number of atoms is " +
					 std::to_string(MAX_NUM_OF_ATOMS) + " and the ligand has " + std::to_string(m_ligand.end) + " atoms."); }
					m_ptr->ligand.pairs.num_pairs = mi.ligands[0].pairs.size();
					if (mi.ligands[0].pairs.size() >= MAX_NUM_OF_LIG_PAIRS) { throw std::runtime_error("Ligand too large! The maximum number of pairs is " +
					 std::to_string(MAX_NUM_OF_LIG_PAIRS) + " and the ligand has " + std::to_string(mi.ligands[0].pairs.size()) + " pairs."); }
					for (int pi = 0; pi < m_ptr->ligand.pairs.num_pairs; pi++) {
						m_ptr->ligand.pairs.type_pair_index[pi] = mi.ligands[0].pairs[pi].type_pair_index;
						m_ptr->ligand.pairs.a[pi] = mi.ligands[0].pairs[pi].a;
						m_ptr->ligand.pairs.b[pi] = mi.ligands[0].pairs[pi].b;
					}
					m_ptr->ligand.begin = mi.ligands[0].begin;
					m_ptr->ligand.end   = mi.ligands[0].end;
					m_ptr->ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
					m_ptr->ligand.rigid.atom_range[0][1] = m_ligand.node.end;
					for (int j = 0; j < 3; j++) m_ptr->ligand.rigid.origin[0][j] = m_ligand.node.get_origin()[j];
					for (int j = 0; j < 9; j++) m_ptr->ligand.rigid.orientation_m[0][j] = m_ligand.node.get_orientation_m().data[j];
					m_ptr->ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
					m_ptr->ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
					m_ptr->ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
					m_ptr->ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();
					for (int j = 0; j < 3; j++) { m_ptr->ligand.rigid.axis[0][j] = 0; m_ptr->ligand.rigid.relative_axis[0][j] = 0; m_ptr->ligand.rigid.relative_origin[0][j] = 0; }
					struct tmp_struct_local {
						int start_index = 0;
						int parent_index = 0;
						void store_node(tree<segment>& child_ptr, rigid_cl& rigid) {
							start_index++;
							rigid.parent[start_index] = parent_index;
							rigid.atom_range[start_index][0] = child_ptr.node.begin;
							rigid.atom_range[start_index][1] = child_ptr.node.end;
							for (int j = 0; j < 9; j++) rigid.orientation_m[start_index][j] = child_ptr.node.get_orientation_m().data[j];
							rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
							rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
							rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
							rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
							for (int j = 0; j < 3; j++) {
								rigid.origin[start_index][j]          = child_ptr.node.get_origin()[j];
								rigid.axis[start_index][j]            = child_ptr.node.get_axis()[j];
								rigid.relative_axis[start_index][j]   = child_ptr.node.relative_axis[j];
								rigid.relative_origin[start_index][j] = child_ptr.node.relative_origin[j];
							}
							if (child_ptr.children.empty()) return;
							if (start_index >= MAX_NUM_OF_RIGID) { throw std::runtime_error("Children map too large!"); }
							int parent_index_tmp = start_index;
							for (int ci = 0; ci < (int)child_ptr.children.size(); ci++) {
								this->parent_index = parent_index_tmp;
								this->store_node(child_ptr.children[ci], rigid);
							}
						}
					};
					tmp_struct_local ts;
					for (int ci = 0; ci < (int)m_ligand.children.size(); ci++) {
						ts.parent_index = 0;
						ts.store_node(m_ligand.children[ci], m_ptr->ligand.rigid);
					}
					m_ptr->ligand.rigid.num_children = ts.start_index;
					m_ptr->ligand.rigid.parent[0] = -1;  // root has no parent (children derived from parent[])

					// ───── Stage 1: populate flexible side-chain forest from mi.flex ─────
					// Each residue (heterotree<first_segment>) → one root node (parent=-1,
					// first_segment, consumes residue torsions[0]) + DFS child segments.
					// torsion_idx counts flex torsions in conf-flattening order (residue, then DFS).
					{
						flex_rigid_cl& fr = m_ptr->flex_rigid;
						struct flex_store {
							void store_seg(tree<segment>& t, int parent, int& ni, int& ft, flex_rigid_cl& fr) {
								int me = ni++;
								if (me >= MAX_NUM_OF_FLEX_RIGID) throw std::runtime_error("Too many flex nodes (raise MAX_NUM_OF_FLEX_RIGID)!");
								fr.parent[me] = parent;
								fr.torsion_idx[me] = ft++;
								fr.atom_range[me][0] = t.node.begin;
								fr.atom_range[me][1] = t.node.end;
								for (int j = 0; j < 9; j++) fr.orientation_m[me][j] = t.node.get_orientation_m().data[j];
								fr.orientation_q[me][0] = t.node.orientation().R_component_1();
								fr.orientation_q[me][1] = t.node.orientation().R_component_2();
								fr.orientation_q[me][2] = t.node.orientation().R_component_3();
								fr.orientation_q[me][3] = t.node.orientation().R_component_4();
								for (int j = 0; j < 3; j++) {
									fr.origin[me][j]          = t.node.get_origin()[j];
									fr.axis[me][j]            = t.node.get_axis()[j];
									fr.relative_axis[me][j]   = t.node.relative_axis[j];
									fr.relative_origin[me][j] = t.node.relative_origin[j];
								}
								for (int ci = 0; ci < (int)t.children.size(); ci++)
									store_seg(t.children[ci], me, ni, ft, fr);
							}
						};
						flex_store fs;
						int ni = 0, ft = 0;
						vector_mutable<residue>& flex_res = mi.get_flex();
						for (int ri = 0; ri < (int)flex_res.size(); ri++) {
							residue& res = flex_res[ri];
							int me = ni++;
							if (me >= MAX_NUM_OF_FLEX_RIGID) throw std::runtime_error("Too many flex nodes (raise MAX_NUM_OF_FLEX_RIGID)!");
							fr.parent[me]      = -1;        // first_segment root (rotates about its fixed axis)
							fr.torsion_idx[me] = ft++;      // residue torsions[0]
							fr.atom_range[me][0] = res.node.begin;
							fr.atom_range[me][1] = res.node.end;
							for (int j = 0; j < 9; j++) fr.orientation_m[me][j] = res.node.get_orientation_m().data[j];
							fr.orientation_q[me][0] = res.node.orientation().R_component_1();
							fr.orientation_q[me][1] = res.node.orientation().R_component_2();
							fr.orientation_q[me][2] = res.node.orientation().R_component_3();
							fr.orientation_q[me][3] = res.node.orientation().R_component_4();
							for (int j = 0; j < 3; j++) {
								fr.origin[me][j]          = res.node.get_origin()[j];
								fr.axis[me][j]            = res.node.get_axis()[j];
								fr.relative_axis[me][j]   = 0.0f;   // root: unused (no parent)
								fr.relative_origin[me][j] = 0.0f;
							}
							for (int ci = 0; ci < (int)res.children.size(); ci++)
								fs.store_seg(res.children[ci], me, ni, ft, fr);
						}
						fr.num_nodes = ni;
						if (ft > MAX_NUM_OF_FLEX_TORSION) throw std::runtime_error("Too many flex torsions (raise MAX_NUM_OF_FLEX_TORSION)!");
					}

					// ───── Stage 2: populate other_pairs (ligand↔flex + flex↔flex) ─────
					{
						const interacting_pairs& op = mi.get_other_pairs();
						if ((int)op.size() >= MAX_NUM_OF_OTHER_PAIRS) throw std::runtime_error("Too many other_pairs (raise MAX_NUM_OF_OTHER_PAIRS)!");
						m_ptr->other_pairs.num_pairs = (int)op.size();
						for (int pi = 0; pi < (int)op.size(); pi++) {
							m_ptr->other_pairs.type_pair_index[pi] = (int)op[pi].type_pair_index;
							m_ptr->other_pairs.a[pi] = (int)op[pi].a;
							m_ptr->other_pairs.b[pi] = (int)op[pi].b;
						}
					}

					m_ptr->m_num_movable_atoms = mi.num_movable_atoms();

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

	// Build mis_cl — use thread from par.mc.thread
	mis_cl* mis_ptr = (mis_cl*)malloc(sizeof(mis_cl));
	mis_ptr->needed_size      = needed.size();
	mis_ptr->epsilon_fl       = epsilon_fl;
	mis_ptr->cutoff_sqr       = cutoff_sqr;
	mis_ptr->max_fl           = max_fl;
	mis_ptr->mutation_amplitude  = par.mc.mutation_amplitude;
	mis_ptr->use_ad4zn           = use_ad4zn ? 1 : 0;
	mis_ptr->flex_torsion_size   = 0;  // Phase 4: rigid receptor default
	mis_ptr->total_wi            = max_wi_size[0] * max_wi_size[1];
	mis_ptr->thread           = par.mc.thread;
	mis_ptr->ar_mi            = ig.m_data.m_i;
	mis_ptr->ar_mj            = ig.m_data.m_j;
	mis_ptr->ar_mk            = ig.m_data.m_k;
	for (int i = 0; i < 3; i++) mis_ptr->authentic_v[i] = authentic_v[i];
	for (int i = 0; i < 3; i++) mis_ptr->hunt_cap[i]    = par.mc.hunt_cap[i];

	// Protein atoms
	pa_cl* pa_ptr = (pa_cl*)malloc(sizeof(pa_cl));
	if (MAX_NUM_OF_PROTEIN_ATOMS <= ma.grid_atoms.size()) throw std::runtime_error("pocket too large!");
	for (int i = 0; i < (int)ma.grid_atoms.size(); i++) {
		pa_ptr->atoms[i].types[0] = ma.grid_atoms[i].el;
		pa_ptr->atoms[i].types[1] = ma.grid_atoms[i].ad;
		pa_ptr->atoms[i].types[2] = ma.grid_atoms[i].xs;
		pa_ptr->atoms[i].types[3] = ma.grid_atoms[i].sy;
		for (int j = 0; j < 3; j++) pa_ptr->atoms[i].coords[j] = ma.grid_atoms[i].coords.data[j];
	}

	// LS metal rescoring for dual mode
	auto dual_receptor_metals = collect_receptor_metals(ma);
	float dual_ls_metal_weight = 0.30f;
	{
		const char* env = std::getenv("VINA_LS_METAL_WEIGHT");
		if (env) dual_ls_metal_weight = (float)std::atof(env);
	}
	if (!dual_receptor_metals.empty())
		printf("QFD LS (dual): %zu receptor metal(s) found (weight=%.3f)\n",
		       dual_receptor_metals.size(), dual_ls_metal_weight);

	// Precalculate LUT
	pre_cl* pre_ptr = (pre_cl*)malloc(sizeof(pre_cl));
	pre_ptr->m_cutoff_sqr = p.cutoff_sqr();
	pre_ptr->factor = p.factor;
	pre_ptr->n = p.n;
	if (MAX_P_DATA_M_DATA_SIZE <= p.data.m_data.size()) throw std::runtime_error("LUT too large!");
	for (int i = 0; i < (int)p.data.m_data.size(); i++) {
		pre_ptr->m_data[i].factor = p.data.m_data[i].factor;
		for (int j = 0; j < FAST_SIZE; j++)   pre_ptr->m_data[i].fast[j]       = p.data.m_data[i].fast[j];
		for (int j = 0; j < SMOOTH_SIZE; j++) { pre_ptr->m_data[i].smooth[j][0] = p.data.m_data[i].smooth[j].first;
		                                         pre_ptr->m_data[i].smooth[j][1] = p.data.m_data[i].smooth[j].second; }
	}

	// Grid boundaries
	gb_cl* gb_ptr = (gb_cl*)malloc(sizeof(gb_cl));
	gb_ptr->dims[0] = ig.m_data.dim0(); gb_ptr->dims[1] = ig.m_data.dim1(); gb_ptr->dims[2] = ig.m_data.dim2();
	for (int i = 0; i < 3; i++) gb_ptr->init[i]  = ig.m_init.data[i];
	for (int i = 0; i < 3; i++) gb_ptr->range[i] = ig.m_range.data[i];

	// Atom relation map
	if (MAX_NUM_OF_AR_CELLS < ig.m_data.m_data.size()) throw std::runtime_error("Relation too large!");
	ar_cl* ar_ptr = (ar_cl*)malloc(sizeof(ar_cl));
	for (int i = 0; i < (int)ig.m_data.m_data.size(); i++) {
		ar_ptr->relation_size[i] = ig.m_data.m_data[i].size();
		for (int j = 0; j < (int)ar_ptr->relation_size[i]; j++)
			ar_ptr->relation[i][j] = ig.m_data.m_data[i][j];
	}

	// Grids
	// QFD: GRIDS_SIZE=21 adds slots 17-20 for QFD grids. c.grids has standard atom-type grids (≤17).
	// Allow c.grids.size() <= GRIDS_SIZE; zero-fill extra QFD slots so kernel skips them (m_i=0).
	{
		int bg = (int)c.grids.size();
		if (bg > GRIDS_SIZE) throw std::runtime_error("grid_size exceeds GRIDS_SIZE=" + std::to_string(GRIDS_SIZE) + "!");
	}
	grids_cl* grids_ptr = (grids_cl*)malloc(sizeof(grids_cl));
	if (!grids_ptr) throw std::runtime_error("Grid too large!");
	memset(grids_ptr, 0, sizeof(grids_cl)); // zero-fills QFD slots 17-20
	grid* tmp_grid_ptr = &c.grids[0];
	grids_ptr->atu   = c.atu;
	grids_ptr->slope = c.slope;
	int grids_front = 0;
	int base_grids2 = (int)c.grids.size();
	for (int i = base_grids2 - 1; i >= 0; i--) {
		for (int j = 0; j < 3; j++) {
			grids_ptr->grids[i].m_init[j]            = tmp_grid_ptr[i].m_init[j];
			grids_ptr->grids[i].m_factor[j]          = tmp_grid_ptr[i].m_factor[j];
			grids_ptr->grids[i].m_dim_fl_minus_1[j]  = tmp_grid_ptr[i].m_dim_fl_minus_1[j];
			grids_ptr->grids[i].m_factor_inv[j]      = tmp_grid_ptr[i].m_factor_inv[j];
		}
		if (tmp_grid_ptr[i].m_data.dim0() != 0) {
			grids_ptr->grids[i].m_i = tmp_grid_ptr[i].m_data.dim0();
			grids_ptr->grids[i].m_j = tmp_grid_ptr[i].m_data.dim1();
			grids_ptr->grids[i].m_k = tmp_grid_ptr[i].m_data.dim2();
			grids_front = i;
		}
		// else: m_i/j/k already 0 from memset
	}
	// QFD: optionally load precomputed field grids from CWD (produced by prep_qfd_grids.py)
	load_qfd_grid_file("qfd_esp.bin",      &grids_ptr->grids[GRID_IDX_ESP]);
	load_qfd_grid_file("qfd_desolv.bin",   &grids_ptr->grids[GRID_IDX_DESOLV]);
	load_qfd_grid_file("qfd_infomap.bin",  &grids_ptr->grids[GRID_IDX_INFOMAP]);
	load_qfd_grid_file("qfd_water.bin",    &grids_ptr->grids[GRID_IDX_WATER]);    // Phase 3: xtal water penalty
	// Phase 5: pipi — load from file if available, else compute from receptor (always active)
	if (!load_qfd_grid_file("qfd_pipi.bin", &grids_ptr->grids[GRID_IDX_PIPI]))
	    compute_pipi_grid_from_receptor(ma, g, &grids_ptr->grids[GRID_IDX_PIPI]);
	mis_ptr->grids_front = grids_front;
	size_t mis_size    = sizeof(mis_cl);
	size_t pre_size    = sizeof(pre_cl);
	size_t pa_size     = sizeof(pa_cl);
	size_t gb_size     = sizeof(gb_cl);
	size_t ar_size     = sizeof(ar_cl);
	size_t grids_size  = sizeof(grids_cl);

	float* needed_ptr = (float*)malloc(mis_ptr->needed_size * sizeof(float));
	for (int i = 0; i < (int)mis_ptr->needed_size; i++) needed_ptr[i] = needed[i];

	// Upload shared (protein-side) buffers
	auto _tbuf0 = std::chrono::steady_clock::now();
	cl_mem pre_gpu, pa_gpu, gb_gpu, ar_gpu, grids_gpu, needed_gpu, mis_gpu;
	CreateDeviceBuffer(&pre_gpu,    CL_MEM_READ_ONLY,  pre_size,                           context);
	CreateDeviceBuffer(&pa_gpu,     CL_MEM_READ_ONLY,  pa_size,                            context);
	CreateDeviceBuffer(&gb_gpu,     CL_MEM_READ_ONLY,  gb_size,                            context);
	CreateDeviceBuffer(&ar_gpu,     CL_MEM_READ_ONLY,  ar_size,                            context);
	CreateDeviceBuffer(&grids_gpu,  CL_MEM_READ_WRITE, grids_size,                         context);
	CreateDeviceBuffer(&needed_gpu, CL_MEM_READ_WRITE, mis_ptr->needed_size * sizeof(float), context);
	CreateDeviceBuffer(&mis_gpu,    CL_MEM_READ_ONLY,  mis_size,                           context);

	err = clEnqueueWriteBuffer(queue, pre_gpu,    false, 0, pre_size,  pre_ptr,    0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, pa_gpu,     false, 0, pa_size,  pa_ptr,     0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, gb_gpu,     false, 0, gb_size,  gb_ptr,     0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, ar_gpu,     false, 0, ar_size,  ar_ptr,     0, NULL, NULL); checkErr(err);

	// grids_gpu: upload ONLY the metadata fields (atu, slope, m_i/j/k, m_init, m_factor…)
	// for standard grids — skip m_data (kernel1 writes those entirely anyway).
	// For QFD grids (slots 17-20), also upload m_data because kernel1 never writes those slots.
	{
		const size_t hdr_sz    = offsetof(grids_cl, grids);     // atu + slope header (8 bytes)
		const size_t meta_sz   = offsetof(grid_cl,  m_data);    // per-grid metadata (60 bytes)
		const size_t grid_full = sizeof(grid_cl);
		const char*  src       = reinterpret_cast<const char*>(grids_ptr);

		err = clEnqueueWriteBuffer(queue, grids_gpu, false, 0, hdr_sz, src, 0, NULL, NULL); checkErr(err);
		// Upload metadata for ALL slots (standard + QFD, all 21)
		for (int i = 0; i < GRIDS_SIZE; i++) {
			size_t off = hdr_sz + (size_t)i * grid_full;
			err = clEnqueueWriteBuffer(queue, grids_gpu, false, off, meta_sz, src + off, 0, NULL, NULL); checkErr(err);
		}
		// Upload m_data for QFD slots that were loaded (kernel1 won't fill these)
		for (int i = base_grids2; i < GRIDS_SIZE; i++) {
			if (grids_ptr->grids[i].m_i > 0) {
				size_t data_off = hdr_sz + (size_t)i * grid_full + meta_sz;
				size_t data_sz  = (size_t)grids_ptr->grids[i].m_i
				                * (size_t)grids_ptr->grids[i].m_j
				                * (size_t)grids_ptr->grids[i].m_k * 8 * sizeof(float);
				err = clEnqueueWriteBuffer(queue, grids_gpu, false, data_off, data_sz, src + data_off, 0, NULL, NULL); checkErr(err);
			}
		}
	}

	err = clEnqueueWriteBuffer(queue, needed_gpu, false, 0, mis_ptr->needed_size * sizeof(float), needed_ptr, 0, NULL, NULL); checkErr(err);
	err = clEnqueueWriteBuffer(queue, mis_gpu,    false, 0, mis_size, mis_ptr,    0, NULL, NULL); checkErr(err);
	clFinish(queue);
	{
		auto _tbuf1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: protein_buf_alloc+upload=%.0fms  (grids skip m_data)\n", gpu_id,
		       std::chrono::duration<double,std::milli>(_tbuf1-_tbuf0).count()); fflush(stdout);
	}

	SetKernelArg(kernels[0], 0, sizeof(cl_mem), &pre_gpu);
	SetKernelArg(kernels[0], 1, sizeof(cl_mem), &pa_gpu);
	SetKernelArg(kernels[0], 2, sizeof(cl_mem), &gb_gpu);
	SetKernelArg(kernels[0], 3, sizeof(cl_mem), &ar_gpu);
	SetKernelArg(kernels[0], 4, sizeof(cl_mem), &grids_gpu);
	SetKernelArg(kernels[0], 5, sizeof(cl_mem), &mis_gpu);
	SetKernelArg(kernels[0], 6, sizeof(cl_mem), &needed_gpu);
	SetKernelArg(kernels[0], 7, sizeof(int),    &c.atu);
	SetKernelArg(kernels[0], 8, sizeof(int),    &nat);

	std::atomic<int> local_status{static_cast<int>(DockingStatus::Docking)};
#ifdef NDEBUG
	std::thread console_thread(print_process, &local_status);
	ThreadGuard console_guard(console_thread);
#endif

	// kernel1: build protein affinity grid
	size_t kernel1_global_size[3] = { 128, 128, 128 };
	size_t kernel1_local_size[3]  = { 4, 4, 4 };
	{
		auto _k1_t0 = std::chrono::steady_clock::now();
		cl_event kernel1_ev;
		err = clEnqueueNDRangeKernel(queue, kernels[0], 3, 0, kernel1_global_size, kernel1_local_size, 0, NULL, &kernel1_ev); checkErr(err);
		clWaitForEvents(1, &kernel1_ev);
		clReleaseEvent(kernel1_ev);
		auto _k1_t1 = std::chrono::steady_clock::now();
		printf("DIAG GPU%d: kernel1 = %.1fms\n", gpu_id,
		       std::chrono::duration<double, std::milli>(_k1_t1 - _k1_t0).count()); fflush(stdout);
	}

	free(pa_ptr); free(gb_ptr); free(ar_ptr); free(needed_ptr); free(grids_ptr);
	err = clReleaseMemObject(pa_gpu);     checkErr(err);
	err = clReleaseMemObject(gb_gpu);     checkErr(err);
	err = clReleaseMemObject(ar_gpu);     checkErr(err);
	err = clReleaseMemObject(needed_gpu); checkErr(err);

	/**** Prepare ligand data (CPU, for both ligands) ****/
	const int thread = par.mc.thread;

	// Helper lambda: populate one m_cl from a model
	auto fill_m_cl = [](const model& m, m_cl* m_ptr) {
		m_ptr->flex_rigid.num_nodes = 0;   // Stage 1: flex forest empty until populated
		m_ptr->other_pairs.num_pairs = 0;  // Stage 2: dual = no flex, empty other_pairs
		for (int ai = 0; ai < (int)m.atoms.size(); ai++) {
			m_ptr->atoms[ai].types[0] = m.atoms[ai].el;
			m_ptr->atoms[ai].types[1] = m.atoms[ai].ad;
			m_ptr->atoms[ai].types[2] = m.atoms[ai].xs;
			m_ptr->atoms[ai].types[3] = m.atoms[ai].sy;
			for (int j = 0; j < 3; j++) m_ptr->atoms[ai].coords[j] = m.atoms[ai].coords[j];
			m_ptr->atoms[ai].charge = (float)m.atoms[ai].charge; // QFD: partial charge [e]
		}
		for (int ci = 0; ci < (int)m.coords.size(); ci++)
			for (int j = 0; j < 3; j++) m_ptr->m_coords.coords[ci][j] = m.coords[ci].data[j];
		for (int ci = 0; ci < (int)m.coords.size(); ci++)
			for (int j = 0; j < 3; j++) m_ptr->minus_forces.coords[ci][j] = m.minus_forces[ci].data[j];

		ligand m_lig = m.ligands[0];
		m_ptr->ligand.pairs.num_pairs = m.ligands[0].pairs.size();
		for (int pi = 0; pi < m_ptr->ligand.pairs.num_pairs; pi++) {
			m_ptr->ligand.pairs.type_pair_index[pi] = m.ligands[0].pairs[pi].type_pair_index;
			m_ptr->ligand.pairs.a[pi] = m.ligands[0].pairs[pi].a;
			m_ptr->ligand.pairs.b[pi] = m.ligands[0].pairs[pi].b;
		}
		m_ptr->ligand.begin = m.ligands[0].begin;
		m_ptr->ligand.end   = m.ligands[0].end;
		m_ptr->ligand.rigid.atom_range[0][0] = m_lig.node.begin;
		m_ptr->ligand.rigid.atom_range[0][1] = m_lig.node.end;
		for (int j = 0; j < 3; j++) m_ptr->ligand.rigid.origin[0][j] = m_lig.node.get_origin()[j];
		for (int j = 0; j < 9; j++) m_ptr->ligand.rigid.orientation_m[0][j] = m_lig.node.get_orientation_m().data[j];
		m_ptr->ligand.rigid.orientation_q[0][0] = m_lig.node.orientation().R_component_1();
		m_ptr->ligand.rigid.orientation_q[0][1] = m_lig.node.orientation().R_component_2();
		m_ptr->ligand.rigid.orientation_q[0][2] = m_lig.node.orientation().R_component_3();
		m_ptr->ligand.rigid.orientation_q[0][3] = m_lig.node.orientation().R_component_4();
		for (int j = 0; j < 3; j++) { m_ptr->ligand.rigid.axis[0][j] = 0; m_ptr->ligand.rigid.relative_axis[0][j] = 0; m_ptr->ligand.rigid.relative_origin[0][j] = 0; }

		struct S {
			int start_index = 0, parent_index = 0;
			void store_node(tree<segment>& ch, rigid_cl& r) {
				start_index++;
				r.parent[start_index] = parent_index;
				r.atom_range[start_index][0] = ch.node.begin;
				r.atom_range[start_index][1] = ch.node.end;
				for (int j = 0; j < 9; j++) r.orientation_m[start_index][j] = ch.node.get_orientation_m().data[j];
				r.orientation_q[start_index][0] = ch.node.orientation().R_component_1();
				r.orientation_q[start_index][1] = ch.node.orientation().R_component_2();
				r.orientation_q[start_index][2] = ch.node.orientation().R_component_3();
				r.orientation_q[start_index][3] = ch.node.orientation().R_component_4();
				for (int j = 0; j < 3; j++) {
					r.origin[start_index][j]          = ch.node.get_origin()[j];
					r.axis[start_index][j]            = ch.node.get_axis()[j];
					r.relative_axis[start_index][j]   = ch.node.relative_axis[j];
					r.relative_origin[start_index][j] = ch.node.relative_origin[j];
				}
				if (ch.children.empty()) return;
				int pi = start_index;
				for (int ci = 0; ci < (int)ch.children.size(); ci++) { this->parent_index = pi; store_node(ch.children[ci], r); }
			}
		} ts;
		for (int ci = 0; ci < (int)m_lig.children.size(); ci++) { ts.parent_index = 0; ts.store_node(m_lig.children[ci], m_ptr->ligand.rigid); }
		m_ptr->ligand.rigid.num_children = ts.start_index;
		m_ptr->ligand.rigid.parent[0] = -1;  // root has no parent (children derived from parent[])
		m_ptr->m_num_movable_atoms = m.num_movable_atoms();
	};

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
	free(mis_ptr); free(pre_ptr);

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