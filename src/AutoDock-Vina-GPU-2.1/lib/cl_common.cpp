#include "cl_common.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <algorithm>
#include <cmath>
#include <thread>

using namespace std;

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

std::vector<output_type> cl_to_vina(output_type_cl result_ptr[],
									ligand_atom_coords_cl result_coords_ptr[],
									int thread, int lig_torsion_size,
									const std::vector<int>& flex_tors_counts)
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

// QFD grid builders moved to qfd_grids.{h,cpp}; LS metal rescoring moved to ls_metal.{h,cpp}.

// Replica Exchange: Metropolis swap between adjacent SA replicas.
// Called after each kernel2 epoch. result_ptrs[li] is the result for ligand li,
// with thread entries sorted by traj_id. rep_id = traj_id % QFD_N_REPLICAS.
// T_r = QFD_T_START_MIN * pow(QFD_T_START_MAX/QFD_T_START_MIN, r/(QFD_N_REPLICAS-1))
void apply_re_swaps(
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
void apply_de_exchange(
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
