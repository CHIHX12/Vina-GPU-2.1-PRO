void get_heavy_atom_movable_coords(output_type_cl* tmp, const m_cl_private* m, ligand_atom_coords_cl* coords) {
	int counter = 0;
	for (int i = 0; i < m->m_num_movable_atoms; i++) {
		if (m->atoms[i].types[0] != EL_TYPE_H) {
			for (int j = 0; j < 3; j++)coords->coords[counter][j] = m->m_coords.coords[i][j];
			counter++;
		}
		else {
			//printf("\n kernel2: removed H atom coords in get_heavy_atom_movable_coords()!");
		}
	}
	//assign 0 for others
	for (int i = counter; i < MAX_NUM_OF_ATOMS; i++) {
		for (int j = 0; j < 3; j++)coords->coords[i][j] = 0;
	}
}


//Generate a random number according to step
float generate_n(__global const float* pi_map, const int step) {
	return fabs(pi_map[step]) / M_PI;
}

bool metropolis_accept(float old_f, float new_f, float temperature, float n) {
	if (new_f < old_f)return true;
	const float acceptance_probability = exp((old_f - new_f) / temperature);
	bool res = n < acceptance_probability;
	return n < acceptance_probability;
}

__kernel
void kernel2(
				const		__global		output_type_cl*			ric,
							__global		m_cl*					mg,
						const __global		pre_cl*					pre,
						const __global		grids_cl*				grids,
					const		__global		random_maps*			rand_maps_arr,
							__global		ligand_atom_coords_cl*	coords,
							__global		output_type_cl*			results,
				const		__global		mis_cl*					mis,
				const		__global		int*					torsion_sizes_arr,
				const		__global		int*					search_depths_arr,
				const		__global		int*					bfgs_steps_arr,
				const 					int						rilc_bfgs_enable,
				const					int						batch_n
) {
	int gl = get_global_linear_id();
	// Use actual launch global size as stride so every work item gets work.
	// With batch_n=2, thread=8000, global_size=16384:
	//   WIs 0..7999 handle lig_id=0, traj_id 0..7999
	//   WIs 8000..15999 handle lig_id=1, traj_id 0..7999
	int total_wi = get_global_size(0) * get_global_size(1);

	for (int gll = gl; gll < mis->thread * batch_n; gll += total_wi) {
		int lig_id  = gll / mis->thread;
		int traj_id = gll % mis->thread;

		if (lig_id >= batch_n) break;

		const __global random_maps* rmap = rand_maps_arr + lig_id;
		int torsion_size  = torsion_sizes_arr[lig_id];
		int search_depth  = search_depths_arr[lig_id];
		int max_bfgs_steps = bfgs_steps_arr[lig_id];
		// Stage 3: packed torsion dims (lig | flex<<16) for bfgs/rilc_bfgs index+dimension handling.
		// mutate_conf_cl below still gets the RAW torsion_size + flex size separately.
		int tcount = (torsion_size & 0xFFFF) | ((mis->flex_torsion_size & 0xFFFF) << 16);

		// Base offsets for this ligand's data
		int base = lig_id * mis->thread;

		// Build a pairs-free private copy of m — keeps GPU private memory ~33 KB
		// instead of ~224 KB (with MAX_NUM_OF_LIG_PAIRS=16384 full copy would OOM).
		// Pairs stay in global memory and are accessed via pairs_g.
		m_cl_private m;
		m.m_num_movable_atoms = mg[lig_id].m_num_movable_atoms;
		for (int _i = 0; _i < m.m_num_movable_atoms; _i++)
			m.atoms[_i] = mg[lig_id].atoms[_i];
		m.m_coords    = mg[lig_id].m_coords;
		m.minus_forces = mg[lig_id].minus_forces;
		m.ligand.begin = mg[lig_id].ligand.begin;
		m.ligand.end   = mg[lig_id].ligand.end;
		m.ligand.rigid = mg[lig_id].ligand.rigid;
		m.flex_rigid   = mg[lig_id].flex_rigid;   // Stage 1: flex forest (num_nodes==0 ⇒ dormant)
		__global const lig_pairs_cl* pairs_g = &mg[lig_id].ligand.pairs;
		__global const other_pairs_cl* other_g = &mg[lig_id].other_pairs;  // Stage 2 (num_pairs==0 ⇒ no-op)

		float best_e = INFINITY;
		output_type_cl best_out;
		ligand_atom_coords_cl best_coords;

		output_type_cl tmp = ric[base + traj_id];

		change_cl g;
		output_type_cl candidate;

		// --- QFD Phase 2: Temperature Annealing ---
		// Each trajectory gets a unique starting temperature from a geometric ladder.
		// Anneals exponentially from T_start → QFD_T_FINAL over search_depth steps.
		int   rep_id  = traj_id % QFD_N_REPLICAS;
		float T_start = QFD_T_START_MIN
		                * pow(QFD_T_START_MAX / QFD_T_START_MIN,
		                      (float)rep_id / (float)(QFD_N_REPLICAS - 1));

		for (int step = 0; step < search_depth; step++) {
			candidate = tmp;

			int map_index = (step + traj_id * search_depth) % MAX_NUM_OF_RANDOM_MAP;
			mutate_conf_cl(	map_index,
							&candidate,
							rmap->int_map,
							rmap->sphere_map,
							rmap->pi_map,
							m.ligand.begin,
							m.ligand.end,
							m.atoms,
							&m.m_coords,
							m.ligand.rigid.origin[0],
							mis->epsilon_fl,
							mis->mutation_amplitude,
							torsion_size,
							mis->flex_torsion_size   // Phase 4: 0 for rigid receptor
			);
			if (rilc_bfgs_enable==1){
				rilc_bfgs(	&candidate,
						&g,
						&m,
						pairs_g,
						other_g,
						pre,
						grids,
						mis,
						tcount,
						max_bfgs_steps
				);
			}else{
				bfgs(	&candidate,
						&g,
						&m,
						pairs_g,
						other_g,
						pre,
						grids,
						mis,
						tcount,
						max_bfgs_steps
				);
			}

			// Decorrelate the Metropolis acceptance draw from the mutation RNG.
			// map_index feeds mutate_conf_cl (torsion value = pi_map[map_index]); reusing the
			// same entry for n made acceptance a deterministic function of the chosen torsion
			// (uphill moves to extreme angles were always rejected), breaking detailed balance.
			int accept_index = (map_index + MAX_NUM_OF_RANDOM_MAP / 2) % MAX_NUM_OF_RANDOM_MAP;
			float n = generate_n(rmap->pi_map, accept_index);

			// QFD temperature: anneal exponentially from T_start → QFD_T_FINAL
			float progress    = (float)step / (float)search_depth;
			float temperature = T_start * pow(QFD_T_FINAL / T_start, progress);

			if (step == 0 || metropolis_accept(tmp.e, candidate.e, temperature, n)) {

				tmp = candidate;

				set(&tmp, &m.ligand.rigid, &m.m_coords,
					m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);
				set_flex(&tmp, &m.flex_rigid, &m.m_coords, m.atoms, mis->epsilon_fl);

				if (tmp.e < best_e) {
					if (rilc_bfgs_enable==1){
						rilc_bfgs(	&tmp,
								&g,
								&m,
								pairs_g,
								other_g,
								pre,
								grids,
								mis,
								tcount,
								max_bfgs_steps
						);
					}else{
						bfgs(	&tmp,
								&g,
								&m,
								pairs_g,
								other_g,
								pre,
								grids,
								mis,
								tcount,
								max_bfgs_steps
						);
					}

					// set
					if (tmp.e < best_e) {
						set(&tmp, &m.ligand.rigid, &m.m_coords,
							m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);

						best_out = tmp;
						get_heavy_atom_movable_coords(&best_out, &m, &best_coords); // get coords
						best_e = tmp.e;
					}

				}
			}

#if BH_ENABLE
			// --- Basin-Hopping hops ---
			// For each hop: large random displacement from current accepted conf (tmp),
			// re-minimise, Metropolis accept at current SA temperature.
			// Uses map indices offset by (hop+1) from the step's map_index to avoid aliasing.
			for (int hop = 0; hop < BH_N_HOPS; hop++) {
				int hop_map = (map_index + hop + 1) % MAX_NUM_OF_RANDOM_MAP;

				output_type_cl hopped = tmp;

				// Large translation: BH_PERTURBATION_ANG * unit sphere vector
				hopped.position[0] += BH_PERTURBATION_ANG * rmap->sphere_map[hop_map][0];
				hopped.position[1] += BH_PERTURBATION_ANG * rmap->sphere_map[hop_map][1];
				hopped.position[2] += BH_PERTURBATION_ANG * rmap->sphere_map[hop_map][2];

				// Randomise one torsion (if any)
				if (torsion_size > 0) {
					int t_idx = (rmap->int_map[hop_map] % torsion_size + torsion_size) % torsion_size;
					hopped.lig_torsion[t_idx] = rmap->pi_map[(hop_map + 1) % MAX_NUM_OF_RANDOM_MAP];
				}

				// Re-minimise to nearest local minimum
				if (rilc_bfgs_enable == 1) {
					rilc_bfgs(&hopped, &g, &m, pairs_g, other_g, pre, grids, mis, tcount, max_bfgs_steps);
				} else {
					bfgs(&hopped, &g, &m, pairs_g, other_g, pre, grids, mis, tcount, max_bfgs_steps);
				}

				float hop_n = generate_n(rmap->pi_map, (hop_map + 2) % MAX_NUM_OF_RANDOM_MAP);

				// Metropolis accept at current SA temperature
				if (metropolis_accept(tmp.e, hopped.e, temperature, hop_n)) {
					tmp = hopped;
					set(&tmp, &m.ligand.rigid, &m.m_coords,
						m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);

					// Promote to global best if improved
					if (tmp.e < best_e) {
						if (rilc_bfgs_enable == 1) {
							rilc_bfgs(&tmp, &g, &m, pairs_g, other_g, pre, grids, mis, tcount, max_bfgs_steps);
						} else {
							bfgs(&tmp, &g, &m, pairs_g, other_g, pre, grids, mis, tcount, max_bfgs_steps);
						}
						if (tmp.e < best_e) {
							set(&tmp, &m.ligand.rigid, &m.m_coords,
								m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);
							best_out = tmp;
							get_heavy_atom_movable_coords(&best_out, &m, &best_coords);
							best_e = tmp.e;
						}
					}
				}
			}
#endif  // BH_ENABLE
		}

		// write the best conformation back to CPU
		results[base + traj_id] = best_out;
		coords[base  + traj_id] = best_coords;
	}
}

// ---------------------------------------------------------------------------
// GPU Dual-Ligand Co-Docking Kernel
// Each work item runs one MC trajectory for the combined A+B system.
// Energy = E(protein-A) + E(protein-B) + E(A-B  lig-lig)
// ---------------------------------------------------------------------------

// Local helper: num atom types per typing scheme (mirrors kernel1 definition)
static int k2_nat(int atu) {
	if (atu == 0) return 11;   // EL_TYPE_SIZE
	if (atu == 1) return 32;   // AD_TYPE_SIZE
	if (atu == 2) return 17;   // XS_TYPE_SIZE
	return 18;                 // SY_TYPE_SIZE (atu==3)
}

// Pairwise lig-lig Vina energy — uses the same precomputed fast[] table
// as intra-ligand scoring. No gradients needed for Metropolis accept/reject.
float eval_lig_lig_cl(
	const m_cl_private* ma,
	const m_cl_private* mb,
	const __global pre_cl* pre,
	int atu
) {
	float e = 0.0f;
	int n = k2_nat(atu);
	for (int i = 0; i < ma->m_num_movable_atoms; i++) {
		int t1 = ma->atoms[i].types[atu];
		if (t1 >= n) continue;
		for (int j = 0; j < mb->m_num_movable_atoms; j++) {
			int t2 = mb->atoms[j].types[atu];
			if (t2 >= n) continue;
			float dx = ma->m_coords.coords[i][0] - mb->m_coords.coords[j][0];
			float dy = ma->m_coords.coords[i][1] - mb->m_coords.coords[j][1];
			float dz = ma->m_coords.coords[i][2] - mb->m_coords.coords[j][2];
			float r2 = dx*dx + dy*dy + dz*dz;
			if (r2 < pre->m_cutoff_sqr) {
				int ti = (t1 <= t2) ? t1 : t2;
				int tj = (t1 <= t2) ? t2 : t1;
				int tpi = ti + tj*(tj+1)/2;
				int idx = (int)(pre->factor * r2);
				if (idx < FAST_SIZE) e += pre->m_data[tpi].fast[idx];
			}
		}
	}
	return e;
}

// Macro: run bfgs or rilc_bfgs depending on rilc flag (og = other_pairs, empty for dual = no flex)
#define DUAL_BFGS(x, g, m, pg, og, torsion) \
	do { \
		if (rilc_bfgs_enable == 1) \
			rilc_bfgs((x), (g), (m), (pg), (og), pre, grids, mis, (torsion), max_bfgs_steps); \
		else \
			bfgs((x), (g), (m), (pg), (og), pre, grids, mis, (torsion), max_bfgs_steps); \
	} while(0)

__kernel
void kernel2_dual(
	// Initial random conformations for A and B
	const __global output_type_cl*      ric_a,
	const __global output_type_cl*      ric_b,
	// Ligand models (modified in-place per work item private copy)
	      __global m_cl*               mg_a,
	      __global m_cl*               mg_b,
	// Shared protein scoring data
	const __global pre_cl*             pre,
	const __global grids_cl*           grids,
	// Per-ligand random maps (one per batch slot)
	const __global random_maps*        rmaps_a,
	const __global random_maps*        rmaps_b,
	// Output heavy-atom coords and best poses
	      __global ligand_atom_coords_cl* coords_a,
	      __global ligand_atom_coords_cl* coords_b,
	      __global output_type_cl*     results_a,
	      __global output_type_cl*     results_b,
	// Shared MC parameters
	const __global mis_cl*             mis,
	// Per-pair scalars
	const __global int*                torsion_a_arr,
	const __global int*                torsion_b_arr,
	const __global int*                search_depths_arr,
	const __global int*                bfgs_steps_arr,
	const           int                rilc_bfgs_enable,
	const           int                batch_n
) {
	int gl = get_global_linear_id();
	int total_wi = get_global_size(0) * get_global_size(1);
	int atu = grids->atu;

	for (int gll = gl; gll < mis->thread * batch_n; gll += total_wi) {
		int lig_id  = gll / mis->thread;
		int traj_id = gll % mis->thread;
		if (lig_id >= batch_n) break;

		int base       = lig_id * mis->thread;
		int torsion_a  = torsion_a_arr[lig_id];
		int torsion_b  = torsion_b_arr[lig_id];
		int search_depth   = search_depths_arr[lig_id];
		int max_bfgs_steps = bfgs_steps_arr[lig_id];

		// Pairs-free private copies — same memory optimisation as kernel2.
		m_cl_private ma;
		ma.m_num_movable_atoms = mg_a[lig_id].m_num_movable_atoms;
		for (int _i = 0; _i < ma.m_num_movable_atoms; _i++)
			ma.atoms[_i] = mg_a[lig_id].atoms[_i];
		ma.m_coords    = mg_a[lig_id].m_coords;
		ma.minus_forces = mg_a[lig_id].minus_forces;
		ma.ligand.begin = mg_a[lig_id].ligand.begin;
		ma.ligand.end   = mg_a[lig_id].ligand.end;
		ma.ligand.rigid = mg_a[lig_id].ligand.rigid;
		__global const lig_pairs_cl* pairs_ga = &mg_a[lig_id].ligand.pairs;
		__global const other_pairs_cl* other_ga = &mg_a[lig_id].other_pairs;  // empty (dual = no flex)

		m_cl_private mb;
		mb.m_num_movable_atoms = mg_b[lig_id].m_num_movable_atoms;
		for (int _i = 0; _i < mb.m_num_movable_atoms; _i++)
			mb.atoms[_i] = mg_b[lig_id].atoms[_i];
		mb.m_coords    = mg_b[lig_id].m_coords;
		mb.minus_forces = mg_b[lig_id].minus_forces;
		mb.ligand.begin = mg_b[lig_id].ligand.begin;
		mb.ligand.end   = mg_b[lig_id].ligand.end;
		mb.ligand.rigid = mg_b[lig_id].ligand.rigid;
		__global const lig_pairs_cl* pairs_gb = &mg_b[lig_id].ligand.pairs;
		__global const other_pairs_cl* other_gb = &mg_b[lig_id].other_pairs;  // empty (dual = no flex)

		// Initial conformations for this trajectory
		output_type_cl tmp_a = ric_a[base + traj_id];
		output_type_cl tmp_b = ric_b[base + traj_id];

		change_cl ga, gb;

		// Apply initial conf → update m_coords
		set(&tmp_a, &ma.ligand.rigid, &ma.m_coords, ma.atoms, ma.m_num_movable_atoms, mis->epsilon_fl);
		set(&tmp_b, &mb.ligand.rigid, &mb.m_coords, mb.atoms, mb.m_num_movable_atoms, mis->epsilon_fl);

		// Quick initial minimization for both
		DUAL_BFGS(&tmp_a, &ga, &ma, pairs_ga, other_ga, torsion_a);
		set(&tmp_a, &ma.ligand.rigid, &ma.m_coords, ma.atoms, ma.m_num_movable_atoms, mis->epsilon_fl);
		DUAL_BFGS(&tmp_b, &gb, &mb, pairs_gb, other_gb, torsion_b);
		set(&tmp_b, &mb.ligand.rigid, &mb.m_coords, mb.atoms, mb.m_num_movable_atoms, mis->epsilon_fl);

		float e_ll   = eval_lig_lig_cl(&ma, &mb, pre, atu);
		float cur_combined = tmp_a.e + tmp_b.e + e_ll;

		// SA ladder — same replica assignment as kernel2
		int   dual_rep_id  = traj_id % QFD_N_REPLICAS;
		float dual_T_start = QFD_T_START_MIN
		                     * pow(QFD_T_START_MAX / QFD_T_START_MIN,
		                           (float)dual_rep_id / (float)(QFD_N_REPLICAS - 1));

		// Track best trajectory
		float best_combined = cur_combined;
		output_type_cl      best_out_a = tmp_a,  best_out_b = tmp_b;
		ligand_atom_coords_cl best_coords_a, best_coords_b;
		get_heavy_atom_movable_coords(&best_out_a, &ma, &best_coords_a);
		get_heavy_atom_movable_coords(&best_out_b, &mb, &best_coords_b);

		// ---- MC loop ----
		for (int step = 0; step < search_depth; step++) {
			int map_index = (step + traj_id * search_depth) % MAX_NUM_OF_RANDOM_MAP;
			float n_rand  = generate_n(rmaps_a->pi_map, map_index);

			// SA temperature: same exponential ladder as kernel2
			float dual_progress = (float)step / (float)search_depth;
			float dual_temp     = dual_T_start * pow(QFD_T_FINAL / dual_T_start, dual_progress);

			// 50/50 branch: pi_map >= 0 → mutate A, otherwise mutate B
			if (rmaps_a->pi_map[map_index] >= 0.0f) {
				// --- Mutate A ---
				output_type_cl cand_a = tmp_a;
				mutate_conf_cl(map_index, &cand_a,
					rmaps_a->int_map, rmaps_a->sphere_map, rmaps_a->pi_map,
					ma.ligand.begin, ma.ligand.end, ma.atoms, &ma.m_coords,
					ma.ligand.rigid.origin[0], mis->epsilon_fl,
					mis->mutation_amplitude, torsion_a, mis->flex_torsion_size);
				DUAL_BFGS(&cand_a, &ga, &ma, pairs_ga, other_ga, torsion_a);
				set(&cand_a, &ma.ligand.rigid, &ma.m_coords,
					ma.atoms, ma.m_num_movable_atoms, mis->epsilon_fl);

				float e_ll_new   = eval_lig_lig_cl(&ma, &mb, pre, atu);
				float new_combined = cand_a.e + tmp_b.e + e_ll_new;

				if (step == 0 || metropolis_accept(cur_combined, new_combined, dual_temp, n_rand)) {
					tmp_a = cand_a;
					cur_combined = new_combined;
					if (new_combined < best_combined) {
						// Extra refinement on the new best
						DUAL_BFGS(&tmp_a, &ga, &ma, pairs_ga, other_ga, torsion_a);
						set(&tmp_a, &ma.ligand.rigid, &ma.m_coords,
							ma.atoms, ma.m_num_movable_atoms, mis->epsilon_fl);
						float e_ll2 = eval_lig_lig_cl(&ma, &mb, pre, atu);
						float refined = tmp_a.e + tmp_b.e + e_ll2;
						if (refined < best_combined) {
							best_combined = refined;
							best_out_a = tmp_a; best_out_b = tmp_b;
							get_heavy_atom_movable_coords(&best_out_a, &ma, &best_coords_a);
							get_heavy_atom_movable_coords(&best_out_b, &mb, &best_coords_b);
						}
						cur_combined = refined;
					}
				} else {
					// Reject: restore ma.m_coords to last-accepted tmp_a
					set(&tmp_a, &ma.ligand.rigid, &ma.m_coords,
						ma.atoms, ma.m_num_movable_atoms, mis->epsilon_fl);
				}

			} else {
				// --- Mutate B ---
				output_type_cl cand_b = tmp_b;
				mutate_conf_cl(map_index, &cand_b,
					rmaps_b->int_map, rmaps_b->sphere_map, rmaps_b->pi_map,
					mb.ligand.begin, mb.ligand.end, mb.atoms, &mb.m_coords,
					mb.ligand.rigid.origin[0], mis->epsilon_fl,
					mis->mutation_amplitude, torsion_b, mis->flex_torsion_size);
				DUAL_BFGS(&cand_b, &gb, &mb, pairs_gb, other_gb, torsion_b);
				set(&cand_b, &mb.ligand.rigid, &mb.m_coords,
					mb.atoms, mb.m_num_movable_atoms, mis->epsilon_fl);

				float e_ll_new   = eval_lig_lig_cl(&ma, &mb, pre, atu);
				float new_combined = tmp_a.e + cand_b.e + e_ll_new;

				if (step == 0 || metropolis_accept(cur_combined, new_combined, dual_temp, n_rand)) {
					tmp_b = cand_b;
					cur_combined = new_combined;
					if (new_combined < best_combined) {
						DUAL_BFGS(&tmp_b, &gb, &mb, pairs_gb, other_gb, torsion_b);
						set(&tmp_b, &mb.ligand.rigid, &mb.m_coords,
							mb.atoms, mb.m_num_movable_atoms, mis->epsilon_fl);
						float e_ll2 = eval_lig_lig_cl(&ma, &mb, pre, atu);
						float refined = tmp_a.e + tmp_b.e + e_ll2;
						if (refined < best_combined) {
							best_combined = refined;
							best_out_a = tmp_a; best_out_b = tmp_b;
							get_heavy_atom_movable_coords(&best_out_a, &ma, &best_coords_a);
							get_heavy_atom_movable_coords(&best_out_b, &mb, &best_coords_b);
						}
						cur_combined = refined;
					}
				} else {
					set(&tmp_b, &mb.ligand.rigid, &mb.m_coords,
						mb.atoms, mb.m_num_movable_atoms, mis->epsilon_fl);
				}
			}
		}

		// Write best trajectory results
		results_a[base + traj_id] = best_out_a;
		results_b[base + traj_id] = best_out_b;
		coords_a[base  + traj_id] = best_coords_a;
		coords_b[base  + traj_id] = best_coords_b;
	}
}
#undef DUAL_BFGS
