void get_heavy_atom_movable_coords(output_type_cl* tmp, const m_cl* m, ligand_atom_coords_cl* coords) {
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

		// Base offsets for this ligand's data
		int base = lig_id * mis->thread;

		m_cl m = mg[lig_id];

		float best_e = INFINITY;
		output_type_cl best_out;
		ligand_atom_coords_cl best_coords;

		output_type_cl tmp = ric[base + traj_id];

		change_cl g;
		output_type_cl candidate;

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
							torsion_size
			);
			if (rilc_bfgs_enable==1){
				rilc_bfgs(	&candidate,
						&g,
						&m,
						pre,
						grids,
						mis,
						torsion_size,
						max_bfgs_steps
				);
			}else{
				bfgs(	&candidate,
						&g,
						&m,
						pre,
						grids,
						mis,
						torsion_size,
						max_bfgs_steps
				);
			}

			float n = generate_n(rmap->pi_map, map_index);

			if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n)) {

				tmp = candidate;

				set(&tmp, &m.ligand.rigid, &m.m_coords,
					m.atoms, m.m_num_movable_atoms, mis->epsilon_fl);

				if (tmp.e < best_e) {
					if (rilc_bfgs_enable==1){
						rilc_bfgs(	&tmp,
								&g,
								&m,
								pre,
								grids,
								mis,
								torsion_size,
								max_bfgs_steps
						);
					}else{
						bfgs(	&tmp,
								&g,
								&m,
								pre,
								grids,
								mis,
								torsion_size,
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
		}

		// write the best conformation back to CPU
		results[base + traj_id] = best_out;
		coords[base  + traj_id] = best_coords;
	}
}
