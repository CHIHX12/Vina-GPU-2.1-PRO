int fl_to_sz(float x, float max_sz) {
	if (x <= 0) return 0;
	if (x >= max_sz) return max_sz;
	return (int)x;
}
int num_atom_types(int atu) {
	switch (atu) {
	case 0: return EL_TYPE_SIZE;
	case 1: return AD_TYPE_SIZE;
	case 2: return XS_TYPE_SIZE;
	case 3: return SY_TYPE_SIZE;
	default: printf("\nkernel1:num_atom_types ERROR!"); return INFINITY;
	}
}

const float vec_distance_sqr(const float* a, const float* b) {
	return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);
}

const int triangular_matrix_index(int n, int i, int j) {
	if (j >= n) printf("\nkernel1:triangular_matrix_index ERROR!");
	if (i > j) printf("\nkernel1:triangular_matrix_index ERROR!");
	return i + j * (j + 1) / 2;
}

const int triangular_matrix_index_permissive(int n, int i, int j) {
	return (i <= j) ? triangular_matrix_index(n, i, j) : triangular_matrix_index(n, j, i);
}

float eval_fast(int type_pair_index, float r2, float cutoff_sqr, const __global pre_cl* pre) {
	if (r2 > cutoff_sqr) printf("\nkernel1:eval_fast ERROR!");
	if (r2 * pre->factor >= FAST_SIZE)printf("\nkernel1:eval_fast ERROR!");
	int i = (int)(pre->factor * r2);
	if (i >= FAST_SIZE)printf("\nkernel1:eval_fast ERROR!");
	float res = pre->m_data[type_pair_index].fast[i];
	return res;
}

const __global int* possibilities(						float*		coords,
												const	__global	ar_cl*		ar,
												const				float		epsilon_fl, 
												const	__global	gb_cl*		gb,
																	int*		relation_count,
												const	__global	mis_cl*		mis
) {
	int index[3];
	float temp_array[3];
	int m_data_dims[3] = { mis->ar_mi, mis->ar_mj, mis->ar_mk };
	for (int i = 0; i < 3; i++) {
		if (coords[i] + epsilon_fl < gb->init[i]) printf("\nkernel1:possibilities ERROR!1 coords = %f,  init = %f", coords[i], gb->init[i]);//replace assert()
		if (coords[i] > gb->init[i] + gb->range[i] + epsilon_fl) printf("\nkernel1:possibilities ERROR!2 coords = %f,  init = %f, range =%f", coords[i], gb->init[i], gb->range[i]);//replace assert()
		const float tmp = (coords[i] - gb->init[i]) * m_data_dims[i] / gb->range[i];
		temp_array[i] = tmp;
		index[i] = fl_to_sz(tmp, m_data_dims[i] - 1);
	}
	int temp = index[0] + m_data_dims[0] * (index[1] + m_data_dims[1] * index[2]);
	const __global int* address;
	address = &(ar->relation[temp][0]);
	*relation_count = ar->relation_size[temp];
	return address;
}


__kernel
void kernel1(
				const		__global		pre_cl*		pre,
				const		__global		pa_cl*		pa,
				const		__global		gb_cl*		gb,
				const		__global		ar_cl*		ar,
							__global		grids_cl*	grids,
				const		__global		mis_cl*		mis,
				const		__global		float*		needed,
				const						int			atu,
				const						int			nat
) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);
	int grids_front = mis->grids_front;

	//printf("here!");
	if (x >=grids->grids[grids_front].m_i)return;
	if (y >=grids->grids[grids_front].m_j)return;
	if (z >=grids->grids[grids_front].m_k)return;
	
	float affinities[17]; if (mis->needed_size > 17) printf("\nkernel1: ERROR!");
	for (int i = 0; i < mis->needed_size; i++) affinities[i] = 0;

	float probe_coords[3] = {	grids->grids[grids_front].m_init[0] + grids->grids[grids_front].m_factor_inv[0] * x,
								grids->grids[grids_front].m_init[1] + grids->grids[grids_front].m_factor_inv[1] * y,
								grids->grids[grids_front].m_init[2] + grids->grids[grids_front].m_factor_inv[2] * z };
	int relation_count;

	const __global int* possibilities_ptr = possibilities(probe_coords, ar, mis->epsilon_fl, gb, &relation_count, mis);

	for (int possibilities_i = 0; possibilities_i < relation_count; possibilities_i++) {
		const int i = possibilities_ptr[possibilities_i];
		const atom_cl* a = &pa->atoms[i];
		const int t1 = a->types[atu];
		if (t1 >= nat) continue;
		const float r2 = vec_distance_sqr(a->coords, probe_coords);

		if (r2 <= mis->cutoff_sqr) {
			for (int j = 0; j < mis->needed_size; j++) {
				const int t2 = needed[j]; if (t2 >= nat) printf("\nkernel1: ERROR!1 t2 = %d,nat=%d", t2, nat);
				const int type_pair_index = triangular_matrix_index_permissive(num_atom_types(atu), t1, t2);
				affinities[j] += eval_fast(type_pair_index, r2, mis->cutoff_sqr, pre);
			}

			// Metal coordination Gaussian: E = -w * exp(-(r-r0)^2 / 2s^2)
			// XS mode (atu==2) only.  Donor types: N_A=4, N_DA=5, O_A=8, O_DA=9, S_P=10
			// Original  AD types 13-17: Mg, Mn, Zn, Ca, Fe
			// Extended  AD types 20-31: Mo, W, Cu, Co, Ni, V, Pt, Pd, Ru, Rh, Cd, Hg
			// HSAB donor rules:
			//   Hard acids  (Mg,Ca,Mn,V)       : N+O only
			//   Borderline  (Fe,Zn,Mo,W,Co,Ni) : N+O+S
			//   Soft acids  (Cu,Pt,Pd,Ru,Rh,Cd): N+O+S  (higher S weight)
			//   Very soft   (Hg)                : S >> N, no O
			// Parameters from CSD medians (Cambridge Structural Database)
			{
				const int ad = a->types[1];
				int is_orig   = (ad >= 13 && ad <= 17);
				int is_extend = (ad >= 20 && ad <= 31);
				if (atu == 2 && (is_orig || is_extend)) {
					// Which donor classes this metal accepts
					// accepts_S: all except hard acids (Mg=13, Mn=14, Ca=16, Fe=17, V=25)
					int accepts_S  = (ad == 15) || (ad >= 20 && ad != 25);
					// accepts_NO: all except very-soft Hg (Hg prefers S, give O/N small weight)
					// (we keep NO for Hg at reduced weight via the Gaussian params)
					int accepts_NO = 1;

					for (int j = 0; j < mis->needed_size; j++) {
						const int t2 = needed[j];
						int is_NO = (t2 == 4 || t2 == 5 || t2 == 8 || t2 == 9);
						int is_S  = (t2 == 10);
						if (!(is_NO && accepts_NO) && !(is_S && accepts_S)) continue;
						if (r2 >= 25.0f) continue;

						float r_ideal, sigma, weight;

						if (is_S) {
							// S-coordination (CSD medians, Å)
							if      (ad == 15) { r_ideal=2.30f; sigma=0.35f; weight=3.5f; } // Zn-S
							else if (ad == 20) { r_ideal=2.45f; sigma=0.40f; weight=3.0f; } // Mo-S
							else if (ad == 21) { r_ideal=2.50f; sigma=0.40f; weight=3.0f; } // W-S
							else if (ad == 22) { r_ideal=2.30f; sigma=0.35f; weight=4.0f; } // Cu-S
							else if (ad == 23) { r_ideal=2.35f; sigma=0.35f; weight=3.5f; } // Co-S
							else if (ad == 24) { r_ideal=2.25f; sigma=0.35f; weight=4.0f; } // Ni-S
							else if (ad == 26) { r_ideal=2.32f; sigma=0.30f; weight=5.0f; } // Pt-S
							else if (ad == 27) { r_ideal=2.32f; sigma=0.30f; weight=5.0f; } // Pd-S
							else if (ad == 28) { r_ideal=2.35f; sigma=0.35f; weight=3.5f; } // Ru-S
							else if (ad == 29) { r_ideal=2.35f; sigma=0.35f; weight=3.5f; } // Rh-S
							else if (ad == 30) { r_ideal=2.55f; sigma=0.40f; weight=4.0f; } // Cd-S
							else               { r_ideal=2.35f; sigma=0.40f; weight=6.0f; } // Hg-S (ad==31)
						} else {
							// N/O coordination (CSD medians, Å)
							if      (ad == 13) { r_ideal=2.05f; sigma=0.35f; weight= 6.0f; } // Mg
							else if (ad == 14) { r_ideal=2.20f; sigma=0.40f; weight= 7.0f; } // Mn
							else if (ad == 15) { r_ideal=2.10f; sigma=0.35f; weight= 4.0f; } // Zn
							else if (ad == 16) { r_ideal=2.40f; sigma=0.40f; weight= 5.0f; } // Ca
							else if (ad == 17) { r_ideal=2.05f; sigma=0.40f; weight= 8.0f; } // Fe
							else if (ad == 20) { r_ideal=2.10f; sigma=0.40f; weight= 6.0f; } // Mo
							else if (ad == 21) { r_ideal=2.15f; sigma=0.40f; weight= 6.0f; } // W
							else if (ad == 22) { r_ideal=2.00f; sigma=0.35f; weight= 6.0f; } // Cu
							else if (ad == 23) { r_ideal=2.10f; sigma=0.35f; weight= 7.0f; } // Co
							else if (ad == 24) { r_ideal=2.05f; sigma=0.35f; weight= 7.0f; } // Ni
							else if (ad == 25) { r_ideal=2.00f; sigma=0.40f; weight= 5.0f; } // V
							else if (ad == 26) { r_ideal=2.02f; sigma=0.30f; weight= 5.0f; } // Pt
							else if (ad == 27) { r_ideal=2.02f; sigma=0.30f; weight= 5.0f; } // Pd
							else if (ad == 28) { r_ideal=2.05f; sigma=0.35f; weight= 5.0f; } // Ru
							else if (ad == 29) { r_ideal=2.05f; sigma=0.35f; weight= 5.0f; } // Rh
							else if (ad == 30) { r_ideal=2.30f; sigma=0.40f; weight= 4.0f; } // Cd
							else               { r_ideal=2.45f; sigma=0.40f; weight= 2.0f; } // Hg N/O (weak)
						}

						float r  = sqrt(r2);
						float dr = r - r_ideal;
						affinities[j] += -weight * exp(-dr * dr / (2.0f * sigma * sigma));
					}
				}
			}
		}
	}


	for (int j = 0; j < mis->needed_size; j++) {
		int t = needed[j]; if (t >= nat)printf("\nkernel1: ERROR!2 t = %d,nat=%d",t,nat);
		int mi = grids->grids[t].m_i;
		int mj = grids->grids[t].m_j;

		int addr_base =  x		+ mi * ( y      + mj *  z     );
		int addr_1	  = (x - 1) + mi * ( y      + mj *  z     );
		int addr_2    =  x      + mi * ((y - 1) + mj *  z     );
		int addr_3    = (x - 1) + mi * ((y - 1) + mj *  z     );
		int addr_4    =  x      + mi * ( y      + mj * (z - 1));
		int addr_5    = (x - 1) + mi * ( y      + mj * (z - 1));
		int addr_6    =  x      + mi * ((y - 1) + mj * (z - 1));
		int addr_7    = (x - 1) + mi * ((y - 1) + mj * (z - 1));

								grids->grids[t].m_data[addr_base * 8    ] = affinities[j];
		if (x != 0)				grids->grids[t].m_data[addr_1    * 8 + 1] = affinities[j];
		if (y != 0)				grids->grids[t].m_data[addr_2    * 8 + 2] = affinities[j];
		if (x != 0 && y != 0)	grids->grids[t].m_data[addr_3    * 8 + 3] = affinities[j];
		if (z != 0)				grids->grids[t].m_data[addr_4    * 8 + 4] = affinities[j];
		if (x != 0 && z != 0)	grids->grids[t].m_data[addr_5    * 8 + 5] = affinities[j];
		if (y != 0 && z != 0)	grids->grids[t].m_data[addr_6    * 8 + 6] = affinities[j];
		if (x * y * z != 0)		grids->grids[t].m_data[addr_7    * 8 + 7] = affinities[j];

	}
}