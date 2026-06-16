// EL/AD/XS/SY type sizes are defined in kernel2.h (included by the build system)

inline int num_atom_types(int atu) {
	switch (atu) {
	case 0: return EL_TYPE_SIZE;
	case 1: return AD_TYPE_SIZE;
	case 2: return XS_TYPE_SIZE;
	case 3: return SY_TYPE_SIZE;
	default: return INFINITY;
	}
}

inline void elementwise_product(float* out, const float* a, const float* b) {
	out[0] = a[0] * b[0];
	out[1] = a[1] * b[1];
	out[2] = a[2] * b[2];
}

inline float elementwise_product_sum(const float* a, const float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float access_m_data(const __global float* m_data, int m_i, int m_j, int i, int j, int k) {
	return m_data[i + m_i * (j + m_j * k)];
}

inline bool not_max(float x) {
	return (x < 0.1 * INFINITY);// Problem: replace max_fl with INFINITY?
}

inline void curl_with_deriv(float* e, float* deriv, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		*e *= tmp;
		for (int i = 0; i < 3; i++)deriv[i] *= pown(tmp, 2);
	}
}

inline void curl_without_deriv(float* e, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		*e *= tmp;
	}
}

float g_evaluate(			const __global	grid_cl*	g,
					const				float*		m_coords,			// double[3]
					const				float		slope,				// double
					const				float		v,					// double
										float*		deriv,				// double[3]
					const				float		epsilon_fl
) {
	int m_i = g->m_i;
	int m_j = g->m_j;
	int m_k = g->m_k;
	// (assert removed)
	float tmp_vec[3] = { m_coords[0] - g->m_init[0],m_coords[1] - g->m_init[1] ,m_coords[2] - g->m_init[2] };
	float tmp_vec2[3] = { g->m_factor[0],g->m_factor[1] ,g->m_factor[2] };
	float s[3];
	elementwise_product(s, tmp_vec, tmp_vec2); // 

	float miss[3] = { 0,0,0 };
	int region[3];
	int a[3];
	int m_data_dims[3] = { m_i,m_j,m_k };
	for (int i = 0; i < 3; i++){
		if (s[i] < 0) {
			miss[i] = -s[i];
			region[i] = -1;
			a[i] = 0;
			s[i] = 0;
		}
		else if (s[i] >= g->m_dim_fl_minus_1[i]) {
			miss[i] = s[i] - g->m_dim_fl_minus_1[i];
			region[i] = 1;
			// (assert removed)
			a[i] = m_data_dims[i] - 2;
			s[i] = 1;
		}
		else {
			region[i] = 0;
			a[i] = (int)s[i];
			s[i] -= a[i];
		}
		// (assert removed)
		// (assert removed)
		// (assert removed)
		// (assert removed)
	}

	float tmp_m_factor_inv[3] = { g->m_factor_inv[0],g->m_factor_inv[1],g->m_factor_inv[2] };
	const float penalty = slope * elementwise_product_sum(miss, tmp_m_factor_inv);
	// (assert removed)

	const int x0 = a[0];
	const int y0 = a[1];
	const int z0 = a[2];
		 
	const int x1 = x0 + 1;
	const int y1 = y0 + 1;
	const int z1 = z0 + 1;

	//const float f000 = access_m_data(g->m_data, m_i, m_j, x0, y0, z0);
	//const float f100 = access_m_data(g->m_data, m_i, m_j, x1, y0, z0);
	//const float f010 = access_m_data(g->m_data, m_i, m_j, x0, y1, z0);
	//const float f110 = access_m_data(g->m_data, m_i, m_j, x1, y1, z0);
	//const float f001 = access_m_data(g->m_data, m_i, m_j, x0, y0, z1);
	//const float f101 = access_m_data(g->m_data, m_i, m_j, x1, y0, z1);
	//const float f011 = access_m_data(g->m_data, m_i, m_j, x0, y1, z1);
	//const float f111 = access_m_data(g->m_data, m_i, m_j, x1, y1, z1);

	int base = (x0 + m_i * (y0 + m_j * z0)) * 8;
	const __global float* base_ptr = &g->m_data[base];

	const float f000 = *base_ptr;
	const float f100 = *(base_ptr + 1);
	const float f010 = *(base_ptr + 2);
	const float f110 = *(base_ptr + 3);
	const float f001 = *(base_ptr + 4);
	const float f101 = *(base_ptr + 5);
	const float f011 = *(base_ptr + 6);
	const float f111 = *(base_ptr + 7);

	const float x = s[0];
	const float y = s[1];
	const float z = s[2];
		  
	const float mx = 1 - x;
	const float my = 1 - y;
	const float mz = 1 - z;

	float f =
		f000 * mx * my * mz +
		f100 * x  * my * mz +
		f010 * mx * y  * mz +
		f110 * x  * y  * mz +
		f001 * mx * my * z	+
		f101 * x  * my * z	+
		f011 * mx * y  * z	+
		f111 * x  * y  * z  ;

	if (deriv) { // valid pointer
		const float x_g =
			f000 * (-1) * my * mz +
			f100 *   1  * my * mz +
			f010 * (-1) * y  * mz +
			f110 *	 1  * y  * mz +
			f001 * (-1) * my * z  +
			f101 *   1  * my * z  +
			f011 * (-1) * y  * z  +
			f111 *   1  * y  * z  ;


		const float y_g =
			f000 * mx * (-1) * mz +
			f100 * x  * (-1) * mz +
			f010 * mx *   1  * mz +
			f110 * x  *   1  * mz +
			f001 * mx * (-1) * z  +
			f101 * x  * (-1) * z  +
			f011 * mx *   1  * z  +
			f111 * x  *   1  * z  ;


		const float z_g =
			f000 * mx * my * (-1) +
			f100 * x  * my * (-1) +
			f010 * mx * y  * (-1) +
			f110 * x  * y  * (-1) +
			f001 * mx * my *   1  +
			f101 * x  * my *   1  +
			f011 * mx * y  *   1  +
			f111 * x  * y  *   1  ;

		float gradient[3] = { x_g, y_g, z_g };

		curl_with_deriv(&f, gradient, v, epsilon_fl);

		float gradient_everywhere[3];

		for (int i = 0; i < 3; i++) {
			gradient_everywhere[i] = ((region[i] == 0) ? gradient[i] : 0);
			deriv[i] = g->m_factor[i] * gradient_everywhere[i] + slope * region[i];
		}
		return f + penalty;
	}	
	else {  // none valid pointer
		curl_without_deriv(&f, v, epsilon_fl);
		return f + penalty;
	}
}


float ig_eval_deriv(						output_type_cl*		x,
											change_cl*			g,
						const				float				v,
								const __global	grids_cl*			grids,
											m_cl_private*		m,
						const				float				epsilon_fl
) {
	float e = 0;
	int nat = num_atom_types(grids->atu);

	// --- Standard atom-type affinity grids (unchanged Vina scoring) ---
	for (int i = 0; i < m->m_num_movable_atoms; i++) {
		int t = m->atoms[i].types[grids->atu];
		if (t >= nat) {
			for (int j = 0; j < 3; j++) m->minus_forces.coords[i][j] = 0;
			continue;
		}
		float deriv[3];
		e += g_evaluate(&grids->grids[t], m->m_coords.coords[i], grids->slope, v, deriv, epsilon_fl);
		for (int j = 0; j < 3; j++) m->minus_forces.coords[i][j] = deriv[j];
	}

	// --- QFD Phase 1: field-charge electrostatics + desolvation penalty ---
	// Only active when QFD grids were loaded (m_i > 0 for grid GRID_IDX_ESP).
	// Uses trilinear interpolation on precomputed receptor field grids.
	if (grids->grids[GRID_IDX_ESP].m_i > 0) {
		for (int i = 0; i < m->m_num_movable_atoms; i++) {
			float q = m->atoms[i].charge;
			if (q == 0.0f) continue;

			// Electrostatic coupling: E_elec = QFD_ELEC_WEIGHT * q * ESP(r)
			// Soft-saturation cap: E_out = raw/(1+|raw|/cap); deriv scaled by 1/(1+|raw|/cap)²
			// Prevents Mg/strong-field over-attraction without killing gradient direction.
			float esp_deriv[3];
			float esp_e = g_evaluate(&grids->grids[GRID_IDX_ESP],
			                         m->m_coords.coords[i],
			                         grids->slope, v, esp_deriv, epsilon_fl);
			float qw_elec = QFD_ELEC_WEIGHT * q;
			float raw_elec = qw_elec * esp_e;
			float cap_denom = 1.0f + fabs(raw_elec) / QFD_ATOM_E_CAP;
			e += raw_elec / cap_denom;
			float cap_scale = 1.0f / (cap_denom * cap_denom);
			for (int j = 0; j < 3; j++)
				m->minus_forces.coords[i][j] += qw_elec * esp_deriv[j] * cap_scale;

			// Desolvation penalty: E_desolv = QFD_DESOLV_WEIGHT * q² * DESOLV(r)
			float desolv_deriv[3];
			float desolv_e = g_evaluate(&grids->grids[GRID_IDX_DESOLV],
			                             m->m_coords.coords[i],
			                             grids->slope, v, desolv_deriv, epsilon_fl);
			float qw_desolv = QFD_DESOLV_WEIGHT * (q * q);
			e += qw_desolv * desolv_e;
			for (int j = 0; j < 3; j++)
				m->minus_forces.coords[i][j] += qw_desolv * desolv_deriv[j];
		}
	}

	// --- QFD Phase 1 addendum: information resonance field ---
	// Adds centre-of-charge coupling to the receptor's information field.
	// Only active when GRID_IDX_INFOMAP was loaded.
	// Phase 4 infomap: only polar atoms couple to the orientation-sensitivity field
	// Threshold 0.12 targets true donors/acceptors (O,N,S polar atoms; skips weak C)
	if (grids->grids[GRID_IDX_INFOMAP].m_i > 0) {
		for (int i = 0; i < m->m_num_movable_atoms; i++) {
			if (fabs(m->atoms[i].charge) < 0.12f) continue;
			float infomap_deriv[3];
			float infomap_e = g_evaluate(&grids->grids[GRID_IDX_INFOMAP],
			                              m->m_coords.coords[i],
			                              grids->slope, v, infomap_deriv, epsilon_fl);
			e += QFD_INFO_WEIGHT * infomap_e;
			for (int j = 0; j < 3; j++)
				m->minus_forces.coords[i][j] += QFD_INFO_WEIGHT * infomap_deriv[j];
		}
	}

	// --- QFD Phase 3: explicit water displacement penalty ---
	// Penalises ligand atoms that occupy known crystallographic water positions
	// without forming compensating H-bonds.  Active when GRID_IDX_WATER loaded.
	// The water grid value at each point = burial depth of the nearest xtal water.
	// Coupling is atom-universal (not charge-gated): all heavy atoms pay the penalty.
	if (grids->grids[GRID_IDX_WATER].m_i > 0) {
		for (int i = m->ligand.begin; i < m->m_num_movable_atoms; i++) {
			if (m->atoms[i].types[0] == EL_TYPE_H) continue;  // skip hydrogens
			float water_deriv[3];
			float water_e = g_evaluate(&grids->grids[GRID_IDX_WATER],
			                            m->m_coords.coords[i],
			                            grids->slope, v, water_deriv, epsilon_fl);
			e += QFD_WATER_WEIGHT * water_e;
			for (int j = 0; j < 3; j++)
				m->minus_forces.coords[i][j] += QFD_WATER_WEIGHT * water_deriv[j];
		}
	}

	// --- QFD Phase 5: π-π aromatic stacking grid ---
	// Rewards aromatic ligand atoms (AD type 'A', types[1]==AD_TYPE_A) near receptor
	// aromatic ring centres.  Active when qfd_pipi.bin is loaded.
	if (grids->grids[GRID_IDX_PIPI].m_i > 0) {
		for (int i = m->ligand.begin; i < m->m_num_movable_atoms; i++) {
			if (m->atoms[i].types[1] != AD_TYPE_A) continue;  // aromatic C only
			float pipi_deriv[3];
			float pipi_e = g_evaluate(&grids->grids[GRID_IDX_PIPI],
			                          m->m_coords.coords[i],
			                          grids->slope, v, pipi_deriv, epsilon_fl);
			e += QFD_PIPI_WEIGHT * pipi_e;
			for (int j = 0; j < 3; j++)
				m->minus_forces.coords[i][j] += QFD_PIPI_WEIGHT * pipi_deriv[j];
		}
	}

	// --- QFD Phase 6: cavity ---
	// The per-atom cavity GRADIENT is DISABLED. Two failed forms: (3) ungated additive → over-buries
	// good poses (~4.5 Å); (6) GATED gradient → does no harm but does NOT rescue (the long-range pull
	// is ineffective: a ligand stranded ~22 Å out sits in flat exposure beyond the field's reach).
	// The cavity grid is consumed only by the pose-level RESCUE gate (cavity_rescue) at the kernel2 MC
	// layer — a RE-RANKING effect (promote a sampled pocket pose over a solvent decoy), which is what
	// actually rescues failures (4DFR 22.9→~7 Å). See QFD_CAVITY_RESCUE_W in kernel2.h.

	return e;
}

// Phase 6b: pose-level cavity RESCUE penalty (energy only; NOT used in the BFGS gradient).
// Returns a non-negative penalty that is EXACTLY 0 unless a large fraction of the ligand's heavy
// atoms sit in solvent (exposure > threshold). Gated by smoothstep over that fraction so in-pocket
// poses are untouched (do-no-harm) and only badly-mislocalised poses are down-ranked. Sampled from
// the cavity exposure grid; no-op when the grid is absent (m_i == 0) or cavity disabled.
float cavity_rescue(m_cl_private* m, const __global grids_cl* grids, float v, float epsilon_fl) {
	if (grids->grids[GRID_IDX_CAVITY].m_i <= 0) return 0.0f;
	int   nheavy = 0, nexposed = 0;
	float sum_exp = 0.0f;
	for (int i = m->ligand.begin; i < m->m_num_movable_atoms; i++) {
		if (m->atoms[i].types[0] == EL_TYPE_H) continue;
		float dummy[3];
		float ex = g_evaluate(&grids->grids[GRID_IDX_CAVITY],
		                      m->m_coords.coords[i], grids->slope, v, dummy, epsilon_fl);
		nheavy++;
		sum_exp += ex;
		if (ex > QFD_CAV_EXPOSED_THR) nexposed++;
	}
	if (nheavy == 0) return 0.0f;
	float frac = (float)nexposed / (float)nheavy;
	float gate = (frac - QFD_CAV_GATE_LO) / (QFD_CAV_GATE_HI - QFD_CAV_GATE_LO);
	gate = gate < 0.0f ? 0.0f : (gate > 1.0f ? 1.0f : gate);   // smooth one-sided gate
	return QFD_CAVITY_RESCUE_W * gate * sum_exp;
}

inline void quaternion_to_r3(const float* q, float* orientation_m) {
	// Omit assert(quaternion_is_normalized(q));
	const float a = q[0];
	const float b = q[1];
	const float c = q[2];
	const float d = q[3];

	const float aa = a * a;
	const float ab = a * b;
	const float ac = a * c;
	const float ad = a * d;
	const float bb = b * b;
	const float bc = b * c;
	const float bd = b * d;
	const float cc = c * c;
	const float cd = c * d;
	const float dd = d * d;

	//Omit assert(eq(aa + bb + cc + dd, 1));
	matrix tmp;
	mat_init(&tmp, 0); // matrix with fixed dimension 3(here we treate this as a regular matrix(not triangular matrix!))

	matrix_set_element(&tmp, 3, 0, 0,		(aa + bb - cc - dd)	);
	matrix_set_element(&tmp, 3, 0, 1, 2 *	(-ad + bc)			);
	matrix_set_element(&tmp, 3, 0, 2, 2 *	(ac + bd)			);
							 
	matrix_set_element(&tmp, 3, 1, 0, 2 *	(ad + bc)			);
	matrix_set_element(&tmp, 3, 1, 1,		(aa - bb + cc - dd)	);
	matrix_set_element(&tmp, 3, 1, 2, 2 *	(-ab + cd)			);
							 
	matrix_set_element(&tmp, 3, 2, 0, 2 *	(-ac + bd)			);
	matrix_set_element(&tmp, 3, 2, 1, 2 *	(ab + cd)			);
	matrix_set_element(&tmp, 3, 2, 2,		(aa - bb - cc + dd)	);

	for (int i = 0; i < 9; i++) orientation_m[i] = tmp.data[i];
}

inline void local_to_lab_direction(			float* out,
									const	float* local_direction,
									const	float* orientation_m
) {
	out[0] =	orientation_m[0] * local_direction[0] +
				orientation_m[3] * local_direction[1] +
				orientation_m[6] * local_direction[2];
	out[1] =	orientation_m[1] * local_direction[0] +
				orientation_m[4] * local_direction[1] +
				orientation_m[7] * local_direction[2];
	out[2] =	orientation_m[2] * local_direction[0] +
				orientation_m[5] * local_direction[1] +
				orientation_m[8] * local_direction[2];
}

inline void local_to_lab(						float*		out,
							const				float*		origin,
							const				float*		local_coords,
							const				float*		orientation_m
) {
	out[0] = origin[0] + (	orientation_m[0] * local_coords[0] +
							orientation_m[3] * local_coords[1] +
							orientation_m[6] * local_coords[2]
							);			 
	out[1] = origin[1] + (	orientation_m[1] * local_coords[0] +
							orientation_m[4] * local_coords[1] +
							orientation_m[7] * local_coords[2]
							);			 
	out[2] = origin[2] + (	orientation_m[2] * local_coords[0] +
							orientation_m[5] * local_coords[1] +
							orientation_m[8] * local_coords[2]
							);
}

inline void angle_to_quaternion2(				float*		out,
									const		float*		axis,
												float		angle
) {
	// (norm check assert removed — fires per-step on all MCMC threads)
	normalize_angle(&angle);
	float c = cos(angle / 2);
	float s = sin(angle / 2);
	out[0] = c;
	out[1] = s * axis[0];
	out[2] = s * axis[1];
	out[3] = s * axis[2];
}

void set(	const				output_type_cl* x,
								rigid_cl*		lig_rigid_gpu,
								m_coords_cl*	m_coords_gpu,	
			const				atom_cl*		atoms,				
			const				int				m_num_movable_atoms,
			const				float			epsilon_fl
) {
	//************** (root) node.set_conf **************// (CHECKED)
	for (int i = 0; i < 3; i++) lig_rigid_gpu->origin[0][i] = x->position[i]; // set origin
	for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[0][i] = x->orientation[i]; // set orientation_q
	quaternion_to_r3(lig_rigid_gpu->orientation_q[0], lig_rigid_gpu->orientation_m[0]);// set orientation_m
	// set coords
	int begin = lig_rigid_gpu->atom_range[0][0];
	int end =	lig_rigid_gpu->atom_range[0][1];
	for (int i = begin; i < end; i++) {
		local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[0], &atoms[i].coords[0], lig_rigid_gpu->orientation_m[0]);
	}
	//************** end node.set_conf **************//

	//************** branches_set_conf **************//
	//update nodes in depth-first order
	for (int current = 1; current < lig_rigid_gpu->num_children + 1; current++) { // current starts from 1 (namely starts from first child node)
		int parent = lig_rigid_gpu->parent[current];
		float torsion = x->lig_torsion[current - 1]; // torsions are all related to child nodes
		local_to_lab(	lig_rigid_gpu->origin[current],
						lig_rigid_gpu->origin[parent],
						lig_rigid_gpu->relative_origin[current],
						lig_rigid_gpu->orientation_m[parent]
						); // set origin
		local_to_lab_direction(	lig_rigid_gpu->axis[current],
								lig_rigid_gpu->relative_axis[current],
								lig_rigid_gpu->orientation_m[parent]
								); // set axis
		float tmp[4];
		float parent_q[4] = {	lig_rigid_gpu->orientation_q[parent][0],
								lig_rigid_gpu->orientation_q[parent][1] ,
								lig_rigid_gpu->orientation_q[parent][2] ,
								lig_rigid_gpu->orientation_q[parent][3] };
		float current_axis[3] = {	lig_rigid_gpu->axis[current][0],
									lig_rigid_gpu->axis[current][1],
									lig_rigid_gpu->axis[current][2] };

		angle_to_quaternion2(tmp, current_axis, torsion);
		angle_to_quaternion_multi(tmp, parent_q);
		quaternion_normalize_approx(tmp, epsilon_fl);

		for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[current][i] = tmp[i]; // set orientation_q
		quaternion_to_r3(lig_rigid_gpu->orientation_q[current], lig_rigid_gpu->orientation_m[current]); // set orientation_m

		// set coords
		begin = lig_rigid_gpu->atom_range[current][0];
		end =	lig_rigid_gpu->atom_range[current][1];
		for (int i = begin; i < end; i++) {
			local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[current], &atoms[i].coords[0], lig_rigid_gpu->orientation_m[current]);
		}
	}
	//************** end branches_set_conf **************//
}

// Stage 1: flexible side-chain forward kinematics.
// Walks the flex forest (anchors before their children, depth-first). Anchor nodes
// (parent < 0) keep their FIXED input frame; torsion nodes rotate about `axis` by
// flex_torsion[torsion_idx], exactly like set()'s branch loop. Writes lab coords for
// flex atoms into m_coords at their atom_range indices. No-op when num_nodes == 0.
void set_flex(	const				output_type_cl*	x,
								flex_rigid_cl*	flex,
								m_coords_cl*	m_coords_gpu,
				const				atom_cl*		atoms,
				const				float			epsilon_fl
) {
	for (int n = 0; n < flex->num_nodes; n++) {
		int par = flex->parent[n];
		float torsion = x->flex_torsion[flex->torsion_idx[n]];
		float tmp[4];
		if (par < 0) {
			// first_segment (residue root): FIXED origin + FIXED axis (loaded from CPU);
			// parent orientation is identity, so orientation_q = angle_to_quaternion(axis, torsion).
			float root_axis[3] = { flex->axis[n][0], flex->axis[n][1], flex->axis[n][2] };
			angle_to_quaternion2(tmp, root_axis, torsion);
			quaternion_normalize_approx(tmp, epsilon_fl);
			// origin[n] stays at its fixed loaded value.
		} else {
			// child segment: derive origin/axis from parent, rotate by its flex torsion
			local_to_lab(			flex->origin[n],
									flex->origin[par],
									flex->relative_origin[n],
									flex->orientation_m[par]);
			local_to_lab_direction(	flex->axis[n],
									flex->relative_axis[n],
									flex->orientation_m[par]);
			float parent_q[4] = {	flex->orientation_q[par][0],
									flex->orientation_q[par][1],
									flex->orientation_q[par][2],
									flex->orientation_q[par][3] };
			float current_axis[3] = { flex->axis[n][0], flex->axis[n][1], flex->axis[n][2] };
			angle_to_quaternion2(tmp, current_axis, torsion);
			angle_to_quaternion_multi(tmp, parent_q);
			quaternion_normalize_approx(tmp, epsilon_fl);
		}
		for (int i = 0; i < 4; i++) flex->orientation_q[n][i] = tmp[i];
		quaternion_to_r3(flex->orientation_q[n], flex->orientation_m[n]);

		int begin = flex->atom_range[n][0];
		int end   = flex->atom_range[n][1];
		for (int i = begin; i < end; i++) {
			local_to_lab(m_coords_gpu->coords[i], flex->origin[n], &atoms[i].coords[0], flex->orientation_m[n]);
		}
	}
}

void p_eval_deriv(						float*		out,
										int			type_pair_index,
										float		r2,
							const __global	pre_cl*		pre,
					const				float		epsilon_fl
) {
	const float cutoff_sqr = pre->m_cutoff_sqr;
	if(r2 > cutoff_sqr) { out[0] = 0.0f; out[1] = 0.0f; return; }
	const __global p_m_data_cl* tmp = &pre->m_data[type_pair_index];
	float r2_factored = tmp->factor * r2;
	if (r2_factored + 1 >= SMOOTH_SIZE) { out[0] = 0.0f; out[1] = 0.0f; return; }
	int i1 = (int)(r2_factored);
	int i2 = i1 + 1;
	if (i1 >= SMOOTH_SIZE || i1 < 0) { out[0] = 0.0f; out[1] = 0.0f; return; }
	if (i2 >= SMOOTH_SIZE || i2 < 0) { out[0] = 0.0f; out[1] = 0.0f; return; }
	float rem = r2_factored - i1;
	if (rem < -epsilon_fl) { rem = 0.0f; }
	if (rem >= 1 + epsilon_fl) { rem = 1.0f - epsilon_fl; }
	float p1[2] = { tmp->smooth[i1][0], tmp->smooth[i1][1] };
	float p2[2] = { tmp->smooth[i2][0], tmp->smooth[i2][1] };
	float e = p1[0] + rem * (p2[0] - p1[0]);
	float dor = p1[1] + rem * (p2[1] - p1[1]);
	out[0] = e;
	out[1] = dor;
}

inline void curl(float* e, float* deriv, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		(*e) = tmp * (*e);
		for (int i = 0; i < 3; i++)deriv[i] = deriv[i] * (tmp * tmp);
	}
}

// Raw-array form so it serves BOTH ligand intra-pairs and flex other_pairs (Stage 2).
float eval_interacting_pairs_deriv(			const __global	pre_cl*			pre,
									const				float			v,
									const				int				num_pairs,
									const __global		int*			tpi,
									const __global		int*			pa,
									const __global		int*			pb,
									const			 	m_coords_cl*	m_coords,
														m_minus_forces* minus_forces,
									const				float			epsilon_fl
) {
	float e = 0;

	for (int i = 0; i < num_pairs; i++) {
		const int ip[3] = { tpi[i], pa[i], pb[i] };
		float coords_b[3] = { m_coords->coords[ip[2]][0], m_coords->coords[ip[2]][1], m_coords->coords[ip[2]][2] };
		float coords_a[3] = { m_coords->coords[ip[1]][0], m_coords->coords[ip[1]][1], m_coords->coords[ip[1]][2] };
		float r[3] = { coords_b[0] - coords_a[0], coords_b[1] - coords_a[1] ,coords_b[2] - coords_a[2] };
		float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
	
		if (r2 < pre->m_cutoff_sqr) {
			float tmp[2];
			p_eval_deriv(tmp, ip[0], r2, pre, epsilon_fl);
			float force[3] = { r[0] * tmp[1], r[1] * tmp[1] ,r[2] * tmp[1] };
			curl(&tmp[0], force, v, epsilon_fl);
			e += tmp[0];
			for (int j = 0; j < 3; j++)minus_forces->coords[ip[1]][j] -= force[j];
			for (int j = 0; j < 3; j++)minus_forces->coords[ip[2]][j] += force[j];
		}
	}
	return e;
}

inline void product(float* res, const float*a,const float*b) {
	res[0] = a[1] * b[2] - a[2] * b[1];
	res[1] = a[2] * b[0] - a[0] * b[2];
	res[2] = a[0] * b[1] - a[1] * b[0];
}

void POT_deriv(	const					m_minus_forces* minus_forces,
				const					rigid_cl*		lig_rigid_gpu,
				const					m_coords_cl*		m_coords,
										change_cl*		g
) {
	int num_torsion = lig_rigid_gpu->num_children;
	int num_rigid = num_torsion + 1;
	float position_derivative_tmp[MAX_NUM_OF_RIGID][3];
	float position_derivative[MAX_NUM_OF_RIGID][3];
	float orientation_derivative_tmp[MAX_NUM_OF_RIGID][3];
	float orientation_derivative[MAX_NUM_OF_RIGID][3];
	float torsion_derivative[MAX_NUM_OF_RIGID]; // torsion_derivative[0] has no meaning(root node has no torsion)

	for (int i = 0; i < num_rigid; i++) {
		int begin = lig_rigid_gpu->atom_range[i][0];
		int end = lig_rigid_gpu->atom_range[i][1];
		for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] = 0; 
		for (int k = 0; k < 3; k++)orientation_derivative_tmp[i][k] = 0;
		for (int j = begin; j < end; j++) {
			for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] += minus_forces->coords[j][k];

			float tmp1[3] = {	m_coords->coords[j][0] - lig_rigid_gpu->origin[i][0],
								m_coords->coords[j][1] - lig_rigid_gpu->origin[i][1],
								m_coords->coords[j][2] - lig_rigid_gpu->origin[i][2] };
			float tmp2[3] = {  minus_forces->coords[j][0],
								minus_forces->coords[j][1],
								minus_forces->coords[j][2] };
			float tmp3[3];
			product(tmp3, tmp1, tmp2);
			for (int k = 0; k < 3; k++)orientation_derivative_tmp[i][k] += tmp3[k];
		}
	}

	// position_derivative 
	for (int i = num_rigid - 1; i >= 0; i--) {// from bottom to top
		for (int k = 0; k < 3; k++)position_derivative[i][k] = position_derivative_tmp[i][k]; // self
		// looking for children node: j is a child of i iff parent[j]==i (j>=1; root has no parent)
		for (int j = 1; j < num_rigid; j++) {
			if (lig_rigid_gpu->parent[j] == i) {
				for (int k = 0; k < 3; k++)position_derivative[i][k] += position_derivative[j][k]; // self+children node
			}
		}
	}

	// orientation_derivetive
	for (int i = num_rigid - 1; i >= 0; i--) { // from bottom to top
		for (int k = 0; k < 3; k++)orientation_derivative[i][k] = orientation_derivative_tmp[i][k]; // self
		// looking for children node: j is a child of i iff parent[j]==i (j>=1; root has no parent)
		for (int j = 1; j < num_rigid; j++) {
			if (lig_rigid_gpu->parent[j] == i) { // self + children node + product
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += orientation_derivative[j][k];
				float product_out[3];
				float origin_temp[3] = {	lig_rigid_gpu->origin[j][0] - lig_rigid_gpu->origin[i][0],
											lig_rigid_gpu->origin[j][1] - lig_rigid_gpu->origin[i][1],
											lig_rigid_gpu->origin[j][2] - lig_rigid_gpu->origin[i][2] };
				product(product_out, origin_temp, position_derivative[j]);
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += product_out[k];
			}
		}
	}

	// torsion_derivative
	for (int i = num_rigid - 1; i >= 0; i--) { // from bottom to top
		float sum = 0;
		for (int j = 0; j < 3; j++) sum += orientation_derivative[i][j] * lig_rigid_gpu->axis[i][j];
		torsion_derivative[i] = sum;
	}

	for (int k = 0; k < 3; k++)	g->position[k] = position_derivative[0][k];
	for (int k = 0; k < 3; k++) g->orientation[k] = orientation_derivative[0][k];
	for (int k = 0; k < num_torsion; k++) g->lig_torsion[k] = torsion_derivative[k + 1];// no meaning for node 0
}

// Stage 3: flex side-chain torsion gradients. Mirrors POT_deriv but over the flex forest
// (uses parent[] instead of children_map; DFS order ⇒ parent index < child index, so
// accumulating high→low into the parent gathers each subtree's force/torque). Every flex node
// has a torsion (first_segment rotates about its fixed axis too), so all get a gradient.
// Writes g->flex_torsion[torsion_idx]. No-op when num_nodes==0 (rigid).
void POT_deriv_flex(	const	m_minus_forces*	minus_forces,
						const	flex_rigid_cl*	flex,
						const	m_coords_cl*	m_coords,
								change_cl*		g
) {
	int N = flex->num_nodes;
	if (N <= 0) return;
	float pos_d[MAX_NUM_OF_FLEX_RIGID][3];
	float ori_d[MAX_NUM_OF_FLEX_RIGID][3];
	for (int i = 0; i < N; i++) {
		int begin = flex->atom_range[i][0];
		int end   = flex->atom_range[i][1];
		float pd[3] = {0,0,0};
		float od[3] = {0,0,0};
		for (int j = begin; j < end; j++) {
			for (int k = 0; k < 3; k++) pd[k] += minus_forces->coords[j][k];
			float tmp1[3] = {	m_coords->coords[j][0] - flex->origin[i][0],
								m_coords->coords[j][1] - flex->origin[i][1],
								m_coords->coords[j][2] - flex->origin[i][2] };
			float tmp2[3] = {	minus_forces->coords[j][0],
								minus_forces->coords[j][1],
								minus_forces->coords[j][2] };
			float tmp3[3];
			product(tmp3, tmp1, tmp2);
			for (int k = 0; k < 3; k++) od[k] += tmp3[k];
		}
		for (int k = 0; k < 3; k++) { pos_d[i][k] = pd[k]; ori_d[i][k] = od[k]; }
	}
	// gather each subtree's contribution into its parent (children have higher index)
	for (int i = N - 1; i >= 0; i--) {
		int p = flex->parent[i];
		if (p >= 0) {
			float origin_temp[3] = {	flex->origin[i][0] - flex->origin[p][0],
										flex->origin[i][1] - flex->origin[p][1],
										flex->origin[i][2] - flex->origin[p][2] };
			float product_out[3];
			product(product_out, origin_temp, pos_d[i]);
			for (int k = 0; k < 3; k++) {
				pos_d[p][k] += pos_d[i][k];
				ori_d[p][k] += ori_d[i][k] + product_out[k];
			}
		}
	}
	for (int i = 0; i < N; i++) {
		float sum = 0;
		for (int k = 0; k < 3; k++) sum += ori_d[i][k] * flex->axis[i][k];
		g->flex_torsion[flex->torsion_idx[i]] = sum;
	}
}


float m_eval_deriv(						output_type_cl*			c,
										change_cl*				g,
										m_cl_private*			m,
						const __global	lig_pairs_cl*			pairs_g,
						const __global	other_pairs_cl*			other_g,
							const __global	pre_cl*				pre,
							const __global	grids_cl*			grids,
					const	__global	float*					v,
					const				float					epsilon_fl
) {
	set(c, &m->ligand.rigid, &m->m_coords, m->atoms, m->m_num_movable_atoms, epsilon_fl);
	// Stage 2: update flexible side-chain coords from the current conf BEFORE scoring,
	// so other_pairs (ligand↔flex, flex↔flex) and the flex↔receptor grid term are current.
	set_flex(c, &m->flex_rigid, &m->m_coords, m->atoms, epsilon_fl);

	float e = ig_eval_deriv(	c,
								g,
								v[1],
								grids,
								m,
								epsilon_fl
							);

	e += eval_interacting_pairs_deriv(	pre, v[0],
										pairs_g->num_pairs, pairs_g->type_pair_index, pairs_g->a, pairs_g->b,
										&m->m_coords,
										&m->minus_forces,
										epsilon_fl
									);

	// Stage 2: ligand↔side-chain + side-chain↔side-chain energy (no-op when num_pairs==0)
	e += eval_interacting_pairs_deriv(	pre, v[0],
										other_g->num_pairs, other_g->type_pair_index, other_g->a, other_g->b,
										&m->m_coords,
										&m->minus_forces,
										epsilon_fl
									);

	POT_deriv(&m->minus_forces, &m->ligand.rigid, &m->m_coords, g);
	// Stage 3: flex side-chain torsion gradients (no-op when no flex)
	POT_deriv_flex(&m->minus_forces, &m->flex_rigid, &m->m_coords, g);

	return e;
}


// Stage 3: `tcount` packs torsion sizes as (lig_torsion_size | flex_torsion_size<<16).
// Backward compatible: a raw lig_torsion_size has zero high bits ⇒ flex=0 (rigid/dual unaffected).
// Change vector layout: position[3], orientation[3], lig_torsion[lig], flex_torsion[flex].
inline float find_change_index_read(const change_cl* g, int index, int tcount) {
	int lig_torsion_size = tcount & 0xFFFF;
	int flex_torsion_size = (tcount >> 16) & 0xFFFF;
	if (index < 3)return g->position[index];
	index -= 3;
	if (index < 3)return g->orientation[index];
	index -= 3;
	if (index < lig_torsion_size)return g->lig_torsion[index]; /* assert removed */
	index -= lig_torsion_size;
	if (index < flex_torsion_size)return g->flex_torsion[index];
	return -1;
}

inline void find_change_index_write(change_cl* g, int index, float data, int tcount) {
	int lig_torsion_size = tcount & 0xFFFF;
	int flex_torsion_size = (tcount >> 16) & 0xFFFF;
	if (index < 3) { g->position[index] = data; return; }
	index -= 3;
	if (index < 3) { g->orientation[index] = data; return; }
	index -= 3;
	if (index < lig_torsion_size) { g->lig_torsion[index] = data; return; } /* assert removed */
	index -= lig_torsion_size;
	if (index < flex_torsion_size) { g->flex_torsion[index] = data; return; }
}

void minus_mat_vec_product(	const		matrix*		h,
							const		change_cl*	in,
										change_cl*  out,
							const		int			lig_torsion_size
) {
	int n = h->dim;
	for (int i = 0; i < n; i++) {
		float sum = 0;
		for (int j = 0; j < n; j++) {
			sum += h->data[index_permissive(h, i, j)] * find_change_index_read(in, j, lig_torsion_size);
		}
		find_change_index_write(out, i, -sum, lig_torsion_size);
	}
}


inline float scalar_product(	const	change_cl*	a,
								const	change_cl*	b,
								const	int			n,
								const	int			lig_torsion_size
) {
	float tmp = 0;
	for (int i = 0; i < n; i++) {
		tmp += find_change_index_read(a, i, lig_torsion_size) * find_change_index_read(b, i, lig_torsion_size);
	}
	return tmp;
}

inline void to_minus(change_cl* out, const int n)
{
	// out = -out
	for (int i = 0; i < 3; i++)out->position[i] = -out->position[i];
	for (int i = 0; i < 3; i++)out->orientation[i] = -out->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)out->lig_torsion[i] = -out->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)out->flex_torsion[i] = -out->flex_torsion[i];
}
inline void get_to_minus(change_cl* a, change_cl* b, const int n)
{
	for (int i = 0; i < 3; i++)a->position[i] = -b->position[i];
	for (int i = 0; i < 3; i++)a->orientation[i] = -b->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)a->lig_torsion[i] = -b->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)a->flex_torsion[i] = -b->flex_torsion[i];
}
inline float to_norm(const change_cl* in, const int n, const int torsion_size)
{
	float d_test = 0;
	for (int i = 0; i < n; i++)
	{
		d_test += find_change_index_read(in, i, torsion_size) * find_change_index_read(in, i, torsion_size);
	}
	d_test = sqrt(d_test);
	return d_test;
}


inline int line_search_lewisoverton(				m_cl_private*	m_cl_gpu,
							const __global	lig_pairs_cl*	pairs_g,
							const __global	other_pairs_cl*	other_g,
											const __global 	pre_cl* 		p_cl_gpu,
											const __global 	grids_cl* 		ig_cl_gpu,
														int 			n,
														float* 			stp,
														output_type_cl* x,
														float* 			f,
														change_cl* 		g,
									const 				change_cl* 		d,
									const 				output_type_cl* xp,
									const 				change_cl* 		gp,
									const 				float 			epsilon_fl,
									const 	__global 	float* 			hunt_cap,
									const 				int 			torsion_size
	)
{
	int count = 0;
	bool brackt = false;
	float finit, dginit, dgtest, dstest;
	float mu = 0.0, nu = 1.0e+20;

	dginit = scalar_product(gp, d, n, torsion_size);

	/* Make sure that s points to a descent direction. */
	if (0.0 < dginit)
	{
		return -1;
	}
	/* The initial value of the cost function. */
	finit = *f;
	//F_DEC_COEFF
	dgtest = 1.0e-4 * dginit;
	// S_CURV_COEFF 
	dstest = 0.1 * dginit;

	while (true)
	{
		//output_type_cl_init_with_output(x, xp);
		*x = *xp;
		output_type_cl_increment(x, d, *stp, epsilon_fl, torsion_size);
		/* Evaluate the function and gradient values. */
		*f = m_eval_deriv(x,
			g,
			m_cl_gpu,
			pairs_g,
			other_g,
			p_cl_gpu,
			ig_cl_gpu,
			hunt_cap,
			epsilon_fl
		);
		++count;
		/* Check the Armijo condition. */
		if (*f > finit + *stp * dgtest)
		{
			nu = *stp;
			brackt = true;
		}
		else
		{
			if (scalar_product(g, d, n, torsion_size) < dstest)
			{
				mu = *stp;
			}
			else
			{
				return count;
			}
		}
		// MAX_LINESEARCH
		if (10 <= count)
		{
			if (*f > finit)
				return -1;
			else
				return count;
		}
		if (brackt)
		{
			(*stp) = 0.5 * (mu + nu);
		}
		else
		{
			(*stp) *= 2.0;
		}
	}
}

float line_search(					 	m_cl_private*		m,
						const __global	lig_pairs_cl*		pairs_g,
						const __global	other_pairs_cl*		other_g,
							const __global	pre_cl*				pre,
							const __global	grids_cl*			grids,
										int					n,
					const				output_type_cl*		x,
					const				change_cl*			g,
					const				float				f0,
					const				change_cl*			p,
										output_type_cl*		x_new,
										change_cl*			g_new,
										float*				f1,
					const				float				epsilon_fl,
					const	__global	float*				hunt_cap,
					const				int					lig_torsion_size
) {
	const float c0 = 0.0001;
	const int max_trials = 10;
	const float multiplier = 0.5;
	float alpha = 1;

	const float pg = scalar_product(p, g, n, lig_torsion_size);

	for (int trial = 0; trial < max_trials; trial++) {

		*x_new =  *x;

		output_type_cl_increment(x_new, p, alpha, epsilon_fl, lig_torsion_size);

		*f1 =  m_eval_deriv(x_new,
							g_new,
							m,
							pairs_g,
							other_g,
							pre,
							grids,
							hunt_cap,
							epsilon_fl
							);
		// printf("alpha: %f", alpha);


		if (*f1 - f0 < c0 * alpha * pg)
			break;
		alpha *= multiplier;
	}
	// if (*f1 - f0> 0)
	//
	// else
	//
	return alpha;
}


bool bfgs_update(			matrix*			h,
					const	change_cl*		p,
					const	change_cl*		y,
					const	float			alpha,
					const	mis_cl*			mis,
					const	int				torsion_size
) {
	const float yp = scalar_product(y, p, h->dim, torsion_size);

	if (alpha * yp < mis->epsilon_fl) return false;

	change_cl minus_hy = *y;
	minus_mat_vec_product(h, y, &minus_hy, torsion_size);
	const float yhy = -scalar_product(y, &minus_hy, h->dim, torsion_size);
	const float r = 1 / (alpha * yp);
	const int n = 6 + (torsion_size & 0xFFFF) + ((torsion_size >> 16) & 0xFFFF);  // Stage 3: unpack lig|flex
	int s = torsion_size;
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			float tmp = alpha * r * (find_change_index_read(&minus_hy, i, s) * find_change_index_read(p, j,s)
									+ find_change_index_read(&minus_hy, j, s) * find_change_index_read(p, i,s)) +
									+alpha * alpha * (r * r * yhy + r) * find_change_index_read(p, i,s) * find_change_index_read(p, j, s);

			h->data[i + j * (j + 1) / 2] += tmp;
		}
	}

	return true;
}

void rilc_bfgs(				output_type_cl* 	x,
						change_cl* 			g,
						m_cl_private*		m_cl_gpu,
			const __global	lig_pairs_cl*		pairs_g,
			const __global	other_pairs_cl*		other_g,
			const __global	pre_cl* 			p_cl_gpu,
			const __global	grids_cl* 			ig_cl_gpu,
	const	__global	mis_cl*				mis,
	const				int					torsion_size,
	const				int					max_steps
)
{

	int n = 6 + (torsion_size & 0xFFFF) + ((torsion_size >> 16) & 0xFFFF); // Stage 3: unpack lig|flex dims

	// rilc_bfgs init
	int k, ls;
	float step, fx, ys, yy, step_min, step_max;
	float beta = 0, cau;
	float lm_s_dot_d, cau_t;
	float d_update_tmp = 0;

	output_type_cl xp = *x;
	change_cl gp = *g;


	float lm_alpha = 0;
	float lm_s[MAX_CONF_DIM] = { 0 };   // L-BFGS history vector, indexed over conf dim n (not n^2/2)
	float lm_y[MAX_CONF_DIM] = { 0 };
	float lm_ys = 0;
	fx = m_eval_deriv(	x,
						g,
						m_cl_gpu,
						pairs_g,
						other_g,
						p_cl_gpu,
						ig_cl_gpu,
						mis->hunt_cap,
						mis->epsilon_fl
	);
	float fxp = fx;
	float fx_orig = fx;
	change_cl g_orig = *g;
	output_type_cl x_orig = *x;

	change_cl d = *g;

	//d = -g
	get_to_minus(&d, g, n);

	if (!(sqrt(scalar_product(g, g, n, torsion_size)) >= 1e-5))
	{
		x->e = fx;
		return;
	}
	else
	{
		step = 1.0 / to_norm(&d, n, torsion_size);
		k = 1;

		while (true) {

			xp = *x;
			gp = *g;

			fxp = fx;


			ls = line_search_lewisoverton(m_cl_gpu,
				pairs_g,
				other_g,
				p_cl_gpu,
				ig_cl_gpu,
				n,
				&step,
				x,
				&fx,
				g,
				&d,
				&xp,
				&gp,
				mis->epsilon_fl,
				mis->hunt_cap,
				torsion_size
			);
			if (ls < 0) {

				*x = xp;
				*g = gp;

				fx = fxp;
				break;
			}
			if (!(sqrt(scalar_product(g, g, n, torsion_size)) >= 1e-5)) {
				x->e = fx;
				return;
			}
			if (max_steps != 0 && max_steps <= k) {
				break;
			}
			/* Count the iteration number. */
			++k;

			for (int i = 0; i < n; i++) {
				lm_s[i] = step * find_change_index_read(&d, i, torsion_size);
			}
			for (int i = 0; i < n; i++) {
				lm_y[i] = find_change_index_read(g, i, torsion_size) - find_change_index_read(&gp, i, torsion_size);
			}

			ys = 0;
			for (int i = 0; i < n; i++) {
				ys += lm_y[i] * lm_s[i];
			}

			yy = 0;
			for (int i = 0; i < n; i++) {
				yy += lm_y[i] * lm_y[i];
			}

			lm_ys = ys;

			/* Compute the negative of gradients. */
			// d = -g
			get_to_minus(&d, g, n);
			cau = 0;
			for (int i = 0; i < n; i++) {
				cau += lm_s[i] * lm_s[i];
			}
			cau_t = 0;
			for (int i = 0; i < n; i++)
			{
				cau_t = cau_t + (find_change_index_read(&gp, i, torsion_size) * find_change_index_read(&gp, i, torsion_size));
			}
			cau = cau * sqrt(cau_t);
			// CAUTIOUS_FACTOR
			cau = cau * 1.0e-6;

			if (ys > cau) {
				/* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
				lm_s_dot_d = 0;
				for (int a = 0; a < n; a++)
				{
					lm_s_dot_d = lm_s_dot_d + lm_s[a] * find_change_index_read(&d, a, torsion_size);
				}
				lm_alpha = lm_s_dot_d / lm_ys;
				for (int b = 0; b < n; b++)
				{
					float mimus_lm_alpha_mulit_lm_y = (-lm_alpha * lm_y[b]);
					find_change_index_write(&d, b, find_change_index_read(&d, b, torsion_size) + lm_alpha * lm_y[b], torsion_size);
				}


				// d *= ys / yy;
				for (int i = 0; i < n; i++)
				{
					find_change_index_write(&d, i, find_change_index_read(&d, i, torsion_size) * (ys / yy), torsion_size);
				}

				beta = 0;
				for (int a = 0; a < n; a++)
				{
					beta += (lm_y[a] * find_change_index_read(&d, a, torsion_size));
				}
				beta /= lm_ys;


				for (int i = 0; i < n; i++)
				{
					find_change_index_write(&d, i, find_change_index_read(&d, i, torsion_size) + ((lm_alpha - beta) * lm_s[i]), torsion_size);
				}

			}
			step = 1.0;
		}

	}

	//  rilc_bfgs init

	if (!(fx <= fx_orig)) {
		fx = fx_orig;

		*x = x_orig;

		*g = g_orig;

	}

	// write output_type_cl energy
	x->e = fx;
}

void bfgs(					output_type_cl*			x,
								change_cl*			g,
								m_cl_private*		m,
					const __global	lig_pairs_cl*	pairs_g,
					const __global	other_pairs_cl*	other_g,
					const __global	pre_cl*				pre,
					const __global	grids_cl*			grids,
			const	__global	mis_cl*				mis,
			const				int					torsion_size,
			const				int					max_bfgs_steps
)
{
	int lig_torsion_size = torsion_size;  // packed (lig|flex<<16) when called from single-ligand kernel
	int n = 6 + (torsion_size & 0xFFFF) + ((torsion_size >> 16) & 0xFFFF); // Stage 3: unpack lig|flex dims

	matrix h;
	matrix_init(&h, n, 0);
	matrix_set_diagonal(&h, 1);

	change_cl g_new = *g;

	output_type_cl x_new = *x ;
	float f0 = m_eval_deriv(	x,
								g,
								m,
								pairs_g,
								other_g,
								pre,
								grids,
								mis->hunt_cap,
								mis->epsilon_fl
							);

	float f_orig = f0;

	change_cl g_orig = *g;
	output_type_cl x_orig = *x;

	change_cl p = *g;

	for (int step = 0; step < max_bfgs_steps; step++) {

		minus_mat_vec_product(&h, g, &p, torsion_size);
		float f1 = 0;

		const float alpha = line_search(	m,
											pairs_g,
											other_g,
											pre,
											grids,
											n,
											x,
											g,
											f0,
											&p,
											&x_new,
											&g_new,
											&f1,
											mis->epsilon_fl,
											mis->hunt_cap,
											torsion_size
										);
		
		change_cl y = g_new;
		// subtract_change
		for (int i = 0; i < n; i++) {
			float tmp = find_change_index_read(&y, i, torsion_size) - find_change_index_read(g, i, torsion_size);
			find_change_index_write(&y, i, tmp, torsion_size);
		}
		
		f0 = f1;
		*x = x_new;
		if (!(sqrt(scalar_product(g, g, n, torsion_size)) >= 1e-5))break;
		*g = g_new;

		if (step == 0) {
			float yy = scalar_product(&y, &y, n, torsion_size);
			if (fabs(yy) > mis->epsilon_fl) {
				matrix_set_diagonal(&h, alpha * scalar_product(&y, &p, n, torsion_size) / yy);
			}
		}
		bool h_updated = bfgs_update(&h, &p, &y, alpha, mis, torsion_size);
	}

	if (!(f0 <= f_orig)) {
		f0 = f_orig;
		*x = x_orig;
		*g = g_orig;
	}

	// write output_type_cl energy
	x->e = f0;
}

// ════════════════════════ Native multi-ligand (P1) device core ════════════════════════
// ADDITIVE: every single-ligand function above is untouched. These take raw float(*)[3]
// pointers so they work over the larger m_multi_cl arrays, and reuse the type-agnostic leaf
// helpers (local_to_lab, local_to_lab_direction, angle_to_quaternion2, angle_to_quaternion_multi,
// quaternion_to_r3, quaternion_normalize_approx, quaternion_increment, normalize_angle,
// g_evaluate, p_eval_deriv, curl, product). Lean STANDARD Vina scoring (no QFD) to match
// CPU AutoDock Vina 1.2.7. N ligands = N independent rigid roots in one shared coordinate frame;
// inter-ligand coupling is automatic: inter-ligand forces land in minus_forces[atom] (via the
// other_pairs eval) and each ligand's POT routes its own atoms' forces to its own DOF.

// Forward kinematics for ONE ligand: writes lab coords for its atoms into coords[].
void set_one_multi(const float* pos, const float* ori, const float* tors,
				   rigid_multi_cl* rigid, m_coords_multi_cl* m_coords, const atom_cl* atoms, float epsilon_fl) {
	for (int i = 0; i < 3; i++) rigid->origin[0][i]        = pos[i];
	for (int i = 0; i < 4; i++) rigid->orientation_q[0][i] = ori[i];
	quaternion_to_r3(rigid->orientation_q[0], rigid->orientation_m[0]);
	int begin = rigid->atom_range[0][0];
	int end   = rigid->atom_range[0][1];
	for (int i = begin; i < end; i++)
		local_to_lab(m_coords->coords[i], rigid->origin[0], &atoms[i].coords[0], rigid->orientation_m[0]);

	for (int current = 1; current < rigid->num_children + 1; current++) {
		int parent = rigid->parent[current];
		float torsion = tors[current - 1];
		local_to_lab(rigid->origin[current], rigid->origin[parent], rigid->relative_origin[current], rigid->orientation_m[parent]);
		local_to_lab_direction(rigid->axis[current], rigid->relative_axis[current], rigid->orientation_m[parent]);
		float tmp[4];
		float parent_q[4] = { rigid->orientation_q[parent][0], rigid->orientation_q[parent][1],
							  rigid->orientation_q[parent][2], rigid->orientation_q[parent][3] };
		float current_axis[3] = { rigid->axis[current][0], rigid->axis[current][1], rigid->axis[current][2] };
		angle_to_quaternion2(tmp, current_axis, torsion);
		angle_to_quaternion_multi(tmp, parent_q);
		quaternion_normalize_approx(tmp, epsilon_fl);
		for (int i = 0; i < 4; i++) rigid->orientation_q[current][i] = tmp[i];
		quaternion_to_r3(rigid->orientation_q[current], rigid->orientation_m[current]);
		begin = rigid->atom_range[current][0];
		end   = rigid->atom_range[current][1];
		for (int i = begin; i < end; i++)
			local_to_lab(m_coords->coords[i], rigid->origin[current], &atoms[i].coords[0], rigid->orientation_m[current]);
	}
}

// Forward kinematics for ALL ligands.
void set_multi(const output_type_multi_cl* c, m_multi_cl_private* m, float epsilon_fl) {
	for (int k = 0; k < m->num_ligands; k++) {
		ligand_multi_cl* L = &m->ligands[k];
		set_one_multi(c->position[k], c->orientation[k], &c->lig_torsion[L->torsion_offset],
					  &L->rigid, &m->m_coords, m->atoms, epsilon_fl);
	}
}

// Raw-pointer interacting-pairs energy+forces (serves intra AND inter-ligand pairs).
float eval_pairs_multi(const __global pre_cl* pre, const float v, const int num_pairs,
					   const __global int* tpi, const __global int* pa, const __global int* pb,
					   m_coords_multi_cl* m_coords, m_minus_forces_multi* minus_forces, const float epsilon_fl) {
	float e = 0;
	for (int i = 0; i < num_pairs; i++) {
		int a = pa[i], b = pb[i];
		float r[3] = { m_coords->coords[b][0] - m_coords->coords[a][0],
		               m_coords->coords[b][1] - m_coords->coords[a][1],
		               m_coords->coords[b][2] - m_coords->coords[a][2] };
		float r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
		if (r2 < pre->m_cutoff_sqr) {
			float tmp[2];
			p_eval_deriv(tmp, tpi[i], r2, pre, epsilon_fl);
			float force[3] = { r[0]*tmp[1], r[1]*tmp[1], r[2]*tmp[1] };
			curl(&tmp[0], force, v, epsilon_fl);
			e += tmp[0];
			for (int j = 0; j < 3; j++) minus_forces->coords[a][j] -= force[j];
			for (int j = 0; j < 3; j++) minus_forces->coords[b][j] += force[j];
		}
	}
	return e;
}

// Standard Vina atom-type affinity grid term over ALL movable atoms. Initialises minus_forces[i]
// for every movable atom (assignment), so the subsequent pair evals can accumulate (+=/-=).
float ig_eval_multi(const __global grids_cl* grids, m_coords_multi_cl* m_coords, m_minus_forces_multi* minus_forces,
					const atom_cl* atoms, const int num_movable, const float v, const float epsilon_fl) {
	float e = 0;
	int nat = num_atom_types(grids->atu);
	for (int i = 0; i < num_movable; i++) {
		int t = atoms[i].types[grids->atu];
		if (t >= nat) { for (int j = 0; j < 3; j++) minus_forces->coords[i][j] = 0; continue; }
		float deriv[3];
		e += g_evaluate(&grids->grids[t], m_coords->coords[i], grids->slope, v, deriv, epsilon_fl);
		for (int j = 0; j < 3; j++) minus_forces->coords[i][j] = deriv[j];
	}
	return e;
}

// Per-ligand force/torque -> DOF gradient. Writes raw output slots g_pos[3], g_ori[3], g_tor[ntors].
// Inter-ligand forces are already summed into minus_forces[atom], so gathering this ligand's atoms
// captures the coupling to all other ligands automatically.
void POT_deriv_one_multi(m_minus_forces_multi* minus_forces, const rigid_multi_cl* rigid, m_coords_multi_cl* m_coords,
						 float* g_pos, float* g_ori, float* g_tor) {
	int num_torsion = rigid->num_children;
	int num_rigid = num_torsion + 1;
	float pos_d_tmp[MAX_NUM_OF_RIGID_MULTI][3], pos_d[MAX_NUM_OF_RIGID_MULTI][3];
	float ori_d_tmp[MAX_NUM_OF_RIGID_MULTI][3], ori_d[MAX_NUM_OF_RIGID_MULTI][3];
	float tor_d[MAX_NUM_OF_RIGID_MULTI];

	for (int i = 0; i < num_rigid; i++) {
		int begin = rigid->atom_range[i][0];
		int end   = rigid->atom_range[i][1];
		for (int k = 0; k < 3; k++) { pos_d_tmp[i][k] = 0; ori_d_tmp[i][k] = 0; }
		for (int j = begin; j < end; j++) {
			for (int k = 0; k < 3; k++) pos_d_tmp[i][k] += minus_forces->coords[j][k];
			float tmp1[3] = { m_coords->coords[j][0] - rigid->origin[i][0], m_coords->coords[j][1] - rigid->origin[i][1], m_coords->coords[j][2] - rigid->origin[i][2] };
			float tmp2[3] = { minus_forces->coords[j][0], minus_forces->coords[j][1], minus_forces->coords[j][2] };
			float tmp3[3]; product(tmp3, tmp1, tmp2);
			for (int k = 0; k < 3; k++) ori_d_tmp[i][k] += tmp3[k];
		}
	}
	for (int i = num_rigid - 1; i >= 0; i--) {
		for (int k = 0; k < 3; k++) pos_d[i][k] = pos_d_tmp[i][k];
		for (int j = 1; j < num_rigid; j++)
			if (rigid->parent[j] == i)
				for (int k = 0; k < 3; k++) pos_d[i][k] += pos_d[j][k];
	}
	for (int i = num_rigid - 1; i >= 0; i--) {
		for (int k = 0; k < 3; k++) ori_d[i][k] = ori_d_tmp[i][k];
		for (int j = 1; j < num_rigid; j++)
			if (rigid->parent[j] == i) {
				for (int k = 0; k < 3; k++) ori_d[i][k] += ori_d[j][k];
				float po[3];
				float ot[3] = { rigid->origin[j][0] - rigid->origin[i][0], rigid->origin[j][1] - rigid->origin[i][1], rigid->origin[j][2] - rigid->origin[i][2] };
				product(po, ot, pos_d[j]);
				for (int k = 0; k < 3; k++) ori_d[i][k] += po[k];
			}
	}
	for (int i = num_rigid - 1; i >= 0; i--) {
		float s = 0;
		for (int j = 0; j < 3; j++) s += ori_d[i][j] * rigid->axis[i][j];
		tor_d[i] = s;
	}
	for (int k = 0; k < 3; k++) g_pos[k] = pos_d[0][k];
	for (int k = 0; k < 3; k++) g_ori[k] = ori_d[0][k];
	for (int k = 0; k < num_torsion; k++) g_tor[k] = tor_d[k + 1];
}

// Energy + gradient for a multi-ligand conf. lig_pairs_g points to mg->lig_pairs[0..N-1].
float m_eval_deriv_multi(output_type_multi_cl* c, change_multi_cl* g, m_multi_cl_private* m,
						 const __global lig_pairs_multi_cl* lig_pairs_g,
						 const __global other_pairs_cl* other_g,
						 const __global pre_cl* pre, const __global grids_cl* grids,
						 const __global float* v, const float epsilon_fl) {
	set_multi(c, m, epsilon_fl);

	float e = ig_eval_multi(grids, &m->m_coords, &m->minus_forces,
							m->atoms, m->m_num_movable_atoms, v[1], epsilon_fl);

	for (int k = 0; k < m->num_ligands; k++)
		e += eval_pairs_multi(pre, v[0], lig_pairs_g[k].num_pairs,
							  lig_pairs_g[k].type_pair_index, lig_pairs_g[k].a, lig_pairs_g[k].b,
							  &m->m_coords, &m->minus_forces, epsilon_fl);

	e += eval_pairs_multi(pre, v[0], other_g->num_pairs,
						  other_g->type_pair_index, other_g->a, other_g->b,
						  &m->m_coords, &m->minus_forces, epsilon_fl);

	for (int k = 0; k < m->num_ligands; k++) {
		ligand_multi_cl* L = &m->ligands[k];
		POT_deriv_one_multi(&m->minus_forces, &L->rigid, &m->m_coords,
							g->position[k], g->orientation[k], &g->lig_torsion[L->torsion_offset]);
	}
	return e;
}

// ── Multi conf-vector ops over the flattened DOF: [ per-ligand 6 (pos+ori) ] then [ torsions ] ──
inline float find_change_index_read_multi(const change_multi_cl* g, int index, int num_ligands) {
	int rigid_dof = 6 * num_ligands;
	if (index < rigid_dof) {
		int k = index / 6, c = index % 6;
		return (c < 3) ? g->position[k][c] : g->orientation[k][c - 3];
	}
	return g->lig_torsion[index - rigid_dof];
}
inline void find_change_index_write_multi(change_multi_cl* g, int index, float data, int num_ligands) {
	int rigid_dof = 6 * num_ligands;
	if (index < rigid_dof) {
		int k = index / 6, c = index % 6;
		if (c < 3) g->position[k][c] = data; else g->orientation[k][c - 3] = data;
		return;
	}
	g->lig_torsion[index - rigid_dof] = data;
}
inline float scalar_product_multi(const change_multi_cl* a, const change_multi_cl* b, int n, int num_ligands) {
	float t = 0;
	for (int i = 0; i < n; i++) t += find_change_index_read_multi(a, i, num_ligands) * find_change_index_read_multi(b, i, num_ligands);
	return t;
}
inline float to_norm_multi(const change_multi_cl* in, int n, int num_ligands) {
	float d = 0;
	for (int i = 0; i < n; i++) { float x = find_change_index_read_multi(in, i, num_ligands); d += x * x; }
	return sqrt(d);
}

// x += factor * c   (per-ligand position + quaternion orientation; shared torsion array)
void output_type_multi_cl_increment(output_type_multi_cl* x, const change_multi_cl* c,
									float factor, float epsilon_fl, int total_torsions) {
	int N = x->num_ligands;
	for (int k = 0; k < N; k++) {
		for (int i = 0; i < 3; i++) x->position[k][i] += factor * c->position[k][i];
		float rotation[3];
		for (int i = 0; i < 3; i++) rotation[i] = factor * c->orientation[k][i];
		quaternion_increment(x->orientation[k], rotation, epsilon_fl);
	}
	for (int t = 0; t < total_torsions; t++) {
		float tmp = factor * c->lig_torsion[t];
		normalize_angle(&tmp);
		x->lig_torsion[t] += tmp;
		normalize_angle(&(x->lig_torsion[t]));
	}
}

// Mutate ONE randomly-chosen DOF across all ligands: each ligand offers 1 translate + 1 rotate +
// its torsions; pick one uniformly. Mirrors single mutate_conf_cl but ligand-aware.
void mutate_conf_multi(const int step, output_type_multi_cl* c, m_multi_cl_private* m,
					   __global const int* random_int_map,
					   __global const float random_inside_sphere_map[][3],
					   __global const float* random_fl_pi_map,
					   const float epsilon_fl, const float amplitude) {
	int index = step;
	int total_choices = 0;
	for (int k = 0; k < m->num_ligands; k++) total_choices += 2 + m->ligands[k].num_torsions;
	if (total_choices <= 0) return;
	int which = random_int_map[index] % total_choices;

	for (int k = 0; k < m->num_ligands; k++) {
		int per = 2 + m->ligands[k].num_torsions;
		if (which < per) {
			if (which == 0) {                       // translate ligand k
				for (int i = 0; i < 3; i++) c->position[k][i] += amplitude * random_inside_sphere_map[index][i];
				return;
			}
			which--;
			if (which == 0) {                       // rotate ligand k about its centre
				int begin = m->ligands[k].begin, end = m->ligands[k].end;
				float origin[3] = { m->ligands[k].rigid.origin[0][0], m->ligands[k].rigid.origin[0][1], m->ligands[k].rigid.origin[0][2] };
				float acc = 0; int cnt = 0;
				for (int i = begin; i < end; i++) {
					if (m->atoms[i].types[0] != EL_TYPE_H) {
						float dx = m->m_coords.coords[i][0] - origin[0];
						float dy = m->m_coords.coords[i][1] - origin[1];
						float dz = m->m_coords.coords[i][2] - origin[2];
						acc += dx*dx + dy*dy + dz*dz; cnt++;
					}
				}
				float gr = (cnt > 0) ? sqrt(acc / cnt) : 0;
				if (gr > epsilon_fl) {
					float rotation[3];
					for (int i = 0; i < 3; i++) rotation[i] = amplitude / gr * random_inside_sphere_map[index][i];
					quaternion_increment(c->orientation[k], rotation, epsilon_fl);
				}
				return;
			}
			which--;                                // torsion `which` of ligand k
			c->lig_torsion[m->ligands[k].torsion_offset + which] = random_fl_pi_map[index];
			return;
		}
		which -= per;
	}
}

// ── Multi dense BFGS (mirrors single bfgs(); operates on the flattened multi-root DOF) ──
void minus_mat_vec_product_multi(const matrix_multi* h, const change_multi_cl* in,
								 change_multi_cl* out, const int num_ligands) {
	int n = h->dim;
	for (int i = 0; i < n; i++) {
		float sum = 0;
		for (int j = 0; j < n; j++)
			sum += h->data[index_permissive_multi(h, i, j)] * find_change_index_read_multi(in, j, num_ligands);
		find_change_index_write_multi(out, i, -sum, num_ligands);
	}
}

float line_search_multi(m_multi_cl_private* m,
						const __global lig_pairs_multi_cl* lig_pairs_g,
						const __global other_pairs_cl* other_g,
						const __global pre_cl* pre, const __global grids_cl* grids,
						int n, const output_type_multi_cl* x, const change_multi_cl* g, float f0,
						const change_multi_cl* p, output_type_multi_cl* x_new, change_multi_cl* g_new, float* f1,
						const float epsilon_fl, const __global float* hunt_cap,
						const int num_ligands, const int total_torsions) {
	const float c0 = 0.0001;
	const int max_trials = 10;
	const float multiplier = 0.5;
	float alpha = 1;
	const float pg = scalar_product_multi(p, g, n, num_ligands);
	for (int trial = 0; trial < max_trials; trial++) {
		*x_new = *x;
		output_type_multi_cl_increment(x_new, p, alpha, epsilon_fl, total_torsions);
		*f1 = m_eval_deriv_multi(x_new, g_new, m, lig_pairs_g, other_g, pre, grids, hunt_cap, epsilon_fl);
		if (*f1 - f0 < c0 * alpha * pg) break;
		alpha *= multiplier;
	}
	return alpha;
}

bool bfgs_update_multi(matrix_multi* h, const change_multi_cl* p, const change_multi_cl* y,
					   const float alpha, const __global mis_cl* mis, const int num_ligands) {
	const float yp = scalar_product_multi(y, p, h->dim, num_ligands);
	if (alpha * yp < mis->epsilon_fl) return false;
	change_multi_cl minus_hy = *y;
	minus_mat_vec_product_multi(h, y, &minus_hy, num_ligands);
	const float yhy = -scalar_product_multi(y, &minus_hy, h->dim, num_ligands);
	const float r = 1 / (alpha * yp);
	const int n = h->dim;
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			float tmp = alpha * r * (find_change_index_read_multi(&minus_hy, i, num_ligands) * find_change_index_read_multi(p, j, num_ligands)
								   + find_change_index_read_multi(&minus_hy, j, num_ligands) * find_change_index_read_multi(p, i, num_ligands))
					  + alpha * alpha * (r * r * yhy + r) * find_change_index_read_multi(p, i, num_ligands) * find_change_index_read_multi(p, j, num_ligands);
			h->data[i + j * (j + 1) / 2] += tmp;
		}
	}
	return true;
}

void bfgs_multi(output_type_multi_cl* x, change_multi_cl* g, m_multi_cl_private* m,
				const __global lig_pairs_multi_cl* lig_pairs_g, const __global other_pairs_cl* other_g,
				const __global pre_cl* pre, const __global grids_cl* grids, const __global mis_cl* mis,
				const int num_ligands, const int total_torsions, const int max_bfgs_steps) {
	int n = 6 * num_ligands + total_torsions;

	matrix_multi h;
	matrix_multi_init(&h, n, 0);
	matrix_multi_set_diagonal(&h, 1);

	change_multi_cl g_new = *g;
	output_type_multi_cl x_new = *x;
	float f0 = m_eval_deriv_multi(x, g, m, lig_pairs_g, other_g, pre, grids, mis->hunt_cap, mis->epsilon_fl);
	float f_orig = f0;
	change_multi_cl g_orig = *g;
	output_type_multi_cl x_orig = *x;
	change_multi_cl p = *g;

	for (int step = 0; step < max_bfgs_steps; step++) {
		minus_mat_vec_product_multi(&h, g, &p, num_ligands);
		float f1 = 0;
		const float alpha = line_search_multi(m, lig_pairs_g, other_g, pre, grids, n, x, g, f0, &p,
											  &x_new, &g_new, &f1, mis->epsilon_fl, mis->hunt_cap,
											  num_ligands, total_torsions);
		change_multi_cl y = g_new;
		for (int i = 0; i < n; i++) {
			float tmp = find_change_index_read_multi(&y, i, num_ligands) - find_change_index_read_multi(g, i, num_ligands);
			find_change_index_write_multi(&y, i, tmp, num_ligands);
		}
		f0 = f1;
		*x = x_new;
		if (!(sqrt(scalar_product_multi(g, g, n, num_ligands)) >= 1e-5)) break;
		*g = g_new;
		if (step == 0) {
			float yy = scalar_product_multi(&y, &y, n, num_ligands);
			if (fabs(yy) > mis->epsilon_fl)
				matrix_multi_set_diagonal(&h, alpha * scalar_product_multi(&y, &p, n, num_ligands) / yy);
		}
		bfgs_update_multi(&h, &p, &y, alpha, mis, num_ligands);
	}

	if (!(f0 <= f_orig)) { f0 = f_orig; *x = x_orig; *g = g_orig; }
	x->e = f0;
}
// ════════════ Multi L-BFGS (rilc_bfgs_multi) — limited-memory, NO dense Hessian ════════════
// Mirrors the single rilc_bfgs but over the flattened multi-root DOF. Private cost is just two
// history vectors (lm_s, lm_y, MAX_MULTI_CONF_DIM each) instead of the O(n^2) Hessian — this is
// what lets N scale to 32 in private (and, with the global m path, toward hundreds).
void get_to_minus_multi(change_multi_cl* a, const change_multi_cl* b, int n, int num_ligands) {
	for (int i = 0; i < n; i++)
		find_change_index_write_multi(a, i, -find_change_index_read_multi(b, i, num_ligands), num_ligands);
}

int line_search_lewisoverton_multi(
		m_multi_cl_private* m, const __global lig_pairs_multi_cl* lig_pairs_g, const __global other_pairs_cl* other_g,
		const __global pre_cl* pre, const __global grids_cl* grids,
		int n, float* stp, output_type_multi_cl* x, float* f, change_multi_cl* g,
		const change_multi_cl* d, const output_type_multi_cl* xp, const change_multi_cl* gp,
		const float epsilon_fl, const __global float* hunt_cap,
		const int num_ligands, const int total_torsions)
{
	int count = 0;
	bool brackt = false;
	float finit, dginit, dgtest, dstest;
	float mu = 0.0, nu = 1.0e+20;
	dginit = scalar_product_multi(gp, d, n, num_ligands);
	if (0.0 < dginit) return -1;
	finit = *f;
	dgtest = 1.0e-4 * dginit;
	dstest = 0.1 * dginit;
	while (true) {
		*x = *xp;
		output_type_multi_cl_increment(x, d, *stp, epsilon_fl, total_torsions);
		*f = m_eval_deriv_multi(x, g, m, lig_pairs_g, other_g, pre, grids, hunt_cap, epsilon_fl);
		++count;
		if (*f > finit + *stp * dgtest) { nu = *stp; brackt = true; }
		else {
			if (scalar_product_multi(g, d, n, num_ligands) < dstest) mu = *stp;
			else return count;
		}
		if (10 <= count) { if (*f > finit) return -1; else return count; }
		if (brackt) (*stp) = 0.5 * (mu + nu);
		else (*stp) *= 2.0;
	}
}

void rilc_bfgs_multi(
		output_type_multi_cl* x, change_multi_cl* g, m_multi_cl_private* m,
		const __global lig_pairs_multi_cl* lig_pairs_g, const __global other_pairs_cl* other_g,
		const __global pre_cl* pre, const __global grids_cl* grids, const __global mis_cl* mis,
		const int num_ligands, const int total_torsions, const int max_steps)
{
	int n = 6 * num_ligands + total_torsions;
	int k, ls;
	float step, fx, ys, yy;
	float beta = 0, cau, cau_t, lm_s_dot_d, lm_alpha = 0, lm_ys = 0;
	output_type_multi_cl xp = *x;
	change_multi_cl gp = *g;
	float lm_s[MAX_MULTI_CONF_DIM] = { 0 };
	float lm_y[MAX_MULTI_CONF_DIM] = { 0 };
	fx = m_eval_deriv_multi(x, g, m, lig_pairs_g, other_g, pre, grids, mis->hunt_cap, mis->epsilon_fl);
	float fxp = fx, fx_orig = fx;
	change_multi_cl g_orig = *g;
	output_type_multi_cl x_orig = *x;
	change_multi_cl d = *g;
	get_to_minus_multi(&d, g, n, num_ligands);
	if (!(sqrt(scalar_product_multi(g, g, n, num_ligands)) >= 1e-5)) { x->e = fx; return; }
	step = 1.0 / to_norm_multi(&d, n, num_ligands);
	k = 1;
	while (true) {
		xp = *x; gp = *g; fxp = fx;
		ls = line_search_lewisoverton_multi(m, lig_pairs_g, other_g, pre, grids, n, &step, x, &fx, g, &d, &xp, &gp,
		                                    mis->epsilon_fl, mis->hunt_cap, num_ligands, total_torsions);
		if (ls < 0) { *x = xp; *g = gp; fx = fxp; break; }
		if (!(sqrt(scalar_product_multi(g, g, n, num_ligands)) >= 1e-5)) { x->e = fx; return; }
		if (max_steps != 0 && max_steps <= k) break;
		++k;
		for (int i = 0; i < n; i++) lm_s[i] = step * find_change_index_read_multi(&d, i, num_ligands);
		for (int i = 0; i < n; i++) lm_y[i] = find_change_index_read_multi(g, i, num_ligands) - find_change_index_read_multi(&gp, i, num_ligands);
		ys = 0; for (int i = 0; i < n; i++) ys += lm_y[i] * lm_s[i];
		yy = 0; for (int i = 0; i < n; i++) yy += lm_y[i] * lm_y[i];
		lm_ys = ys;
		get_to_minus_multi(&d, g, n, num_ligands);
		cau = 0; for (int i = 0; i < n; i++) cau += lm_s[i] * lm_s[i];
		cau_t = 0; for (int i = 0; i < n; i++) cau_t += find_change_index_read_multi(&gp, i, num_ligands) * find_change_index_read_multi(&gp, i, num_ligands);
		cau = cau * sqrt(cau_t) * 1.0e-6;
		if (ys > cau) {
			lm_s_dot_d = 0; for (int a = 0; a < n; a++) lm_s_dot_d += lm_s[a] * find_change_index_read_multi(&d, a, num_ligands);
			lm_alpha = lm_s_dot_d / lm_ys;
			for (int b = 0; b < n; b++) find_change_index_write_multi(&d, b, find_change_index_read_multi(&d, b, num_ligands) + lm_alpha * lm_y[b], num_ligands);
			for (int i = 0; i < n; i++) find_change_index_write_multi(&d, i, find_change_index_read_multi(&d, i, num_ligands) * (ys / yy), num_ligands);
			beta = 0; for (int a = 0; a < n; a++) beta += lm_y[a] * find_change_index_read_multi(&d, a, num_ligands);
			beta /= lm_ys;
			for (int i = 0; i < n; i++) find_change_index_write_multi(&d, i, find_change_index_read_multi(&d, i, num_ligands) + (lm_alpha - beta) * lm_s[i], num_ligands);
		}
		step = 1.0;
	}
	if (!(fx <= fx_orig)) { fx = fx_orig; *x = x_orig; *g = g_orig; }
	x->e = fx;
}
