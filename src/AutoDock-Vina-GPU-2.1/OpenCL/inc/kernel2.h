#pragma once
// Macros below are shared in both device and host
#define TOLERANCE 1e-16
// kernel1 macros
#define MAX_NUM_OF_EVERY_M_DATA_ELEMENT 512
#define MAX_M_DATA_MI 16
#define MAX_M_DATA_MJ 16
#define MAX_M_DATA_MK 16
#define MAX_NUM_OF_TOTAL_M_DATA MAX_M_DATA_MI*MAX_M_DATA_MJ*MAX_M_DATA_MK*MAX_NUM_OF_EVERY_M_DATA_ELEMENT

//kernel2 macros
#define MAX_NUM_OF_LIG_TORSION 100
#define MAX_NUM_OF_FLEX_TORSION 1
#define MAX_NUM_OF_RIGID 104  // must be >= MAX_NUM_OF_LIG_TORSION + 1 (torsions + root)
#define MAX_NUM_OF_ATOMS 272
#define SIZE_OF_MOLEC_STRUC ((3+4+MAX_NUM_OF_LIG_TORSION+MAX_NUM_OF_FLEX_TORSION+ 1)*sizeof(float) )
#define SIZE_OF_CHANGE_STRUC ((3+3+MAX_NUM_OF_LIG_TORSION+MAX_NUM_OF_FLEX_TORSION + 1)*sizeof(float))
#define MAX_HESSIAN_MATRIX_SIZE ((6 +  MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION)*(6 +  MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) / 2)
#define MAX_NUM_OF_LIG_PAIRS 65536  // C(272,2)=36856 max theoretical; 65536 guarantees full coverage
#define MAX_NUM_OF_BFGS_STEPS 64
#define MAX_NUM_OF_RANDOM_MAP 20000 // not too large (stack overflow!)
#define GRIDS_SIZE 21

// QFD (Quantum Field Docking) grid indices — slots 17-20 appended after standard atom-type grids
#define GRID_IDX_ESP      17   // receptor electrostatic potential  [kcal/(mol·e)]
#define GRID_IDX_DESOLV   18   // desolvation susceptibility grid   [kcal/mol per unit q²]
#define GRID_IDX_INFOMAP  19   // information resonance field       [dimensionless coupling]
#define GRID_IDX_RESERVED 20   // reserved for future QFD terms

// QFD scoring weights — calibrated for real Gasteiger charges on ligands
// w_elec=0.05: at 2Å from Zn with q=-0.19 → -0.95 kcal/mol (gentle guidance)
// Per-atom soft cap QFD_ATOM_E_CAP=1.5: prevents Mg/strong-field over-attraction
//   without cap: 2 chelating O (q=-0.5) near Mg → 2×(-0.05×0.5×120) = -6 kcal/mol (too strong)
//   with cap:    capped at 2×1.5 = -3 kcal/mol max (safer)
// Tune further with tools/prep/calibrate_qfd_weights.py
#define QFD_ELEC_WEIGHT   0.05f
#define QFD_DESOLV_WEIGHT 0.05f
#define QFD_INFO_WEIGHT   0.02f
// Per-atom ESP energy soft cap [kcal/mol]: E_out = E_raw/(1+|E_raw|/cap)
// Smoothly prevents any single atom from dominating ranking; gradient-consistent.
#define QFD_ATOM_E_CAP    1.5f

// QFD Phase 2: Temperature Annealing ladder
// Each trajectory anneals from T_start → QFD_T_FINAL over search_depth steps.
// T_start is assigned from a geometric ladder: traj_id % QFD_N_REPLICAS → index.
// Max temp reduced 8→3 to keep search directed (not purely random at start).
#define QFD_N_REPLICAS    8
#define QFD_T_START_MIN   0.30f   // coldest replica initial temperature
#define QFD_T_START_MAX   3.00f   // hottest replica (was 8; reduced to keep search directed)
#define QFD_T_FINAL       0.15f   // all replicas cool to this at end of search
#define MAX_NUM_OF_PROTEIN_ATOMS 50000

#ifdef LARGE_BOX
	// docking box size <= 70x70x70
	#define MAX_NUM_OF_GRID_MI 300
	#define MAX_NUM_OF_GRID_MJ 300
	#define MAX_NUM_OF_GRID_MK 300
	#define MAX_NUM_OF_GRID_DIM 300      // per-dimension guard: matches array dims for large box
	#define MAX_NUM_OF_AR_CELLS 16384    // max szv_grid cells: ceil(70/0.375)=187 → int(187*0.375/3)=23 per dim → 23^3=12167
	#define MAX_NUM_OF_ATOM_RELATION_COUNT 1024  // max protein atoms per szv_grid cell
#endif

#ifdef SMALL_BOX
	// docking box size <= 72x72x72 (192 grid points × 0.375 Å)
	// m_data uses actual mi*mj*mk as stride product; 128³=2M voxels covers all PDBbind targets
	// MAX_NUM_OF_GRID_MI/MJ/MK size the m_data array; MAX_NUM_OF_GRID_DIM guards per-dimension
	#define MAX_NUM_OF_GRID_MI 128       // array-size constant — product 128³ fits all PDBbind boxes
	#define MAX_NUM_OF_GRID_MJ 128
	#define MAX_NUM_OF_GRID_MK 128
	#define MAX_NUM_OF_GRID_DIM 256      // per-dimension guard: covers elongated boxes up to ~95 Å
	#define MAX_NUM_OF_AR_CELLS 4096     // max szv_grid cells: worst-case ~2535 across PDBbind
	#define MAX_NUM_OF_ATOM_RELATION_COUNT 1024  // max protein atoms per szv_grid cell
#endif
//#define GRID_MI 65//55
//#define GRID_MJ 71//55
//#define GRID_MK 61//81
#define MAX_P_DATA_M_DATA_SIZE 600
//#define MAX_NUM_OF_GRID_ATOMS 130
#define FAST_SIZE 2051
#define SMOOTH_SIZE 2051
#define MAX_CONTAINER_SIZE_EVERY_WI 5
//#define EL_TYPE_H_CL 0
#define EL_TYPE_H 0

#define EL_TYPE_SIZE 11
#define AD_TYPE_SIZE 32
#define XS_TYPE_SIZE 17
#define SY_TYPE_SIZE 18


typedef struct {
	float data[GRIDS_SIZE];
} affinities_cl;

typedef struct {
	int   types[4]; // el  ad  xs  sy
	float coords[3];
	float charge;   // partial charge [e]; used by QFD electrostatic + desolvation terms
} atom_cl;
				   
typedef struct {
	atom_cl atoms[MAX_NUM_OF_PROTEIN_ATOMS];
} pa_cl;

typedef struct {
	float coords[MAX_NUM_OF_ATOMS][3];
} m_coords_cl;

typedef struct {
	float coords[MAX_NUM_OF_ATOMS][3];
} ligand_atom_coords_cl;

typedef struct {
	float coords[MAX_NUM_OF_ATOMS][3];
} m_minus_forces;

typedef struct  { // namely molec_struc
	float e;
	float position		[3];
	float orientation	[4];
	float lig_torsion	[MAX_NUM_OF_LIG_TORSION];
	float flex_torsion	[MAX_NUM_OF_FLEX_TORSION];
	//float coords		[MAX_NUM_OF_ATOMS][3];
	//float lig_torsion_size;
} output_type_cl;

typedef struct  { // namely change_struc
	//float lig_torsion_size;
	float position		[3];
	float orientation	[3];
	float lig_torsion	[MAX_NUM_OF_LIG_TORSION];
	float flex_torsion	[MAX_NUM_OF_FLEX_TORSION];
} change_cl;


typedef struct { // depth-first order
	int		num_children;
	bool	children_map	[MAX_NUM_OF_RIGID][MAX_NUM_OF_RIGID]; // chidren_map[i][j] = true if node i's child is node j
	int		parent			[MAX_NUM_OF_RIGID]; // every node has only 1 parent node
	
	int		atom_range		[MAX_NUM_OF_RIGID][2];
	float	origin			[MAX_NUM_OF_RIGID][3];
	float	orientation_m	[MAX_NUM_OF_RIGID][9]; // This matrix is fixed to 3*3
	float	orientation_q	[MAX_NUM_OF_RIGID][4];
	
	float	axis			[MAX_NUM_OF_RIGID][3]; // 1st column is root node, all 0s
	float	relative_axis	[MAX_NUM_OF_RIGID][3]; // 1st column is root node, all 0s
	float	relative_origin	[MAX_NUM_OF_RIGID][3]; // 1st column is root node, all 0s
	
	
} rigid_cl;

typedef struct {
	int num_pairs;
	int type_pair_index	[MAX_NUM_OF_LIG_PAIRS];
	int a				[MAX_NUM_OF_LIG_PAIRS];
	int b				[MAX_NUM_OF_LIG_PAIRS];
} lig_pairs_cl;

typedef struct {
	int begin;
	int end;
	lig_pairs_cl pairs;
	rigid_cl rigid;
} ligand_cl;

// Pair-free ligand for the in-kernel private copy.
// Pairs are accessed via __global lig_pairs_cl* from the mg[] global buffer.
typedef struct {
	int begin;
	int end;
	rigid_cl rigid;
} ligand_private_cl;

typedef struct {
	int		int_map		[MAX_NUM_OF_RANDOM_MAP];
	float	pi_map		[MAX_NUM_OF_RANDOM_MAP];
	float	sphere_map	[MAX_NUM_OF_RANDOM_MAP][3];
} random_maps;

typedef struct {
	int m_num_movable_atoms;
	atom_cl atoms[MAX_NUM_OF_ATOMS];
	m_coords_cl m_coords;
	m_minus_forces minus_forces;
	ligand_cl ligand;
} m_cl;

// In-kernel private copy of m_cl — excludes lig_pairs_cl to fit GPU private memory.
// With MAX_NUM_OF_LIG_PAIRS=16384 the full m_cl is ~224 KB; m_cl_private is ~33 KB.
// Pairs are accessed directly from __global mg[lig_id].ligand.pairs.
typedef struct {
	int m_num_movable_atoms;
	atom_cl atoms[MAX_NUM_OF_ATOMS];
	m_coords_cl m_coords;
	m_minus_forces minus_forces;
	ligand_private_cl ligand;
} m_cl_private;

typedef struct {
	int m_i;
	int m_j;
	int m_k;
	float m_init[3];
	float m_range[3];
	float m_factor[3];
	float m_dim_fl_minus_1[3];
	float m_factor_inv[3];
	float m_data [(MAX_NUM_OF_GRID_MI) * (MAX_NUM_OF_GRID_MJ) * (MAX_NUM_OF_GRID_MK) * 8];
} grid_cl;

typedef struct {
	int atu;
	float slope;
	grid_cl grids[GRIDS_SIZE];
} grids_cl;

typedef struct {
	float factor;
	float fast[FAST_SIZE];
	float smooth[SMOOTH_SIZE][2];
} p_m_data_cl;

typedef struct {
	int n;
	float m_cutoff_sqr;
	float factor;
	p_m_data_cl m_data[MAX_P_DATA_M_DATA_SIZE];
} pre_cl;

typedef struct {
	int dims[3];
	float init[3];
	float range[3];
} gb_cl;

typedef struct {
	int relation[MAX_NUM_OF_AR_CELLS][MAX_NUM_OF_ATOM_RELATION_COUNT];  // [cell_idx][atom_idx]
	int relation_size[MAX_NUM_OF_AR_CELLS];
} ar_cl;

typedef struct {
	int needed_size;
	//int torsion_size;
	//int search_depth;
	//int max_bfgs_steps;
	int total_wi;
	int thread;
	int ar_mi;
	int ar_mj;
	int ar_mk;
	int grids_front;
	int use_ad4zn;   // 1 = use AutoDock4Zn Zn coordination parameters; 0 = default




	float epsilon_fl;
	float cutoff_sqr;
	float max_fl;
	float mutation_amplitude;
	float hunt_cap[3];
	float authentic_v[3];
} mis_cl;





typedef struct  {
	int max_steps;
	float average_required_improvement;
	int over;
	int ig_grids_m_data_step;
	int	p_data_m_data_step;
	int	atu;
	int m_num_movable_atoms;
	float slope;
	float epsilon_fl;
	float epsilon_fl2;
	float epsilon_fl3;
	float epsilon_fl4;
	float epsilon_fl5;
}variables_bfgs;

typedef struct {
	output_type_cl container[MAX_CONTAINER_SIZE_EVERY_WI];
	int current_size;
}out_container;

