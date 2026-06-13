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
#define MAX_NUM_OF_FLEX_TORSION 32   // flexible receptor side-chain torsions (raised 12→32 for larger flex sets)
#define MAX_NUM_OF_RIGID 104  // must be >= MAX_NUM_OF_LIG_TORSION + 1 (torsions + root)
// Flex forward-kinematics forest: one anchor node per flexible residue + one node per flex
// torsion. Generous cap so a few residues (each with several rotatable bonds) fit.
#define MAX_NUM_OF_FLEX_RIGID 48  // anchors + torsion nodes across all flexible side chains (raised 24→48)
#define MAX_NUM_OF_ATOMS 272
#define SIZE_OF_MOLEC_STRUC ((3+4+MAX_NUM_OF_LIG_TORSION+MAX_NUM_OF_FLEX_TORSION+ 1)*sizeof(float) )
#define SIZE_OF_CHANGE_STRUC ((3+3+MAX_NUM_OF_LIG_TORSION+MAX_NUM_OF_FLEX_TORSION + 1)*sizeof(float))
#define MAX_HESSIAN_MATRIX_SIZE ((6 +  MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION)*(6 +  MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) / 2)
// Conformation-vector dimension n = position(3)+orientation(3)+lig_torsion+flex_torsion.
// L-BFGS history vectors (lm_s, lm_y) are indexed over n, NOT the packed-matrix size — sizing
// them as MAX_HESSIAN_MATRIX_SIZE wasted ~38 KB each of per-work-item private memory (doubled in
// the dual kernel). MAX_CONF_DIM is the correct size.
#define MAX_CONF_DIM (6 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 2)
#define MAX_NUM_OF_LIG_PAIRS 65536  // C(272,2)=36856 max theoretical; 65536 guarantees full coverage
#define MAX_NUM_OF_BFGS_STEPS 64
#define MAX_NUM_OF_RANDOM_MAP 20000 // not too large (stack overflow!)
#define GRIDS_SIZE 22

// QFD (Quantum Field Docking) grid indices — slots 17-21 appended after standard atom-type grids
#define GRID_IDX_ESP      17   // receptor electrostatic potential  [kcal/(mol·e)]
#define GRID_IDX_DESOLV   18   // desolvation susceptibility grid   [kcal/mol per unit q²]
#define GRID_IDX_INFOMAP  19   // information resonance field       [dimensionless coupling]
#define GRID_IDX_WATER    20   // explicit water displacement grid  [kcal/mol]
#define GRID_IDX_PIPI     21   // aromatic ring proximity grid      [dimensionless]

// QFD scoring weights — calibrated for real Gasteiger charges on ligands
#define QFD_ELEC_WEIGHT   0.05f
#define QFD_DESOLV_WEIGHT 0.05f
#define QFD_INFO_WEIGHT   0.02f
// Per-atom ESP energy soft cap [kcal/mol]: E_out = E_raw/(1+|E_raw|/cap)
#define QFD_ATOM_E_CAP    1.5f
// Phase 3: water displacement penalty weight.
#define QFD_WATER_WEIGHT  0.03f
// Phase 5: π-π aromatic stacking grid weight.
// Rewards aromatic ligand atoms (AD_TYPE_A) near receptor ring centres.
#define QFD_PIPI_WEIGHT   0.015f
// AD atom type index for aromatic carbon (consistent with atom_constants.h AD_TYPE_A=1)
#define AD_TYPE_A         1
// LS metal rescoring: tight Gaussian well for C++ post-docking reranking
// σ=0.20 Å selects only true N/O/S coordination bonds (r≈2.1Å); C atoms at ≥2.5Å ≈ 0
#define LS_METAL_R_OPT    2.10f  // optimal coordination distance [Å]
#define LS_METAL_SIGMA    0.20f  // Gaussian width [Å] — tight to exclude C
#define LS_METAL_CUTOFF   5.0f  // per-metal search radius [Å]

// QFD Phase 2: Temperature Annealing ladder
// Each trajectory anneals from T_start → QFD_T_FINAL over search_depth steps.
// T_start is assigned from a geometric ladder: traj_id % QFD_N_REPLICAS → index.
// Max temp reduced 8→3 to keep search directed (not purely random at start).
#define QFD_N_REPLICAS    8
#define QFD_T_START_MIN   0.30f   // coldest replica initial temperature
#define QFD_T_START_MAX   3.00f   // hottest replica (was 8; reduced to keep search directed)
#define QFD_T_FINAL       0.15f   // all replicas cool to this at end of search

// QFD Phase 2b: Basin-Hopping MC
// After each accepted MC step, BH_N_HOPS random perturbations are applied to
// escape local minima.  Each hop: large translation (BH_PERTURBATION_ANG Å) +
// one random torsion flip, then re-minimise, then Metropolis accept at current
// SA temperature.  Set BH_ENABLE 0 to compile out BH entirely.
#define BH_ENABLE            1
#define BH_N_HOPS            2
#define BH_PERTURBATION_ANG  1.5f
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
	// children_map removed (was bool[104][104] ~= 10.8 KB of per-work-item private memory):
	// node i's children are exactly { j>=1 : parent[j]==i }, derived on the fly in POT_deriv.
	// Shrinks rigid_cl -> m_cl_private; critical for the dual kernel (holds 2x m_cl_private).
	int		parent			[MAX_NUM_OF_RIGID]; // every node has only 1 parent node
	
	int		atom_range		[MAX_NUM_OF_RIGID][2];
	float	origin			[MAX_NUM_OF_RIGID][3];
	float	orientation_m	[MAX_NUM_OF_RIGID][9]; // This matrix is fixed to 3*3
	float	orientation_q	[MAX_NUM_OF_RIGID][4];
	
	float	axis			[MAX_NUM_OF_RIGID][3]; // 1st column is root node, all 0s
	float	relative_axis	[MAX_NUM_OF_RIGID][3]; // 1st column is root node, all 0s
	float	relative_origin	[MAX_NUM_OF_RIGID][3]; // 1st column is root node, all 0s
	
	
} rigid_cl;

// Flexible side-chain forward-kinematics forest (Stage 1).
// Compact analogue of rigid_cl: a flat list of nodes in depth-first order.
// Anchor nodes (one per flexible residue) have parent < 0 and a FIXED frame
// (origin/orientation loaded from input, never recomputed). Torsion nodes have a
// valid parent and rotate about `axis` by flex_torsion[torsion_idx].
// origin/orientation_m/orientation_q/axis are MUTABLE (recomputed each conf);
// parent/torsion_idx/atom_range/relative_axis/relative_origin are CONSTANT.
typedef struct {
	int   num_nodes;                               // total flex nodes (anchors + torsion nodes)
	int   parent        [MAX_NUM_OF_FLEX_RIGID];   // -1 for anchor root nodes
	int   torsion_idx   [MAX_NUM_OF_FLEX_RIGID];   // index into flex_torsion[]; -1 for anchors
	int   atom_range    [MAX_NUM_OF_FLEX_RIGID][2];// [begin,end) into the movable-atom array
	float origin        [MAX_NUM_OF_FLEX_RIGID][3];
	float orientation_m [MAX_NUM_OF_FLEX_RIGID][9];
	float orientation_q [MAX_NUM_OF_FLEX_RIGID][4];
	float axis          [MAX_NUM_OF_FLEX_RIGID][3];
	float relative_axis   [MAX_NUM_OF_FLEX_RIGID][3];
	float relative_origin [MAX_NUM_OF_FLEX_RIGID][3];
} flex_rigid_cl;

typedef struct {
	int num_pairs;
	int type_pair_index	[MAX_NUM_OF_LIG_PAIRS];
	int a				[MAX_NUM_OF_LIG_PAIRS];
	int b				[MAX_NUM_OF_LIG_PAIRS];
} lig_pairs_cl;

// Stage 2: ligand↔side-chain and side-chain↔side-chain interaction pairs (model.other_pairs).
// Same layout as lig_pairs_cl; atom indices a,b reference the full movable-atom array (ligand+flex).
// Lives only in the GLOBAL m_cl (accessed via __global ptr), never in the private copy.
#define MAX_NUM_OF_OTHER_PAIRS 16384
typedef struct {
	int num_pairs;
	int type_pair_index	[MAX_NUM_OF_OTHER_PAIRS];
	int a				[MAX_NUM_OF_OTHER_PAIRS];
	int b				[MAX_NUM_OF_OTHER_PAIRS];
} other_pairs_cl;

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
	flex_rigid_cl flex_rigid;   // Stage 1: flexible side-chain forest (empty when num_nodes==0)
	other_pairs_cl other_pairs; // Stage 2: ligand↔flex + flex↔flex pairs (num_pairs==0 when rigid)
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
	flex_rigid_cl flex_rigid;   // Stage 1: ~3 KB, small enough for the private copy
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
	int use_ad4zn;          // 1 = use AutoDock4Zn Zn coordination parameters; 0 = default
	int flex_torsion_size;  // Phase 4: number of flexible receptor torsions (0 = rigid receptor)




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

