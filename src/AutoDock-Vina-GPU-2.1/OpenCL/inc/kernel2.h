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
// ── Native multi-ligand caps — separate from single-ligand caps so the working single-ligand
//    m_cl is not bloated. Multi mode joins N ligands into one model (CPU Vina 1.2.x mechanism:
//    model.append()), each ligand an independent rigid root.
// The multi engine is CAP-INDEPENDENT and correct: with a fixed seed, N=8/12/16/32 give consistent
// docking (verified on DHFR, tests/03_dual — the earlier "cap-bug" was auto-seed stochasticity +
// mixing configs, not a real bug). The real limit is GPU memory: per-work-item private state
// × occupancy needs local-memory backing. Two optimizer regimes (selected in kernel2_multi by this
// cap): N<=16 uses dense BFGS (full Hessian, best quality; compact rigid_multi_cl tree keeps it
// ~206 KB/thread). N>16 uses L-BFGS (rilc_bfgs_multi: no dense Hessian → ~177 KB at N=32, runs).
// To co-dock up to 32 ligands, set this to 32 and rebuild (auto-switches to L-BFGS).
// Beyond 32 / toward hundreds: the GLOBAL-memory path (move m to global) + L-BFGS.
#define MAX_NUM_OF_LIGANDS          16    // default: dense BFGS, best quality; set 32 for L-BFGS scale
#define MAX_NUM_OF_MULTI_ATOMS      1024  // combined movable atoms across all ligands in a job
#define MAX_NUM_OF_LIG_PAIRS_MULTI  4096  // intra-ligand interacting pairs per ligand (multi mode)
#define MAX_NUM_OF_RIGID_MULTI      24    // max tree nodes per multi ligand — shrinks m_multi_cl_private
// Multi-ligand flattened DOF = 6*N (per-ligand root pos+ori) + total ligand torsions (shared array).
#define MAX_MULTI_CONF_DIM          (6 * MAX_NUM_OF_LIGANDS + MAX_NUM_OF_LIG_TORSION + 2)
#define MAX_MULTI_HESSIAN_MATRIX_SIZE ((6 * MAX_NUM_OF_LIGANDS + MAX_NUM_OF_LIG_TORSION) * (6 * MAX_NUM_OF_LIGANDS + MAX_NUM_OF_LIG_TORSION + 1) / 2)
#define SIZE_OF_MOLEC_STRUC ((3+4+MAX_NUM_OF_LIG_TORSION+MAX_NUM_OF_FLEX_TORSION+ 1)*sizeof(float) )
#define SIZE_OF_CHANGE_STRUC ((3+3+MAX_NUM_OF_LIG_TORSION+MAX_NUM_OF_FLEX_TORSION + 1)*sizeof(float))
#define MAX_HESSIAN_MATRIX_SIZE ((6 +  MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION)*(6 +  MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 1) / 2)
// Conformation-vector dimension n = position(3)+orientation(3)+lig_torsion+flex_torsion.
// L-BFGS history vectors (lm_s, lm_y) are indexed over n, NOT the packed-matrix size — sizing
// them as MAX_HESSIAN_MATRIX_SIZE wasted ~38 KB each of per-work-item private memory (doubled in
// the dual kernel). MAX_CONF_DIM is the correct size.
#define MAX_CONF_DIM (6 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + 2)
// Algorithm options (flexible-ligand search improvements).
// ROTAMER_BIAS: snap mutated torsions toward staggered rotamers (-60/+60/180 deg)+jitter,
//   since sp3 single bonds prefer staggered — shrinks the effective torsional search space.
#define ROTAMER_BIAS 1
#define MAX_NUM_OF_LIG_PAIRS 65536  // C(272,2)=36856 max theoretical; 65536 guarantees full coverage
#define MAX_NUM_OF_BFGS_STEPS 64
#define MAX_NUM_OF_RANDOM_MAP 20000 // not too large (stack overflow!)
#define GRIDS_SIZE 23

// QFD (Quantum Field Docking) grid indices — slots 17-22 appended after standard atom-type grids
#define GRID_IDX_ESP      17   // receptor electrostatic potential  [kcal/(mol·e)]
#define GRID_IDX_DESOLV   18   // desolvation susceptibility grid   [kcal/mol per unit q²]
#define GRID_IDX_INFOMAP  19   // information resonance field       [dimensionless coupling]
#define GRID_IDX_WATER    20   // explicit water displacement grid  [kcal/mol]
#define GRID_IDX_PIPI     21   // aromatic ring proximity grid      [dimensionless]
#define GRID_IDX_CAVITY   22   // pocket cavity / buriedness field   [0..1, 1 = deep pocket]

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
// Phase 6: cavity containment grid (shape-complementarity prior). The grid stores per-voxel
// EXPOSURE: 0 inside the pocket (buried) and 1 in solvent. Scoring adds e += QFD_CAVITY_WEIGHT *
// exposure (POSITIVE weight = a penalty for straying into solvent). Inside the pocket exposure is
// flat ~0, so there is NO gradient there and the force field alone decides the exact pose; only
// atoms leaving the pocket are pushed back in. This is the corrected formulation: an earlier
// "reward buriedness" version dragged the whole ligand to the single deepest void (5ofu 14.8 Å).
// Set 0 to disable (or omit the grid). Opt-in via env VINA_CAVITY=1.
// NOTE: this per-atom ADDITIVE form is a proven dead end (iteration 3) — it drags the ligand to the
// maximally-buried spot, ~4.5 Å off the crystal pose, weight-independently. Disabled in ig_eval_deriv.
#define QFD_CAVITY_WEIGHT (0.005f)

// Phase 6b: pose-level cavity RESCUE gate (iteration 4). Computed per pose in kernel2 (MC layer,
// NOT the BFGS gradient), so it never distorts an in-pocket optimisation. One-sided: a penalty
// QFD_CAVITY_RESCUE_W * gate * Σexposure added to the pose energy for Metropolis/best ranking,
// where gate = smoothstep(GATE_LO, GATE_HI) over the FRACTION of heavy atoms in solvent. In-pocket
// poses (fraction < GATE_LO) get gate=0 ⇒ exactly 0 rescue ⇒ do-no-harm; only poses with most atoms
// in solvent are down-ranked, pulling the search toward the pocket.
#define QFD_CAVITY_RESCUE_W   0.30f
#define QFD_CAV_GATE_LO       0.40f   // mean/fraction exposure below this ⇒ gate 0 (in pocket)
#define QFD_CAV_GATE_HI       0.70f   // mean/fraction exposure above this ⇒ gate 1 (in solvent)
#define QFD_CAV_EXPOSED_THR   0.50f   // an atom counts as "exposed" when grid exposure > this
// Phase 6c (iteration 6): GATED cavity GRADIENT. The per-atom pull-in force, scaled by the same
// pose-level gate (mean exposure over heavy atoms). gate≈0 in the pocket ⇒ no force ⇒ do-no-harm;
// gate>0 in solvent ⇒ active inward pull so a mislocalised ligand is dragged INTO the pocket during
// BFGS (not just re-ranked). Replaces the passive MC-acceptance rescue. 0 disables.
#define QFD_CAVITY_GRAD_W     0.10f
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

// Dual co-docking weight on Vina's (net-attractive) ligand-ligand interaction. Kept at 1
// (natural). Separation is enforced by an explicit STERIC-exclusion penalty instead, because
// Vina's pairwise lig-lig score is net-attractive even at overlap (gauss/hydrophobic outweigh
// the curl-capped repulsion), so simply up-weighting it REWARDS stacking.
#define QFD_LIGLIG_W      1.0f

// Steric exclusion between the two ligands: an uncapped quadratic penalty for every inter-atom
// pair closer than QFD_CLASH_THR. Dominates at true overlap so interpenetration is rejected,
// while contact at ~the threshold costs ~0 — ligands end up adjacent (like CPU Vina ~3 Å), not
// fused. Added inside eval_lig_lig_cl so all dual combined-energy/Metropolis checks see it.
#define QFD_CLASH_THR     2.6f
#define QFD_CLASH_THR2    (QFD_CLASH_THR * QFD_CLASH_THR)
#define QFD_CLASH_W       40.0f

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

// ════════════════ Native multi-ligand (P1): joint co-docking of up to MAX_NUM_OF_LIGANDS ════════════════
// Mirrors CPU AutoDock Vina 1.2.x: ONE model holds N ligands (built host-side via model.append()),
// each ligand an independent rigid root in a shared coordinate frame. Inter-ligand + ligand-flex
// pairs live in other_pairs (already populated by the model); intra-ligand pairs are per-ligand.
// Kept as a SEPARATE struct family + kernel2_multi so the single-ligand engine is untouched.

typedef struct { float coords[MAX_NUM_OF_MULTI_ATOMS][3]; } m_coords_multi_cl;
typedef struct { float coords[MAX_NUM_OF_MULTI_ATOMS][3]; } m_minus_forces_multi;

typedef struct {
	int num_pairs;
	int type_pair_index	[MAX_NUM_OF_LIG_PAIRS_MULTI];
	int a				[MAX_NUM_OF_LIG_PAIRS_MULTI];
	int b				[MAX_NUM_OF_LIG_PAIRS_MULTI];
} lig_pairs_multi_cl;

// Compact per-ligand tree (rigid_cl layout, small MAX_NUM_OF_RIGID_MULTI cap) — keeps
// m_multi_cl_private small enough to scale N within the GPU per-thread/backing limits.
typedef struct {
	int		num_children;
	int		parent			[MAX_NUM_OF_RIGID_MULTI];
	int		atom_range		[MAX_NUM_OF_RIGID_MULTI][2];
	float	origin			[MAX_NUM_OF_RIGID_MULTI][3];
	float	orientation_m	[MAX_NUM_OF_RIGID_MULTI][9];
	float	orientation_q	[MAX_NUM_OF_RIGID_MULTI][4];
	float	axis			[MAX_NUM_OF_RIGID_MULTI][3];
	float	relative_axis	[MAX_NUM_OF_RIGID_MULTI][3];
	float	relative_origin	[MAX_NUM_OF_RIGID_MULTI][3];
} rigid_multi_cl;

// One ligand inside a multi-ligand job. Atom indices reference the shared combined arrays.
typedef struct {
	int            begin;          // first atom index (into shared atom/coord arrays)
	int            end;            // one-past-last atom index
	int            torsion_offset; // this ligand owns lig_torsion[torsion_offset .. +num_torsions)
	int            num_torsions;
	rigid_multi_cl rigid;          // this ligand's own tree (root node 0 + branch nodes)
} ligand_multi_cl;

// conf (output_type) for a multi-ligand pose: one root (pos+quat) per ligand,
// torsions packed into one shared array partitioned by each ligand's torsion_offset.
typedef struct {
	float e;
	int   num_ligands;
	float position		[MAX_NUM_OF_LIGANDS][3];
	float orientation	[MAX_NUM_OF_LIGANDS][4];
	float lig_torsion	[MAX_NUM_OF_LIG_TORSION];
	float flex_torsion	[MAX_NUM_OF_FLEX_TORSION];
} output_type_multi_cl;

// change (gradient) for a multi-ligand conf: orientation is a 3-vector (axis-angle) per root.
typedef struct {
	int   num_ligands;
	float position		[MAX_NUM_OF_LIGANDS][3];
	float orientation	[MAX_NUM_OF_LIGANDS][3];
	float lig_torsion	[MAX_NUM_OF_LIG_TORSION];
	float flex_torsion	[MAX_NUM_OF_FLEX_TORSION];
} change_multi_cl;

// Global multi-ligand model (one per job). Intra pairs per ligand; cross pairs in other_pairs.
typedef struct {
	int m_num_movable_atoms;
	int num_ligands;
	atom_cl atoms[MAX_NUM_OF_MULTI_ATOMS];
	m_coords_multi_cl m_coords;
	m_minus_forces_multi minus_forces;
	ligand_multi_cl ligands[MAX_NUM_OF_LIGANDS];
	lig_pairs_multi_cl lig_pairs[MAX_NUM_OF_LIGANDS]; // intra-ligand pairs (global only)
	flex_rigid_cl flex_rigid;
	other_pairs_cl other_pairs;                       // inter-ligand + ligand-flex + flex-flex
} m_multi_cl;

// Per-work-item private copy: pair-free (pairs read from the global m_multi_cl), trees kept.
// At N=32 this is large (~500 KB); multi mode therefore runs fewer trajectories (see P2).
typedef struct {
	int m_num_movable_atoms;
	int num_ligands;
	atom_cl atoms[MAX_NUM_OF_MULTI_ATOMS];
	m_coords_multi_cl m_coords;
	m_minus_forces_multi minus_forces;
	ligand_multi_cl ligands[MAX_NUM_OF_LIGANDS];
	flex_rigid_cl flex_rigid;
} m_multi_cl_private;

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

