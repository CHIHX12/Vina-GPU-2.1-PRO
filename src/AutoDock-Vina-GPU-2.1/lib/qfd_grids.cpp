#include "qfd_grids.h"
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>

// QFD grid binary format (written by scripts/prep_qfd_grids.py):
//   int[3]   : m_i, m_j, m_k
//   float[12]: m_init[3], m_factor[3], m_dim_fl_minus_1[3], m_factor_inv[3]
//   float[m_i*m_j*m_k*8]: trilinear interpolation coefficients
// Returns true and fills slot if loaded; returns false and leaves slot zeroed if file absent.
bool load_qfd_grid_file(const std::string& path, grid_cl* slot, const grid& ref_grid) {
	FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	int dims[3];
	float meta[12];   // m_init[3], m_factor[3], m_dim_fl_minus_1[3], m_factor_inv[3]
	if (std::fread(dims, sizeof(int), 3, f) != 3 || std::fread(meta, sizeof(float), 12, f) != 12) {
		std::fclose(f); return false;
	}
	int mi = dims[0], mj = dims[1], mk = dims[2];
	if (mi <= 0 || mj <= 0 || mk <= 0 || mi > MAX_NUM_OF_GRID_DIM || mj > MAX_NUM_OF_GRID_DIM || mk > MAX_NUM_OF_GRID_DIM) {
		std::fclose(f); return false;
	}
	// Staleness guard: reject a cache whose spatial frame does not COVER the current docking box.
	// Without this, a qfd_*.bin left in the CWD from a *different* receptor is loaded blindly and
	// silently corrupts every score (observed: a DHFR pipi grid wrongly applied to another target
	// flipped a -8.3 redock to +52). A grid built for (or covering) this box passes; one offset by
	// tens of Å for another protein fails on at least one axis.
	{
		const float* gi  = meta + 0;   // loaded grid origin
		const float* gsp = meta + 9;   // loaded grid spacing (m_factor_inv)
		const float lmin[3] = { gi[0], gi[1], gi[2] };
		const float lmax[3] = { gi[0] + (mi - 1) * gsp[0],
		                        gi[1] + (mj - 1) * gsp[1],
		                        gi[2] + (mk - 1) * gsp[2] };
		const float bs = (float)ref_grid.m_factor_inv[0];
		const float bmin[3] = { (float)ref_grid.m_init[0], (float)ref_grid.m_init[1], (float)ref_grid.m_init[2] };
		const float bmax[3] = { bmin[0] + ((float)ref_grid.m_data.dim0() - 1) * bs,
		                        bmin[1] + ((float)ref_grid.m_data.dim1() - 1) * bs,
		                        bmin[2] + ((float)ref_grid.m_data.dim2() - 1) * bs };
		const float tol = 2.0f;  // [Å] absorb spacing/rounding differences between builders
		for (int ax = 0; ax < 3; ax++) {
			if (lmin[ax] > bmin[ax] + tol || lmax[ax] < bmax[ax] - tol) {
				std::fclose(f);
				printf("QFD: ignoring %s — frame does not cover current box (stale cache from a different receptor/box)\n",
				       path.c_str());
				return false;
			}
		}
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

// ── QFD pipi grid: built directly from receptor aromatic carbons ────────────
// Eliminates the need to run prep_qfd_grids.py before docking.
// AD_TYPE_A = 1 (aromatic carbon, see atom_constants.h).
// Matches the Python formula in prep_qfd_grids.py compute_pipi_grid().
static const float PIPI_SIGMA_CPP   = 2.5f;  // Gaussian width [Å]
static const float PIPI_SPACING_CPP = 1.0f;  // grid spacing [Å]

bool compute_pipi_grid_from_receptor(
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

// ── QFD cavity containment grid: per-voxel EXPOSURE (shape-complementarity prior) ──
// exposure = 0 inside the pocket (buried, where the ligand belongs), 1 in solvent. Penalizing
// exposed ligand atoms keeps the ligand inside the concavity without dragging it to the deepest
// void (the earlier "reward buriedness" version did exactly that → 5ofu 14.8 A). Fast: a binary
// occupancy grid is stamped from the receptor atoms once, then buriedness is found by integer
// ray-marching that occupancy grid in 14 directions (no per-step atom-distance math).
static const float CAV_SPACING  = 1.0f;   // grid spacing [A]
static const float CAV_CLASH    = 2.0f;   // a voxel is "protein" if within this of any atom
static const float CAV_RAY_MAX  = 9.0f;   // ray length [A]
// Dead-zone thresholds: exposure ramps 1→0 only across buriedness [CAV_LO, CAV_HI]. Set LOW so the
// whole pocket + its walls (where ligand atoms legitimately sit, buriedness > ~0.35) is a FLAT
// exposure≈0 dead-zone with no gradient — the cavity then cannot bias an already-in-pocket pose
// (that was the weight-independent ~4 Å shift). Only clearly-exposed atoms (buriedness < ~0.35,
// i.e. sticking out into solvent) feel the pull-in penalty.
static const float CAV_LO       = 0.15f;  // buriedness below this ⇒ fully exposed (penalty 1)
static const float CAV_HI       = 0.35f;  // buriedness above this ⇒ buried (penalty 0, dead-zone)

bool compute_cavity_grid_from_receptor(
    const model& m,
    const grid& ref_grid,   // any initialized Vina atom-type grid (box reference)
    grid_cl* slot)
{
    std::vector<std::array<float,3>> P;
    P.reserve(m.grid_atoms.size());
    for (sz i = 0; i < m.grid_atoms.size(); i++)
        P.push_back({(float)m.grid_atoms[i].coords[0],
                     (float)m.grid_atoms[i].coords[1],
                     (float)m.grid_atoms[i].coords[2]});
    if (P.empty()) return false;

    // Output grid frame from the reference Vina grid (same as pipi)
    const float bs  = (float)ref_grid.m_factor_inv[0];
    const float bmi = (float)ref_grid.m_data.dim0() - 1.0f;
    const float bmj = (float)ref_grid.m_data.dim1() - 1.0f;
    const float bmk = (float)ref_grid.m_data.dim2() - 1.0f;
    if (bmi <= 0.0f || bmj <= 0.0f || bmk <= 0.0f) return false;
    const float cx = (float)ref_grid.m_init[0] + bmi * 0.5f * bs;
    const float cy = (float)ref_grid.m_init[1] + bmj * 0.5f * bs;
    const float cz = (float)ref_grid.m_init[2] + bmk * 0.5f * bs;
    const float sx = bmi * bs, sy = bmj * bs, sz = bmk * bs;

    const float sp = CAV_SPACING;
    // Extend the cavity grid GMARGIN voxels BEYOND the docking box on every side. The kernel's
    // g_evaluate applies grids->slope (Vina's ~1e6 box-containment penalty) to any atom outside a
    // grid's bounds; if the cavity grid only covered the box, heavy atoms exploring near the box
    // edge would hit that slope (× weight) and blow up the optimisation. Padding keeps every in-box
    // atom strictly inside the cavity grid (region=0, no slope term).
    const int GMARGIN = 6;
    int mi = (int)std::round(sx / sp) + 1 + 2*GMARGIN;
    int mj = (int)std::round(sy / sp) + 1 + 2*GMARGIN;
    int mk = (int)std::round(sz / sp) + 1 + 2*GMARGIN;
    if (mi > MAX_NUM_OF_GRID_MI) mi = MAX_NUM_OF_GRID_MI;
    if (mj > MAX_NUM_OF_GRID_MJ) mj = MAX_NUM_OF_GRID_MJ;
    if (mk > MAX_NUM_OF_GRID_MK) mk = MAX_NUM_OF_GRID_MK;
    const float ox = cx - (mi - 1) * 0.5f * sp;  // still centred on the box, just larger
    const float oy = cy - (mj - 1) * 0.5f * sp;
    const float oz = cz - (mk - 1) * 0.5f * sp;

    // Occupancy grid: covers the output box padded by CAV_RAY_MAX so rays stay inside it.
    const int   pad = (int)std::ceil(CAV_RAY_MAX / sp);
    const int   omi = mi + 2*pad, omj = mj + 2*pad, omk = mk + 2*pad;
    const float oox = ox - pad*sp, ooy = oy - pad*sp, ooz = oz - pad*sp;
    std::vector<unsigned char> occ((size_t)omi*omj*omk, 0);
    auto oidx = [&](int i,int j,int k){ return ((size_t)i*omj + j)*omk + k; };
    // Stamp each receptor atom: mark occupancy voxels within CAV_CLASH.
    const int srad = (int)std::ceil(CAV_CLASH / sp);
    const float clash2 = CAV_CLASH * CAV_CLASH;
    for (const auto& a : P) {
        const int ai = (int)std::lround((a[0]-oox)/sp);
        const int aj = (int)std::lround((a[1]-ooy)/sp);
        const int ak = (int)std::lround((a[2]-ooz)/sp);
        for (int di=-srad; di<=srad; di++) for (int dj=-srad; dj<=srad; dj++) for (int dk=-srad; dk<=srad; dk++) {
            const int i=ai+di, j=aj+dj, k=ak+dk;
            if (i<0||j<0||k<0||i>=omi||j>=omj||k>=omk) continue;
            const float vx=oox+i*sp - a[0], vy=ooy+j*sp - a[1], vz=ooz+k*sp - a[2];
            if (vx*vx+vy*vy+vz*vz < clash2) occ[oidx(i,j,k)] = 1;
        }
    }

    // 14 ray directions (6 axes + 8 cube diagonals), expressed in voxel steps.
    const float sdg = 1.0f/std::sqrt(3.0f);
    const float dirs[14][3] = {
        { 1,0,0},{-1,0,0},{0, 1,0},{0,-1,0},{0,0, 1},{0,0,-1},
        { sdg, sdg, sdg},{ sdg, sdg,-sdg},{ sdg,-sdg, sdg},{ sdg,-sdg,-sdg},
        {-sdg, sdg, sdg},{-sdg, sdg,-sdg},{-sdg,-sdg, sdg},{-sdg,-sdg,-sdg}
    };
    const int nsteps = (int)(CAV_RAY_MAX / sp);

    std::vector<float> raw((size_t)mi*mj*mk, 0.0f);  // stores EXPOSURE [0,1]
    for (int i = 0; i < mi; i++) {
        for (int j = 0; j < mj; j++) {
            for (int k = 0; k < mk; k++) {
                const int oi = i+pad, oj = j+pad, ok = k+pad;   // index into occ
                if (occ[oidx(oi,oj,ok)]) { raw[(size_t)(i*mj + j)*mk + k] = 1.0f; continue; }  // inside protein: penalty (don't pull ligand into walls; the only exposure≈0 basin must be the empty pocket)
                int blocked = 0;
                for (int d = 0; d < 14; d++) {
                    for (int st = 1; st <= nsteps; st++) {
                        const int qi = oi + (int)std::lround(dirs[d][0]*st);
                        const int qj = oj + (int)std::lround(dirs[d][1]*st);
                        const int qk = ok + (int)std::lround(dirs[d][2]*st);
                        if (qi<0||qj<0||qk<0||qi>=omi||qj>=omj||qk>=omk) break;
                        if (occ[oidx(qi,qj,qk)]) { blocked++; break; }
                    }
                }
                const float bur = blocked / 14.0f;                         // buriedness
                float t = (bur - CAV_LO) / (CAV_HI - CAV_LO);              // → 0 exposed .. 1 buried
                t = std::max(0.0f, std::min(1.0f, t));
                raw[(size_t)(i*mj + j)*mk + k] = 1.0f - t;                  // EXPOSURE = 1 - buriedness
            }
        }
    }

    // Pack to Vina 8-corner trilinear format (identical to pipi)
    auto vget = [&](int i,int j,int k)->float{
        i = std::max(0,std::min(mi-1,i)); j = std::max(0,std::min(mj-1,j)); k = std::max(0,std::min(mk-1,k));
        return raw[(size_t)(i*mj + j)*mk + k];
    };
    for (int i = 0; i < mi; i++) for (int j = 0; j < mj; j++) for (int k = 0; k < mk; k++) {
        const size_t base = ((size_t)(i*mj + j)*mk + k) * 8;
        slot->m_data[base+0]=vget(i,j,k);     slot->m_data[base+1]=vget(i+1,j,k);
        slot->m_data[base+2]=vget(i,j+1,k);   slot->m_data[base+3]=vget(i+1,j+1,k);
        slot->m_data[base+4]=vget(i,j,k+1);   slot->m_data[base+5]=vget(i+1,j,k+1);
        slot->m_data[base+6]=vget(i,j+1,k+1); slot->m_data[base+7]=vget(i+1,j+1,k+1);
    }
    slot->m_i=mi; slot->m_j=mj; slot->m_k=mk;
    slot->m_init[0]=ox; slot->m_init[1]=oy; slot->m_init[2]=oz;
    const float inv_sp = 1.0f/sp;
    slot->m_factor[0]=slot->m_factor[1]=slot->m_factor[2]=inv_sp;
    slot->m_dim_fl_minus_1[0]=(float)(mi-1); slot->m_dim_fl_minus_1[1]=(float)(mj-1); slot->m_dim_fl_minus_1[2]=(float)(mk-1);
    slot->m_factor_inv[0]=slot->m_factor_inv[1]=slot->m_factor_inv[2]=sp;

    int nburied = 0; for (float v : raw) if (v < 0.5f) nburied++;
    const float exp_center = raw[(size_t)((mi/2)*mj + mj/2)*mk + mk/2];
    const float exp_corner = raw[(size_t)((size_t)1*mj + 1)*mk + 1];
    printf("QFD cavity: containment(exposure) grid dims=%d×%d×%d, %d buried voxels (exposure<0.5)\n", mi, mj, mk, nburied);
    printf("  DIAG exposure: box-center(pocket)=%.2f  box-corner(solvent)=%.2f  (want center≈0, corner≈1)\n",
           exp_center, exp_corner);
    return true;
}

// ── Cavity-biased initial placement: collect buried (pocket) voxel centres in the box ───────────
static const float POCKET_BURIED_THR = 0.45f;  // buriedness >= this ⇒ pocket interior
std::vector<std::array<float,3>> cavity_pocket_points(const model& m, const grid& ref_grid) {
    std::vector<std::array<float,3>> pts;
    std::vector<std::array<float,3>> P;
    for (sz i = 0; i < m.grid_atoms.size(); i++)
        P.push_back({(float)m.grid_atoms[i].coords[0],(float)m.grid_atoms[i].coords[1],(float)m.grid_atoms[i].coords[2]});
    if (P.empty()) return pts;
    const float bs  = (float)ref_grid.m_factor_inv[0];
    const float bmi = (float)ref_grid.m_data.dim0()-1.0f, bmj=(float)ref_grid.m_data.dim1()-1.0f, bmk=(float)ref_grid.m_data.dim2()-1.0f;
    if (bmi<=0.0f||bmj<=0.0f||bmk<=0.0f) return pts;
    const float cx=(float)ref_grid.m_init[0]+bmi*0.5f*bs, cy=(float)ref_grid.m_init[1]+bmj*0.5f*bs, cz=(float)ref_grid.m_init[2]+bmk*0.5f*bs;
    const float sx=bmi*bs, sy=bmj*bs, sz=bmk*bs, sp=CAV_SPACING;
    int mi=(int)std::round(sx/sp)+1, mj=(int)std::round(sy/sp)+1, mk=(int)std::round(sz/sp)+1;  // box only (no GMARGIN)
    if (mi>MAX_NUM_OF_GRID_MI) mi=MAX_NUM_OF_GRID_MI;
    if (mj>MAX_NUM_OF_GRID_MJ) mj=MAX_NUM_OF_GRID_MJ;
    if (mk>MAX_NUM_OF_GRID_MK) mk=MAX_NUM_OF_GRID_MK;
    const float ox=cx-(mi-1)*0.5f*sp, oy=cy-(mj-1)*0.5f*sp, oz=cz-(mk-1)*0.5f*sp;
    const int pad=(int)std::ceil(CAV_RAY_MAX/sp);
    const int omi=mi+2*pad, omj=mj+2*pad, omk=mk+2*pad;
    const float oox=ox-pad*sp, ooy=oy-pad*sp, ooz=oz-pad*sp;
    std::vector<unsigned char> occ((size_t)omi*omj*omk,0);
    auto oidx=[&](int i,int j,int k){ return ((size_t)i*omj+j)*omk+k; };
    const int srad=(int)std::ceil(CAV_CLASH/sp); const float clash2=CAV_CLASH*CAV_CLASH;
    for (const auto& a:P){
        const int ai=(int)std::lround((a[0]-oox)/sp), aj=(int)std::lround((a[1]-ooy)/sp), ak=(int)std::lround((a[2]-ooz)/sp);
        for (int di=-srad; di<=srad; di++) for (int dj=-srad; dj<=srad; dj++) for (int dk=-srad; dk<=srad; dk++){
            const int i=ai+di,j=aj+dj,k=ak+dk;
            if (i<0||j<0||k<0||i>=omi||j>=omj||k>=omk) continue;
            const float vx=oox+i*sp-a[0], vy=ooy+j*sp-a[1], vz=ooz+k*sp-a[2];
            if (vx*vx+vy*vy+vz*vz < clash2) occ[oidx(i,j,k)]=1;
        }
    }
    const float sdg=1.0f/std::sqrt(3.0f);
    const float dirs[14][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
        {sdg,sdg,sdg},{sdg,sdg,-sdg},{sdg,-sdg,sdg},{sdg,-sdg,-sdg},{-sdg,sdg,sdg},{-sdg,sdg,-sdg},{-sdg,-sdg,sdg},{-sdg,-sdg,-sdg}};
    const int nsteps=(int)(CAV_RAY_MAX/sp);
    for (int i=0;i<mi;i++) for (int j=0;j<mj;j++) for (int k=0;k<mk;k++){
        const int oi=i+pad,oj=j+pad,ok=k+pad;
        if (occ[oidx(oi,oj,ok)]) continue;               // inside protein
        int blocked=0;
        for (int d=0;d<14;d++) for (int st=1; st<=nsteps; st++){
            const int qi=oi+(int)std::lround(dirs[d][0]*st), qj=oj+(int)std::lround(dirs[d][1]*st), qk=ok+(int)std::lround(dirs[d][2]*st);
            if (qi<0||qj<0||qk<0||qi>=omi||qj>=omj||qk>=omk) break;
            if (occ[oidx(qi,qj,qk)]){ blocked++; break; }
        }
        if (blocked/14.0f >= POCKET_BURIED_THR) pts.push_back({ox+i*sp, oy+j*sp, oz+k*sp});
    }
    return pts;
}
