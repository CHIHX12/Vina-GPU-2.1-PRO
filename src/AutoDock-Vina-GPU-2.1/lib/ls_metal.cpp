#include "ls_metal.h"
#include <cmath>

// LS metal coordination rescoring (Part A).
// Constants mirror kernel2.h — defined here so they're available in C++ code.
static const float LS_METAL_R_OPT_CPP  = 2.10f; // optimal coordination distance [Å]
static const float LS_METAL_SIGMA_CPP  = 0.20f; // tight Gaussian width [Å]
static const float LS_METAL_CUTOFF_CPP = 5.0f;  // per-metal search radius [Å]

// AD type range for metals: Mg=13 .. Hg=31 (see atom_constants.h)
bool is_receptor_metal(sz ad_type) {
    return ad_type >= 13 && ad_type < 32;
}

// Collect receptor metal atom positions from model's grid_atoms.
std::vector<std::array<float,3>> collect_receptor_metals(const model& m) {
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
void apply_ls_metal_scores(
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
