#include "dual_mc.h"
#include "lig_lig_score.h"
#include "mutate.h"
#include "quasi_newton.h"
#include "coords.h"

namespace {

bool metropolis_accept_dual(fl old_f, fl new_f, fl temperature, rng& generator) {
    if (new_f < old_f) return true;
    const fl prob = std::exp((old_f - new_f) / temperature);
    return random_fl(0, 1, generator) < prob;
}

} // anonymous namespace

void dual_mc_search(
    model& ma, model& mb,
    const precalculate& p,
    const non_cache& nc_a, const non_cache& nc_b,
    const vec& corner1, const vec& corner2,
    sz num_steps, sz num_runs,
    fl temperature, fl mutation_amplitude,
    fl min_rmsd, sz num_saved_mins,
    output_container& out_a, output_container& out_b,
    rng& generator)
{
    // Use a large cap to disable curl (same role as authentic_v in Vina MC)
    const fl v = 1e8;
    const vec authentic_v(1000, 1000, 1000);

    conf_size sa = ma.get_size();
    conf_size sb = mb.get_size();
    change ga(sa), gb(sb);

    // Use the same BFGS step count as Vina's quick local minimizer
    quasi_newton qn;
    qn.max_steps = 20;

    for (sz run = 0; run < num_runs; run++) {
        // ---- Initial random conformations ----
        output_type cur_a(conf(sa), 0);
        output_type cur_b(conf(sb), 0);
        cur_a.c.randomize(corner1, corner2, generator);
        cur_b.c.randomize(corner1, corner2, generator);

        // Quick protein-only minimization from the random start
        ma.set(cur_a.c);
        mb.set(cur_b.c);
        qn(ma, p, nc_a, cur_a, ga, authentic_v);
        qn(mb, p, nc_b, cur_b, gb, authentic_v);
        ma.set(cur_a.c);
        mb.set(cur_b.c);

        cur_a.e = nc_a.eval(ma, v);
        cur_b.e = nc_b.eval(mb, v);
        fl e_ll  = eval_lig_lig(ma, mb, p);
        fl cur_combined = cur_a.e + cur_b.e + e_ll;

        fl best_combined = cur_combined;
        output_type best_a = cur_a;
        output_type best_b = cur_b;

        // ---- MC loop ----
        for (sz step = 0; step < num_steps; step++) {
            // 50/50 choice: mutate ligand A or B
            if (random_fl(0, 1, generator) < 0.5) {
                // --- Mutate ligand A ---
                output_type candidate_a(cur_a.c, max_fl);
                mutate_conf(candidate_a.c, ma, mutation_amplitude, generator);
                ma.set(candidate_a.c);
                qn(ma, p, nc_a, candidate_a, ga, authentic_v);
                ma.set(candidate_a.c);

                fl ea  = nc_a.eval(ma, v);
                fl new_ll = eval_lig_lig(ma, mb, p);
                fl new_combined = ea + cur_b.e + new_ll;

                if (metropolis_accept_dual(cur_combined, new_combined, temperature, generator)) {
                    candidate_a.e = ea;
                    cur_a = candidate_a;
                    cur_combined = new_combined;
                    if (new_combined < best_combined) {
                        best_combined = new_combined;
                        best_a = cur_a;
                        best_b = cur_b;
                    }
                } else {
                    ma.set(cur_a.c); // restore model A coords
                }
            } else {
                // --- Mutate ligand B ---
                output_type candidate_b(cur_b.c, max_fl);
                mutate_conf(candidate_b.c, mb, mutation_amplitude, generator);
                mb.set(candidate_b.c);
                qn(mb, p, nc_b, candidate_b, gb, authentic_v);
                mb.set(candidate_b.c);

                fl eb  = nc_b.eval(mb, v);
                fl new_ll = eval_lig_lig(ma, mb, p);
                fl new_combined = cur_a.e + eb + new_ll;

                if (metropolis_accept_dual(cur_combined, new_combined, temperature, generator)) {
                    candidate_b.e = eb;
                    cur_b = candidate_b;
                    cur_combined = new_combined;
                    if (new_combined < best_combined) {
                        best_combined = new_combined;
                        best_a = cur_a;
                        best_b = cur_b;
                    }
                } else {
                    mb.set(cur_b.c); // restore model B coords
                }
            }
        }

        // ---- Final refinement of best poses from this run ----
        ma.set(best_a.c);
        mb.set(best_b.c);
        qn(ma, p, nc_a, best_a, ga, authentic_v);
        qn(mb, p, nc_b, best_b, gb, authentic_v);
        ma.set(best_a.c);
        mb.set(best_b.c);

        best_a.coords = ma.get_heavy_atom_movable_coords();
        best_b.coords = mb.get_heavy_atom_movable_coords();
        best_a.e = nc_a.eval(ma, v);
        best_b.e = nc_b.eval(mb, v);

        add_to_output_container(out_a, best_a, min_rmsd, num_saved_mins);
        add_to_output_container(out_b, best_b, min_rmsd, num_saved_mins);
    }

    out_a.sort();
    out_b.sort();
}
