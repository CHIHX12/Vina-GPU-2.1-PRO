#include "lig_lig_score.h"
#include "atom_type.h"

fl eval_lig_lig(const model& ma, const model& mb, const precalculate& p) {
    const fl cutoff_sqr = p.cutoff_sqr();
    const atom_type::t atu = p.atom_typing_used();
    const sz n = num_atom_types(atu);
    fl e = 0;

    const sz na = ma.num_movable_atoms();
    const sz nb = mb.num_movable_atoms();

    for (sz i = 0; i < na; i++) {
        const atom& a = ma.atoms[i];
        sz t1 = a.get(atu);
        if (t1 >= n) continue;
        const vec& ca = ma.coords[i];

        for (sz j = 0; j < nb; j++) {
            const atom& b = mb.atoms[j];
            sz t2 = b.get(atu);
            if (t2 >= n) continue;
            const vec& cb = mb.coords[j];
            vec r_ab; r_ab = ca - cb;
            fl r2 = sqr(r_ab);
            if (r2 < cutoff_sqr) {
                sz type_pair_index = get_type_pair_index(atu, a, b);
                e += p.eval_fast(type_pair_index, r2);
            }
        }
    }
    return e;
}
