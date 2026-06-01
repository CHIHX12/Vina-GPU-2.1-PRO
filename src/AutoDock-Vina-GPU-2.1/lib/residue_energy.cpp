/*
 * residue_energy.cpp
 *
 * Per-residue energy decomposition.
 * Loops over all ligand × receptor atom pairs within the Vina cutoff (8 Å),
 * accumulates the precalculated pair energy onto the owning receptor residue.
 */

#include "residue_energy.h"
#include "precalculate.h"
#include "atom_constants.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>

// ── helpers ──────────────────────────────────────────────────────────────────

static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

static bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

// ── parse_receptor_residue_labels ────────────────────────────────────────────

ResidueLabels parse_receptor_residue_labels(const std::string& receptor_path) {
    ResidueLabels labels;
    std::ifstream in(receptor_path);
    if (!in) return labels;

    std::string line;
    while (std::getline(in, line)) {
        if (!starts_with(line, "ATOM  ") && !starts_with(line, "HETATM"))
            continue;

        // PDB/PDBQT column layout (1-indexed):
        //   17-20  residue name
        //   22     chain ID
        //   23-26  residue sequence number
        if (line.size() < 26) { labels.push_back("?:UNK:0"); continue; }

        std::string resname = trim(line.substr(17, 3));
        char chain   = line[21];
        std::string resnum_str = trim(line.substr(22, 4));

        if (resname.empty()) resname = "UNK";
        if (chain == ' ')    chain   = '_';

        std::string label;
        label += chain;
        label += ':';
        label += resname;
        label += ':';
        label += resnum_str.empty() ? "0" : resnum_str;
        labels.push_back(label);
    }
    return labels;
}

// ── compute_residue_energy ────────────────────────────────────────────────────

ResidueEnergy compute_residue_energy(const model& m,
                                     const precalculate& prec,
                                     const ResidueLabels& labels) {
    ResidueEnergy result;

    const fl cutoff_sqr = prec.cutoff_sqr();
    const sz  n_types   = num_atom_types(prec.atom_typing_used());
    const sz  n_rec     = m.grid_atoms.size();

    if (labels.size() < n_rec) return result; // safety: labels don't cover all atoms

    for (sz i = 0; i < m.num_movable_atoms(); ++i) {
        const atom& lig_atom = m.atoms[i];
        sz t1 = lig_atom.get(prec.atom_typing_used());
        if (t1 >= n_types) continue;            // hydrogen or unknown type

        const vec& lc = m.coords[i];

        for (sz j = 0; j < n_rec; ++j) {
            const atom& rec_atom = m.grid_atoms[j];
            sz t2 = rec_atom.get(prec.atom_typing_used());
            if (t2 >= n_types) continue;

            vec dr;
            dr[0] = lc[0] - rec_atom.coords[0];
            dr[1] = lc[1] - rec_atom.coords[1];
            dr[2] = lc[2] - rec_atom.coords[2];
            fl r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];

            if (r2 >= cutoff_sqr) continue;

            sz tpi = get_type_pair_index(prec.atom_typing_used(), lig_atom, rec_atom);
            fl e   = prec.eval_fast(tpi, r2);

            result[labels[j]] += e;
        }
    }

    return result;
}

// ── write_residue_energy ─────────────────────────────────────────────────────

void write_residue_energy(const ResidueEnergy& contrib,
                          std::ostream& out,
                          int pose_index) {
    if (contrib.empty()) return;

    // Collect and sort by energy (most negative first)
    std::vector<std::pair<fl, std::string>> sorted;
    for (const auto& kv : contrib)
        sorted.emplace_back(kv.second, kv.first);
    std::sort(sorted.begin(), sorted.end());

    out << "## Pose " << pose_index << "\n";
    out << "pose\tchain\tresname\tresnum\tenergy_kcal_mol\n";

    for (const auto& kv : sorted) {
        const std::string& label = kv.second;
        fl energy = kv.first;

        // Parse label "CHAIN:RESNAME:RESNUM"
        std::string chain, resname, resnum;
        std::istringstream ss(label);
        std::getline(ss, chain,   ':');
        std::getline(ss, resname, ':');
        std::getline(ss, resnum);

        out << pose_index << "\t"
            << chain   << "\t"
            << resname << "\t"
            << resnum  << "\t"
            << std::fixed << std::setprecision(4) << energy << "\n";
    }
    out << "\n";
}

void write_residue_energy_file(const ResidueEnergy& contrib,
                               const std::string& path,
                               int pose_index) {
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::cerr << "Warning: cannot open residue energy output: " << path << "\n";
        return;
    }
    if (pose_index == 1)
        out << "# Per-residue energy decomposition (Vina-GPU PRO)\n"
            << "# Negative = stabilising, Positive = clashing\n\n";
    write_residue_energy(contrib, out, pose_index);
}
