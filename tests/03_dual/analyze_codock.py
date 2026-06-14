#!/usr/bin/env python3
"""
Synergy-vs-competition analysis for two-ligand co-docking.

Compares three dockings of the SAME receptor + box:
  - ligand A docked alone
  - ligand B docked alone
  - A and B docked together (co-docking)

and reports whether the two ligands are cooperative (coexist at ~no cost),
mildly competitive, or competitive (one is forced off its preferred site).

Usage:
  analyze_codock.py soloA.pdbqt soloB.pdbqt coA.pdbqt coB.pdbqt

The "verdict" is a methodology aid, NOT a biological claim — always validate
against known cases (e.g. a true ternary complex should read as coexisting),
and check stability over a few seeds / a deeper search before trusting a number.
"""
import math
import sys
from pathlib import Path

# A pair of heavy atoms closer than this (Å) is counted as a steric clash.
CLASH_CUTOFF = 2.5
# |E_A+E_B(together) - E_A+E_B(separate)| thresholds (kcal/mol) for the verdict.
COOP_MAX = 1.0   # <= this: cooperative / coexist (no real cost)
MILD_MAX = 4.0   # <= this: mild competition; above: competition


def first_pose_energy(pdbqt):
    """Vina affinity (kcal/mol) of the first MODEL from its REMARK line."""
    for line in Path(pdbqt).read_text().splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            return float(line.split()[3])
    raise ValueError(f"no 'REMARK VINA RESULT' found in {pdbqt}")


def first_pose_heavy_atoms(pdbqt):
    """Heavy-atom coords of the first MODEL (drops H/HD/HS by trailing element)."""
    pts = []
    for line in Path(pdbqt).read_text().splitlines():
        if line[:6] == "ENDMDL":
            break
        if line[:6] in ("ATOM  ", "HETATM"):
            tail = line[77:].strip()
            element = tail.split()[0] if tail else ""
            if element not in ("H", "HD", "HS"):
                pts.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return pts


def min_distance_and_clashes(a_pts, b_pts):
    mind = math.inf
    clashes = 0
    for x in a_pts:
        for y in b_pts:
            d = math.dist(x, y)
            if d < mind:
                mind = d
            if d < CLASH_CUTOFF:
                clashes += 1
    return mind, clashes


def same_order_rmsd(p, q):
    """Heavy-atom RMSD assuming identical atom ordering (same ligand prep)."""
    if len(p) != len(q):
        return None
    return math.sqrt(sum(math.dist(a, b) ** 2 for a, b in zip(p, q)) / len(p))


def main(argv):
    if len(argv) != 5:
        sys.exit("usage: analyze_codock.py soloA.pdbqt soloB.pdbqt coA.pdbqt coB.pdbqt")
    solo_a, solo_b, co_a, co_b = argv[1:5]

    ea_solo = first_pose_energy(solo_a)
    eb_solo = first_pose_energy(solo_b)
    ea_co = first_pose_energy(co_a)
    eb_co = first_pose_energy(co_b)

    a_solo = first_pose_heavy_atoms(solo_a)
    b_solo = first_pose_heavy_atoms(solo_b)
    a_co = first_pose_heavy_atoms(co_a)
    b_co = first_pose_heavy_atoms(co_b)

    sum_solo = ea_solo + eb_solo
    sum_co = ea_co + eb_co
    penalty = sum_co - sum_solo  # positive = co-docking costs energy

    co_mind, co_clash = min_distance_and_clashes(a_co, b_co)
    solo_mind, solo_clash = min_distance_and_clashes(a_solo, b_solo)
    shift_a = same_order_rmsd(a_solo, a_co)
    shift_b = same_order_rmsd(b_solo, b_co)

    bar = "=" * 64
    print(bar)
    print("  Two-ligand synergy-vs-competition comparison")
    print(bar)
    print(f"{'':24}{'A (kcal)':>11}{'B (kcal)':>11}{'A+B sum':>11}")
    print(f"{'docked SEPARATELY':24}{ea_solo:>11.1f}{eb_solo:>11.1f}{sum_solo:>11.1f}")
    print(f"{'docked TOGETHER':24}{ea_co:>11.1f}{eb_co:>11.1f}{sum_co:>11.1f}")
    print(f"{'change (together-sep)':24}{ea_co-ea_solo:>+11.1f}{eb_co-eb_solo:>+11.1f}{penalty:>+11.1f}")
    print("-" * 64)
    print(f"complex (A+B together): inter-ligand min dist = {co_mind:.2f} A, clashes = {co_clash}")
    print(f"two SOLO best poses superimposed: min dist = {solo_mind:.2f} A, clashes = {solo_clash}")
    sa = f"{shift_a:.2f} A" if shift_a is not None else "n/a (atom count differs)"
    sb = f"{shift_b:.2f} A" if shift_b is not None else "n/a (atom count differs)"
    print(f"site shift solo->complex:  A moved {sa},  B moved {sb}")
    print("-" * 64)
    print("READING:")
    if solo_clash > 0:
        print(f"  - The independently-preferred poses OVERLAP ({solo_clash} clashes): the")
        print(f"    ligands contend for shared space, so co-docking had to rearrange them.")
    else:
        print(f"  - The solo-preferred poses do NOT overlap: the sites look compatible.")
    if penalty <= COOP_MAX:
        print(f"  - Co-docking penalty {penalty:+.1f} kcal ~ 0  =>  COOPERATIVE / coexist (no real cost).")
    elif penalty <= MILD_MAX:
        print(f"  - Co-docking penalty {penalty:+.1f} kcal  =>  MILD competition (one slightly displaced).")
    else:
        print(f"  - Co-docking penalty {penalty:+.1f} kcal (large)  =>  COMPETITION (a ligand forced off its best site).")
    print("-" * 64)
    print("CAVEAT: this is a methodology aid, not a biological verdict. Validate against")
    print("known cases and check the penalty is stable over a few seeds / deeper search.")
    print(bar)


if __name__ == "__main__":
    main(sys.argv)
