#!/usr/bin/env python3
# Best-of-N symmetry-corrected RMSD (spyrmsd) of docked poses vs a reference ligand.
# Usage: spyrmsd_best.py <ref.pdbqt> <docked.pdbqt>
import sys, subprocess, tempfile, os
OBABEL = "/home/cycheng/miniforge3/envs/jp_214/bin/obabel"
from spyrmsd import io, rmsd

def to_sdf(pdbqt):
    fd, sdf = tempfile.mkstemp(suffix=".sdf"); os.close(fd)
    subprocess.run([OBABEL, pdbqt, "-O", sdf], capture_output=True)
    return sdf

ref_sdf = to_sdf(sys.argv[1]); pose_sdf = to_sdf(sys.argv[2])
ref = io.loadmol(ref_sdf); ref.strip()
poses = io.loadallmols(pose_sdf)
best = None
ca, an, am = ref.coordinates, ref.atomicnums, ref.adjacency_matrix
for p in poses:
    p.strip()
    if len(p.atomicnums) != len(an):  # atom-count mismatch (bad conversion)
        continue
    try:
        r = rmsd.symmrmsd(ca, p.coordinates, an, p.atomicnums, am, p.adjacency_matrix)
        if best is None or r < best: best = r
    except Exception:
        pass
os.remove(ref_sdf); os.remove(pose_sdf)
print(f"{best:.2f}" if best is not None else "NA")
