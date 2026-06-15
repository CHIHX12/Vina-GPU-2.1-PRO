#!/usr/bin/env python3
# Reliable redocking RMSD. Reference graph+coords come from the SDF (correct bond orders); docked
# pose heavy-atom coords are read from the docking pdbqt in the SAME order (the ligand pdbqt was
# prepared from this SDF, so heavy-atom order is preserved through docking). Symmetry-corrected
# via spyrmsd using the SDF graph. Usage: spyrmsd_sdf.py <ref.sdf> <docked.pdbqt>
import sys, numpy as np
from pathlib import Path
from spyrmsd import io, rmsd
ref = io.loadmol(sys.argv[1]); ref.strip()
nheavy = len(ref.atomicnums)
def models(p):
    out=[];cur=[]
    for l in Path(p).read_text().splitlines():
        if l[:5]=="MODEL": cur=[]
        elif l[:6] in ("ATOM  ","HETATM"):
            t=(l[77:].strip().split() or [""])[0]
            if t not in ("H","HD","HS"): cur.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        elif l[:6]=="ENDMDL": out.append(cur);cur=[]
    if cur: out.append(cur)
    return out
best=None
for m in models(sys.argv[2]):
    if len(m)!=nheavy: continue
    r=rmsd.symmrmsd(ref.coordinates, np.array(m,float), ref.atomicnums, ref.atomicnums, ref.adjacency_matrix, ref.adjacency_matrix)
    if best is None or r<best: best=r
print(f"{best:.2f}" if best is not None else f"NA(ref={nheavy})")
