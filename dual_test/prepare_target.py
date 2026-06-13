#!/usr/bin/env python3
"""
prepare_target.py PDB_ID [PDB_ID ...]

Download a PDB, auto-detect the two closest drug-like ligands (a two-ligand
pocket), and prepare everything for dual co-docking into dual_test/<ID>/:
  rec.pdbqt           receptor (prepare_receptor4)
  ligA.pdbqt ligB.pdbqt  the two ligands prepped for docking (obabel, pH 7.4)
  ligA_xtal.pdbqt ligB_xtal.pdbqt  crystal-pose references (for RMSD validation)
  box.txt             docking box: center_x y z size_x y z (spans both ligands)
  info.txt            ligand names, heavy-atom counts, torsions, centroid gap

Drug-like = HETATM residue, >=6 heavy atoms, not water/ion/common buffer.
The chosen pair is the two such ligands whose centroids are closest (one pocket).
"""
import sys, subprocess, os, math, itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDA = "/home/cycheng/miniforge3/envs/jp_214"
PREP_REC = f"{CONDA}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
OBABEL = f"{CONDA}/bin/obabel"
PYMGL = f"{CONDA}/bin/python"

EXCLUDE = {"HOH","WAT","NA","CL","K","CA","MG","ZN","MN","FE","CU","NI","CO","CD","SO4","PO4",
           "GOL","EDO","PEG","PGE","ACT","DMS","MES","TRS","IOD","BR","NO3","FMT","BME","EPE",
           "NAG","MAN","BMA","FUC","CIT","TLA","MPD","IMD","ACY","CO3","FLC","P6G","1PE","SPM"}

def parse_hetatm(pdb):
    groups = {}  # (resn,chain,resi) -> list of (x,y,z,elem,line)
    for line in Path(pdb).read_text().splitlines():
        if line[:6] != "HETATM": continue
        resn = line[17:20].strip(); chain = line[21]; resi = line[22:27].strip()
        elem = line[76:78].strip()
        if elem == "H" or elem == "D": continue
        x,y,z = float(line[30:38]),float(line[38:46]),float(line[46:54])
        groups.setdefault((resn,chain,resi), []).append((x,y,z,line))
    return groups

def centroid(atoms):
    n=len(atoms); return (sum(a[0] for a in atoms)/n, sum(a[1] for a in atoms)/n, sum(a[2] for a in atoms)/n)

def prepare(pid):
    pid = pid.upper()
    outdir = HERE/pid; outdir.mkdir(exist_ok=True)
    pdb = outdir/f"{pid}.pdb"
    if not pdb.exists() or pdb.stat().st_size == 0:
        subprocess.run(["curl","-sL","-m","60","-o",str(pdb),
                        f"https://files.rcsb.org/download/{pid}.pdb"], timeout=90)
    if not pdb.exists() or pdb.stat().st_size == 0:
        print(f"{pid}: download FAILED"); return
    groups = parse_hetatm(pdb)
    # drug-like candidates
    cands = {k:v for k,v in groups.items() if k[0] not in EXCLUDE and len(v) >= 6}
    if len(cands) < 2:
        print(f"{pid}: only {len(cands)} drug-like ligand(s) found {[k[0] for k in cands]} — not a two-ligand pocket"); return
    # pick the closest pair of centroids
    cents = {k:centroid(v) for k,v in cands.items()}
    best=None
    for a,b in itertools.combinations(cands.keys(),2):
        d=math.dist(cents[a],cents[b])
        if best is None or d<best[0]: best=(d,a,b)
    gap,ka,kb = best
    # extract protein (all ATOM) + the two ligands
    prot=[l for l in Path(pdb).read_text().splitlines() if l[:4]=="ATOM"]
    (outdir/"rec.pdb").write_text("\n".join(prot)+"\n")
    for tag,key in [("A",ka),("B",kb)]:
        lines=[a[3] for a in cands[key]]
        (outdir/f"lig{tag}_xtal.pdb").write_text("\n".join(lines)+"\n")
    # prepare receptor
    subprocess.run([PYMGL,PREP_REC,"-r",str(outdir/"rec.pdb"),"-o",str(outdir/"rec.pdbqt"),
                    "-A","hydrogens"],capture_output=True,timeout=300)
    # prepare ligands (docking input + crystal reference are the same coords)
    tors={}
    for tag in ("A","B"):
        subprocess.run([OBABEL,"-ipdb",str(outdir/f"lig{tag}_xtal.pdb"),"-opdbqt",
                        "-O",str(outdir/f"lig{tag}.pdbqt"),"-p","7.4"],capture_output=True,timeout=120)
        # crystal ref = same prep (used for RMSD)
        subprocess.run(["cp",str(outdir/f"lig{tag}.pdbqt"),str(outdir/f"lig{tag}_xtal.pdbqt")])
        txt=(outdir/f"lig{tag}.pdbqt").read_text() if (outdir/f"lig{tag}.pdbqt").exists() else ""
        tors[tag]=txt.count("BRANCH")//1
    # box spanning both ligands
    allat=cands[ka]+cands[kb]
    xs=[a[0] for a in allat]; ys=[a[1] for a in allat]; zs=[a[2] for a in allat]
    cx,cy,cz=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2,(min(zs)+max(zs))/2
    sx,sy,sz=[max(mx-mn+8.0,16.0) for mn,mx in [(min(xs),max(xs)),(min(ys),max(ys)),(min(zs),max(zs))]]
    (outdir/"box.txt").write_text(f"{cx:.3f} {cy:.3f} {cz:.3f} {sx:.1f} {sy:.1f} {sz:.1f}\n")
    info=(f"{pid}: ligA={ka[0]}(chain {ka[1]}, {len(cands[ka])} heavy, {tors['A']} tors)  "
          f"ligB={kb[0]}(chain {kb[1]}, {len(cands[kb])} heavy, {tors['B']} tors)  "
          f"centroid_gap={gap:.1f}A  box=({sx:.0f},{sy:.0f},{sz:.0f})")
    (outdir/"info.txt").write_text(info+"\n")
    flexnote = " [DRUG-LIKE: clean test]" if max(tors['A'],tors['B'])<=10 else " [flexible: hard]"
    print(info+flexnote)

if __name__=="__main__":
    ids = sys.argv[1:]
    if not ids:
        ids=[l.split()[0] for l in (HERE/"targets.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    for pid in ids:
        try: prepare(pid)
        except Exception as e: print(f"{pid}: ERROR {e}")
