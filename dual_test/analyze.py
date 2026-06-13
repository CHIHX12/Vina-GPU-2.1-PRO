#!/usr/bin/env python3
"""
analyze.py PDB_ID
Compare docked poses to the crystal for both ligands, and give a
synergy/competition readout for the two-ligand pocket.

Reports per ligand (A,B): best-of-N heavy-atom RMSD vs crystal, for
  - Vina-GPU dual  (gpu_ligA/B.pdbqt)
  - Vina 1.2.7     (cpu_both.pdbqt, split into the two ligands by vina_split)
Then inter-ligand: clashes / contacts between the two docked best poses
(competition if they overlap, synergy candidate if adjacent & in contact).
"""
import sys, math, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent

def models(p):
    p=Path(p)
    if not p.exists(): return []
    blocks=[]; cur=[]; inmodel=False
    for l in p.read_text().splitlines():
        if l[:5]=="MODEL": cur=[]; inmodel=True
        elif l[:6]=="ENDMDL": blocks.append(cur); inmodel=False
        elif l[:6] in ("ATOM  ","HETATM"): cur.append(l)
    if not blocks and cur: blocks=[cur]
    return blocks

def heavy(lines):
    out=[]
    for l in lines:
        el=l[76:78].strip(); nm=l[12:16].strip()
        if el in ("H","D") or (not el and nm[:1]=="H"): continue
        out.append((float(l[30:38]),float(l[38:46]),float(l[46:54])))
    return out

def rmsd(a,b):
    n=min(len(a),len(b))
    if len(a)!=len(b) or n==0: return None
    return math.sqrt(sum((a[i][0]-b[i][0])**2+(a[i][1]-b[i][1])**2+(a[i][2]-b[i][2])**2 for i in range(n))/n)

def best_rmsd(dock, xtal):
    cry=models(xtal)
    if not cry: return None
    ch=heavy(cry[0]); best=None
    for m in models(dock):
        r=rmsd(heavy(m),ch)
        if r is not None and (best is None or r<best): best=r
    return best

def cpu_multiligand_poses(cpu_both):
    """Vina 1.2.x multi-ligand output: each MODEL holds BOTH ligands, each as a
    ROOT..TORSDOF block. Returns (ligA_poses, ligB_poses) as lists of atom-line lists."""
    p=Path(cpu_both)
    if not p.exists(): return [],[]
    A=[]; B=[]
    cur_ligs=[]; cur=[]
    for l in p.read_text().splitlines():
        if l[:5]=="MODEL": cur_ligs=[]; cur=[]
        elif l[:6] in ("ATOM  ","HETATM"): cur.append(l)
        elif l[:7]=="TORSDOF":
            if cur: cur_ligs.append(cur); cur=[]
        elif l[:6]=="ENDMDL":
            if len(cur_ligs)>=2: A.append(cur_ligs[0]); B.append(cur_ligs[1])
    return A,B

def best_rmsd_poses(poses, xtal):
    cry=models(xtal)
    if not cry or not poses: return None
    ch=heavy(cry[0]); best=None
    for m in poses:
        r=rmsd(heavy(m),ch)
        if r is not None and (best is None or r<best): best=r
    return best

def interligand(a_lines,b_lines):
    A=heavy(a_lines); B=heavy(b_lines)
    if not A or not B: return None
    cl=sum(1 for x in A for y in B if math.dist(x,y)<2.0)
    close=sum(1 for x in A for y in B if math.dist(x,y)<4.0)
    return cl, close

def main(pid):
    pid=pid.upper(); D=HERE/pid
    print(f"=== {pid} : {(D/'info.txt').read_text().strip() if (D/'info.txt').exists() else ''}")
    print(f"{'ligand':8s} {'GPU best':>9s} {'CPU1.2.7 best':>13s}")
    cpuA_poses,cpuB_poses = cpu_multiligand_poses(D/"cpu_both.pdbqt")
    for tag,xtal,gpu,cpu_poses in [("A",D/"ligA_xtal.pdbqt",D/"gpu_ligA.pdbqt",cpuA_poses),
                                   ("B",D/"ligB_xtal.pdbqt",D/"gpu_ligB.pdbqt",cpuB_poses)]:
        g=best_rmsd(gpu,xtal); c=best_rmsd_poses(cpu_poses,xtal)
        print(f"lig{tag:5s} {('%.2f'%g) if g else '  -':>9s} {('%.2f'%c) if c else '   -':>13s}")
    # synergy/competition from GPU dual best poses (model 1)
    ga=models(D/"gpu_ligA.pdbqt"); gb=models(D/"gpu_ligB.pdbqt")
    if ga and gb:
        il=interligand(ga[0],gb[0])
        if il:
            cl,close=il
            verdict=("COMPETITIVE (overlap/clash, same site)" if cl>3 else
                     "SYNERGY candidate (adjacent, in contact)" if close>=5 else
                     "INDEPENDENT (separate regions)")
            print(f"inter-ligand (GPU best poses): clashes<2A={cl}  contacts<4A={close}  -> {verdict}")

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "1HVY")
