import sys, os, subprocess, shutil, zipfile, pickle, argparse
from gather_results import gather_results

bdir = "/gpfs/data/fs70654/lkogler/stokes2"

jtrs = { "st3d" : [-1],
         "plane" : [-1] }

for jt, whichruns in jtrs.items():
    jt_path = os.path.join(bdir, jt)
    if len(whichruns)==1 and whichruns[0]==-1:
        runs = []
        for sth in os.listdir(jt_path):
            if sth[:3] == "run":
                try:
                    rnum = int(sth[3:])
                    runs.append(rnum)
                except:
                    pass
    else:
        runs = whichruns
    print("runs = ", runs)
    for run in runs:
        path = os.path.join(jt_path, "run"+str(run)+"/")
        zfile = "results/" + jt + "_r" + str(run) + ".zip"
        pfile = "results/" + jt + "_r" + str(run) + ".pickle"
        print("start gather job ", jt, "run", run, "   dir = ", path, ", zfile =", zfile, ", pfile =", pfile)
        gather_results(basedir=path, zfile=zfile, pfile=pfile)
        print("done  gather job ", jt, "run", run, "   dir = ", path, ", zfile =", zfile, ", pfile =", pfile)
