##
## Generates job files for 3d unit cube benchmark
##
import sys, os, subprocess, shutil
from script_generator_utils import *

# template file
template =  "/home/fs70654/lkogler/jobs/stokes2/template_st3d.py"

# where to put the generated files
run_num = 6
base_dir = "/gpfs/data/fs70654/lkogler/stokes2/st3d/run" + str(run_num) + "/"
os.makedirs(base_dir, exist_ok=True)

# Copy these files to every job folder
nvsdir = "/home/fs70654/lkogler/src/nvssolver/"
dep_files = [ nvsdir + x for x in ["FlowTemplates.py", "krylovspace_extension.py", "utils.py"] ]


if False:
    # maxh, nref, nv for generated st3d meshes
    if False:
        mdlines = [x for x in open("st3d_meshes.data", "r")]
        mdata = [ (float(x[0]), int(x[1]), int(x[4])) for x in (y.split() for y in mdlines) ]
        print(mdata)
        # quit()
    else:
        mdlines = [x for x in open("st3d_meshes.data", "r")]
        mdata = [ (float(x[0]), int(x[1]), int(x[4])) for x in (y.split() for y in mdlines) ]
    maxref = max(mdata, key = lambda x : x[1])
    mddict = {}
    for d in mdata:
        if d[0] in mddict:
            mddict[d[0]].append( (d[1], d[2]) )
        else:
            mddict[d[0]] = [ (d[1], d[2]) ]
    for k,v in mddict.items():
        v.sort(key = lambda x : x[0])
    # print(mdata)
    min_fl_epp = 100       # minimum # of elements per proc on finest level
    epp = 20000            # aim for this # of facets per proc
    hrnps = []
    max_nps = 25000
    # for k, D in mdata.values():
    for h in mddict:
        rfs = [ (x[0], 0, x[1]) for x in mddict[h] ]
        # print(h, rfs)
        nbase = len(rfs)
        lrf = rfs[-1]
        # print(h, rfs, "\n")
        for ref in range(nbase, 6):
            baseref = min(2, nbase-1) # do not take the very largest - loading might take a long time!
            extra_ref = ref-baseref
            rfs.append( (baseref, extra_ref, lrf[-1] * 8**extra_ref) )
        # print(h, rfs, "\n")
        for r, nref, nffr in rfs:
            nps = max(1, nffr//epp)
            frac = round(nffr/nps/epp, 4)
            if nps > 1:        # if NP>1, rank 0 does nothing
                nps = nps + 1
            nps = int(nps)
            # print(h, r, nref, nffr, nps)
            if nps <= max_nps:
                hrnps.append( (nps, h, r, nref, round(nffr/1e6,1), frac) )
    hrnps.sort(key = lambda x : x[0] + 1e-2*abs(1-x[-1]))
    currk = -1
    for k, x in enumerate(hrnps):
        # if x[0] > currk:
        currk = x[0]
        print(k, ":", x)
        # if k+1 < len(hrnps) and hrnps[k+1][2] > x[2]:
            # print("")
    # sel = [ 6, 8, 11, 14, 15, 19, 22, 23, 27, 29, 30, 31, 34, 35 ] # 30k
    sel = [  0, 6, 8, 10, 14, 15, 18, 20, 22, 23, 25, 28, 29, 30, 31, 34 ] # 20k
    job_descriptions = []        
    for j in sel:
        job_descriptions.append( (hrnps[j][1], hrnps[j][2], hrnps[j][3], hrnps[j][0]) )
    for k, x in enumerate(job_descriptions):
        print(k, ":", x)
    print(job_descriptions)    
    quit()

    
## maxh/nref/nref_add/ntasks
# 40k facets/proc
# job_descriptions = [(0.05, 0, 0, 1), (0.2, 1, 0, 3), (0.025, 0, 0, 10), (0.2, 2, 0, 18), (0.015, 0, 0, 41), (0.01, 0, 0, 103), (0.2, 3, 0, 143), (0.1, 3, 0, 204), (0.015, 1, 0, 360), (0.025, 1, 1, 662), (0.01, 1, 0, 916)]
# job_descriptions = [(0.05, 0, 0, 1), (0.1, 1, 0, 4), (0.2, 2, 0, 18), (0.01, 0, 0, 103), (0.1, 3, 0, 204), (0.025, 1, 1, 662), (0.2, 4, 0, 1139), (0.015, 1, 1, 2879), (0.025, 1, 2, 5291)]#, (0.01, 1, 1, 7325), (0.1, 2, 2, 13042)]
# 30k facets/proc
# job_descriptions = [(0.1, 1, 0, 5), (0.05, 1, 0, 17), (0.1, 2, 0, 35), (0.015, 0, 0, 54), (0.025, 1, 0, 111), (0.2, 3, 0, 191), (0.1, 3, 0, 272), (0.015, 1, 0, 480), (0.05, 3, 0, 1040), (0.075, 4, 0, 2698), (0.015, 2, 0, 3876), (0.025, 3, 0, 7097), (0.005, 1, 0, 9820)]
# job_descriptions = [(0.1, 1, 0, 5), (0.025, 0, 0, 13), (0.035, 1, 0, 28), (0.015, 0, 0, 54), (0.025, 1, 0, 111), (0.035, 2, 0, 224), (0.015, 1, 0, 480), (0.025, 2, 0, 889), (0.2, 4, 0, 1519), (0.075, 4, 0, 2698), (0.015, 2, 0, 3876), (0.025, 3, 0, 7097), (0.005, 1, 0, 9820), (0.1, 2, 2, 17389)]
# 20k facets/proc
job_descriptions = [(0.075, 0, 0, 1), (0.1, 1, 0, 7), (0.025, 0, 0, 19), (0.2, 2, 0, 36), (0.015, 0, 0, 81), (0.025, 1, 0, 166), (0.2, 3, 0, 286), (0.1, 3, 0, 408), (0.015, 1, 0, 720), (0.025, 2, 0, 1333), (0.005, 0, 0, 1642), (0.035, 3, 0, 2667), (0.075, 4, 0, 4046), (0.015, 2, 0, 5814), (0.025, 3, 0, 10645), (0.005, 1, 0, 14729)]

job_cnt = 0
sfiles = list()
for maxh, nref, nref_add, nt in job_descriptions:
    jobdir = base_dir + "job" + str(job_cnt) + "/"

    mesh_file = "/gpfs/data/fs70654/lkogler/stokes2/meshes/st3d/st3d_h" + str(maxh).replace(".", "_") + "_r" + str(nref) + ".vol"
    # cte = "mesh = gen_ref_mesh(geo=None, mesh_file=\\\"" + mesh_file + "\\\", nref=" + str(nref_add) + ", load=True, comm=mpi_world, maxh=" + str(maxh) + ")"
    params = { "mesh_file" : mesh_file,
               "nref" : nref_add,
               "maxh" : maxh }
    cte = ""
    job_name = str(job_cnt) + "r" + str(run_num) + "_st3"
    sfile = generate_job(jobdir=jobdir, job_name=job_name, cte=cte, py_params=params, template=template, nt=nt, dep_files=dep_files, maxtime="30:00")
    sfiles.append(sfile)
    job_cnt += 1


submit_files(sfiles, jf = ".running_st3d_jobs")
