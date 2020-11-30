##
## Generates job files for 3d unit cube benchmark
##
import sys, os, subprocess, shutil
from script_generator_utils import *

# template file, etc
template =  "/home/fs70654/lkogler/jobs/stokes2/template_rev_plane.py"
mesh_data_file = "rev_big_plane_meshes.data" # "rev_plane_meshes.data"
base_mesh_name = "rev_big_plane" # "rev_plane"

# where to put the generated files
run_num = 4
base_dir = "/gpfs/data/fs70654/lkogler/stokes2/plane/run" + str(run_num) + "/"
os.makedirs(base_dir, exist_ok=True)

# Copy these files to every job folder
nvsdir = "/home/fs70654/lkogler/src/nvssolver/"
dep_files = [ nvsdir + x for x in ["FlowTemplates.py", "krylovspace_extension.py", "utils.py"] ]


if False:
    # maxh, nref, nv for generated plane meshes
    mdlines = [x for x in open(mesh_data_file, "r")]
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
    epp = 30000            # aim for this # of facets per proc
    hrnps = []
    max_nps = 20000
    # for k, D in mdata.values():
    for h in mddict:
        rfs = [ (x[0], 0, x[1]) for x in mddict[h] ]
        # print(h, rfs)
        nbase = len(rfs)
        lrf = rfs[-1]
        # print(h, rfs, "\n")
        for ref in range(nbase, 5):
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
    sel = [ 0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12]
    job_descriptions = [ ]
    for j in sel:
        job_descriptions.append( (hrnps[j][1], hrnps[j][2], hrnps[j][3], hrnps[j][0]) )
    for k, x in enumerate(job_descriptions):
        print(k, ":", x)
    print(job_descriptions)    
    quit()

    
## maxh/nref/nref_add/ntasks
# 20k facets/proc, big box (rX.vol meshes avaiable)
# job_descriptions = [(5.0, 0, 0, 95), (0.2, 0, 0, 198), (5.0, 1, 0, 838), (0.1, 0, 0, 1054), (0.2, 1, 0, 1764), (5.0, 1, 1, 6702), (0.1, 1, 0, 9450), (0.2, 1, 1, 14106)]
# 20k facets/proc, big box (more rX.vol meshes avaiable)
# job_descriptions = [(5.0, 0, 0, 95), (0.25, 0, 0, 148), (0.15, 0, 0, 379), (5.0, 1, 0, 838), (0.25, 1, 0, 1313), (0.15, 1, 0, 3391), (5.0, 2, 0, 6660), (0.25, 2, 0, 10442), (0.2, 1, 1, 14106)]

# 30k facets/proc, big box (rX.vol meshes avaiable)
# job_descriptions = [(5.0, 0, 0, 63), (0.25, 0, 0, 99), (5.0, 1, 0, 559), (0.25, 1, 0, 876), (0.2, 1, 0, 1176), (5.0, 1, 1, 4468), (0.1, 1, 0, 6300), (0.2, 1, 1, 9404)]
# 30k facets/proc, big box (more rX.vol meshes avaiable)
job_descriptions = [(5.0, 0, 0, 63), (0.25, 0, 0, 99), (0.2, 0, 0, 132), (0.15, 0, 0, 253), (5.0, 1, 0, 559), (0.25, 1, 0, 876), (0.2, 1, 0, 1176), (0.15, 1, 0, 2261), (5.0, 2, 0, 4440), (0.25, 2, 0, 6961), (0.2, 1, 1, 9404)]


job_cnt = 0
sfiles = list()
for maxh, nref, nref_add, nt in job_descriptions:
    jobdir = base_dir + "job" + str(job_cnt) + "/"

    mesh_file = "/gpfs/data/fs70654/lkogler/stokes2/meshes/plane/" + base_mesh_name + "_h" + str(maxh).replace(".", "_") + "_r" + str(nref) + ".vol"
    # cte = "mesh = gen_ref_mesh(geo=None, mesh_file=\\\"" + mesh_file + "\\\", nref=" + str(nref_add) + ", load=True, comm=mpi_world, maxh=" + str(maxh) + ")"
    params = { "mesh_file" : mesh_file,
               "nref" : nref_add,
               "pickle_file_sol" : os.path.join(jobdir, "stokes_sol.pickle"),
               "maxh" : maxh }
    cte = ""
    job_name = str(job_cnt) + "r" + str(run_num) + "_plane"
    sfile = generate_job(jobdir=jobdir, job_name=job_name, cte=cte, py_params=params, template=template, nt=nt, dep_files=dep_files)
    sfiles.append(sfile)
    job_cnt += 1


submit_files(sfiles, jf = ".running_plane_jobs")
