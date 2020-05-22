import sys, os, subprocess, shutil


def gen_pyfile(ofile, tfile, **params):
    tf = open(tfile, 'r')
    pyf = open(ofile, 'w')
    pyf.write("# --- GENERATED PART --- \n")
    for key, val in params.items():
        if type(val) == str:
            pyf.write(key + " = \"" + val + "\"\n")
        else:
            pyf.write(key + " = " + str(val) + "\n")
    # pyf.write(fs_cmd + "\n")
    pyf.write("# --- GENERATED PART END --- \n\n\n")
    pyf.write(tf.read())
    pyf.close()

    
def gen_sfile(sfile, pyfile, wdir, nnodes = 1, ntasks = 1, cpt = 1, npc = 1, nps = 8, spn = 2, tpn = 16, NN=1, NP=1, job_name='default',
              ofile = None, use_numactl = True, tpc = 1):
    f = open(sfile, 'w')
    f.write("#!/usr/bin/bash\n")
    wopt = lambda key, val : f.write("#SBATCH --" + key + "=" + str(val) + "\n")
    wopt("ntasks", ntasks)
    f.write("#SBATCH -N " + str(nnodes) + "\n")
    wopt("job-name", job_name)
    wopt("ntasks-per-node", tpn)
    wopt("sockets-per-node", spn)
    wopt("ntasks-per-socket", nps)
    wopt("ntasks-per-core", npc)
    wopt("cpus-per-task", cpt)
    wopt("threads-per-core", tpc)
    wopt("mem-bind", "local")
    if ofile is not None:
        wopt("output", ofile)
    f.write("\n\n\n")
    f.write('cd ' + wdir + '\n\n')
    launch_cmd = "prun "
    python_cmd = "/opt/ohpc/pub/spack/linux-centos7-x86_64/gcc-8.2.0/python-3.6.3-3id5fmf3fhwxnax6bzkkvds2bxg6qnv2/bin/python3 "
    numa_cmd = "/home/lkogler/local/numa/bin/numactl --localalloc "
    if use_numactl:
        launch_cmd = launch_cmd + numa_cmd
    f.write(launch_cmd + python_cmd + pyfile + "\n")
    f.close()


sfiles = []

#
base_dir = "/home/lkogler/tests/stokes_solver/bench3/run1/"
os.makedirs(base_dir, exist_ok=True)

# Copy these files to every job folder
nvsdir = "/home/lkogler/src/nvssolver/"
dep_files = [nvsdir + "utils.py", nvsdir + "FlowTemplates.py", nvsdir + "krylovspace_extension.py"]

# This line is dumped after parameters - change the problem here
# fsc = "flow_settings = ST_3d(maxh=maxh, nref = nref)\\nflow_settings.mesh.Curve(3)"
fscT = "flow_settings = ST_3d(maxh=maxh, nref = nref)\\nflow_settings.mesh.Curve(3)"
fscF = "flow_settings = ST_3d(maxh=maxh, nref = nref, obstacle = False)"

## ST_3d mesh : maxh/nref/Kfacets for generated meshes ... 
# hrfs = [(0.05,  0, 55),   (0.05,  1, 434), 
#         (0.04,  0, 76),   (0.04,  1, 596), 
#         (0.035, 0, 74),   (0.035, 1, 740), 
#         (0.03,  0, 221),  (0.03,  1, 1750), 
#         (0.025, 0, 371),  (0.025, 1, 2900), 
#         (0.02,  0, 514),  (0.02,  1, 4000), 
#         (0.015, 0, 1600), (0.015, 1, 128000), 
#         (0.01,  0, 4008), (0.01,  1, 32064)] # <- this one is guessed 
# hrnps = []
# for h, r, nf in hrfs:
#     nps = int(round(nf/62.5))
#     rfrac = nf/63.5
#     if nps >= 1 and nps <=64:
#         print((h, r, int(nps)))
#         hrnps.append( (h, r, nps, rfrac) )
# hrnps.sort(key = lambda x : x[2])
# for x in hrnps:
#     print(x)
# quit()

# (maxh, nref, ntasks)
job_descriptions = [(0.04, 0, 1), (0.025, 0, 7), (0.02, 0, 9), (0.04, 1, 10), (0.035, 1, 13), (0.015, 0, 27), (0.03, 1, 29), (0.025, 1, 47), (0.02, 1, 64)]

job_cnt = 0
for obstacle, fsc in [(True, fscT)]:#, (False, fscF)]:
    for solver in ["gmres"]:
        jc2 = 0
        for maxh, nref, nt in job_descriptions:
            obsn = "T" if obstacle else "F"
            jobdir = base_dir + "ST3D_" + obsn + "_" + solver + "/job" + str(jc2) + "/"
            os.makedirs(jobdir, exist_ok=True)

            sfile    = jobdir + "job.slurm"
            pyfile   = jobdir + "run.py"
            pickfile = jobdir + "results.pickle"
            outfile  = jobdir + "run.out"

            for f in dep_files:
                shutil.copyfile(f, jobdir + os.path.basename(f))

            gen_pyfile(ofile = pyfile, tfile = "template.py", pickle_file = pickfile, fs_cmd = fsc, maxh = maxh, nref = nref, solver = solver)

            nnodes = nt//16 + (0 if nt%16 == 0 else 1)
            if nnodes < 1 or nnodes > 4:
                raise Exception("invalid nnodes = " + str(nnodes))

            gen_sfile(job_name = "ST3D_" + str(job_cnt), sfile = sfile, ofile = outfile, pyfile = pyfile, wdir = jobdir, nnodes = nnodes, ntasks = nt)
            sfiles.append(sfile)
            job_cnt += 1
            jc2 += 1


sub = input("Submit?")

if sub in ["Y", 1, "y", "hell_yeah"]:
    jf = open('.running_jobs', 'w')
    for f in sfiles:
        proc = subprocess.Popen(['/usr/bin/sbatch', f], stdout=subprocess.PIPE)
        line = str(proc.stdout.readline())
        if line.startswith("b'Submitted batch job "):
            rest = line[22:-3]
            j_id = int(rest)
            print("Submitted batch job: ", j_id)
            jf.write(rest+' ')
        else:
            print('could not start job for py-file ', f)
    jf.close()


