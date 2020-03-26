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

    
def gen_sfile(sfile, pyfile, wdir, nnodes = 1, ntasks = 1, cpt = 1, npc = 1, nps = 8, spn = 2, NN=1, NP=1, job_name='default',
              ofile = None, use_numactl = True):
    f = open(sfile, 'w')
    f.write("#!/usr/bin/bash\n")
    f.write("#SBATCH -N 4\n")
    f.write("#SBATCH --ntasks-per-node=16\n")
    f.write("#SBATCH --sockets-per-node=2\n")
    f.write("#SBATCH --ntasks-per-socket=8\n")
    f.write("#SBATCH --ntasks-per-core=1\n")
    f.write("#SBATCH --cpus-per-task=1\n")
    f.write("#SBATCH --threads-per-core=2\n")
    f.write("#SBATCH --mem-bind=local\n")
    if ofile is not None:
        f.write("#SBATCH --output=" + ofile + "\n")
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
base_dir = "/home/lkogler/tests/stokes_solver/bench1/run0/"
os.makedirs(base_dir, exist_ok=True)

# Copy these files to every job folder
nvsdir = "/home/lkogler/src/nvssolver/"
dep_files = [nvsdir + "utils.py", nvsdir + "FlowTemplates.py", nvsdir + "bramblepasciak.py"]

# This line is dumped after parameters - change the problem here
fsc = "flow_settings = ST_2d(maxh=maxh, nref = nref)\\nflow_settings.mesh.Curve(3)"

# (ncpus, maxh, nref)
job_descriptions = [(1, 0.1, 0), (8, 0.05, 0), (64, 0.025, 0)]

job_cnt = 0
for nt, maxh, nref in job_descriptions:
    for bp in [True, False]:
        jobdir = base_dir + "job" + str(job_cnt) + "/"
        os.makedirs(jobdir, exist_ok=True)

        sfile    = jobdir + "job.slurm"
        pyfile   = jobdir + "run.py"
        pickfile = jobdir + "results.pickle"
        outfile  = jobdir + "run.out"

        for f in dep_files:
            shutil.copyfile(f, jobdir + os.path.basename(f))

        gen_pyfile(ofile = pyfile, tfile = "template.py", pickle_file = pickfile, fs_cmd = fsc, maxh = maxh, nref = nref)

        nnodes = nt//16 + (0 if nt%16 == 0 else 1)
        if nnodes < 1 or nnodes > 4:
            raise Exception("invalid nnodes = " + str(nnodes))

        gen_sfile(job_name = "ST3D_" + str(job_cnt), sfile = sfile, ofile = outfile, pyfile = pyfile, wdir = jobdir, nnodes = nnodes, ntasks = nt)
        sfiles.append(sfile)
        job_cnt += 1


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


