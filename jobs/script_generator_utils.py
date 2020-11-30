import sys, os, subprocess, shutil

## Add some code at the top of a python file and write it to destination
def gen_pyfile(ofile, tfile, cte = None, **params):
    tf = open(tfile, 'r')
    pyf = open(ofile, 'w')
    pyf.write("# --- GENERATED PART --- \n")
    pyf.write("# ----- Parameters --- \n")
    for key, val in params.items():
        if type(val) == str:
            pyf.write(key + " = \"" + val + "\"\n")
        else:
            pyf.write(key + " = " + str(val) + "\n")
    pyf.write("# ----- Code to exec --- \n")
    if cte is not None:
        if type(cte) == list:
            allcte = ""
            for line in cte:
                allcte += line + "\n"
        elif type(cte) == str:
            allcte = cte
        else:
            raise("I do not know what to do with cte = ", cte, "!")
        pyf.write("__generated_code_to_execute = \"" + allcte + "\"\n")
    pyf.write("# --- GENERATED PART END --- \n\n\n")
    pyf.write(tf.read())
    pyf.close()

## Generate a slurm file that executes "pyfile" in working directory "wdir".
## Parameters are set to default for VSC4.
def gen_sfile(sfile, pyfile, wdir, nnodes = 1, ntasks = 1, cpt = 1, tpc = 1, tps = 24, spn = 2, tpn = 48, NN=1, NP=1, job_name='default',
              ofile = None, use_numactl = True, thrpc = 1, maxtime = "60:00", qos = "mem_0096"):
    f = open(sfile, 'w')
    f.write("#!/usr/bin/bash\n")
    wopt = lambda key, val : f.write("#SBATCH --" + key + "=" + str(val) + "\n")
    wopt("ntasks", ntasks)
    f.write("#SBATCH -N " + str(nnodes) + "\n")
    wopt("job-name", job_name)
    # wopt("ntasks-per-node", tpn) # this is not a MAX, so it clashes with ntasks (but does no harm bc ntasks is prioretized)
    wopt("sockets-per-node", spn)
    wopt("ntasks-per-socket", tps) # this is a MAX
    wopt("ntasks-per-core", tpc)
    wopt("cpus-per-task", cpt)
    wopt("threads-per-core", thrpc)
    wopt("mem-bind", "local")
    wopt("qos", qos)
    wopt("time", maxtime)
    if ofile is not None:
        wopt("output", ofile)
    f.write("\n\n\n")
    f.write('cd ' + wdir + '\n\n')
    launch_cmd = "srun "
    python_cmd = "/opt/sw/spack-0.12.1/opt/spack/linux-centos7-x86_64/gcc-9.1.0/python-3.7.4-jlu2hiezkhy3lblskt232asjwyxjx2fx/bin/python3 "
    numa_cmd = "/opt/sw/spack-0.12.1/opt/spack/linux-centos7-x86_64/gcc-9.1.0/numactl-2.0.12-vqkmtbijxd365wv5p57d775p34bne3ov/bin/numactl --localalloc "
    if use_numactl:
        launch_cmd = launch_cmd + numa_cmd
    f.write(launch_cmd + python_cmd + pyfile + "\n")
    f.close()
    


def generate_job(jobdir, job_name, template, py_params = dict(), cte = "", dep_files = [], nt = 1, maxtime = "60:00"):
    os.makedirs(jobdir, exist_ok=True)

    sfile    = jobdir + "job.slurm"
    pyfile   = jobdir + "run.py"
    pickfile = jobdir + "results.pickle"
    outfile  = jobdir + "run.out"

    for f in dep_files:
        shutil.copyfile(f, jobdir + os.path.basename(f))

    gen_pyfile(ofile = pyfile, tfile = template, pickle_file = pickfile, cte = cte, **py_params)

    nnodes = nt//48 + (0 if nt%48 == 0 else 1)
    if nnodes < 1:
        raise Exception("invalid nnodes = " + str(nnodes))

    gen_sfile(job_name = job_name, sfile = sfile, ofile = outfile, pyfile = pyfile, wdir = jobdir, nnodes = nnodes, ntasks = nt,
              tpn=48, spn=2, tps=24, tpc=1, cpt=2, thrpc=2, maxtime=maxtime)

    return sfile


def submit_files(sfiles, jf = ".running_jobs"):
    sub = input("Submit?")

    if sub in ["Y", 1, "y", "hell_yeah"]:
        jf = open(jf, 'w')
        for f in sfiles:
            proc = subprocess.Popen(['/opt/vsc4/slurm/18-08-7-1/bin/sbatch', f], stdout=subprocess.PIPE)
            line = str(proc.stdout.readline())
            if line.startswith("b'Submitted batch job "):
                rest = line[22:-3]
                j_id = int(rest)
                print("Submitted batch job: ", j_id)
                jf.write(rest+' ')
            else:
                print('could not start job for py-file ', f)
        jf.close()

