import sys, os, subprocess, shutil, zipfile, pickle, argparse

def gather_results(basedir, zfile = "", jfiles = True, ofiles = True, htmls = True, traces = True, pfile = "", zbdir = None):
    if not os.path.isdir(basedir):
        raise Exception("directory " + basedir + " does not exist")

    dozip = len(zfile) > 0
    if dozip:
        zf = zipfile.ZipFile(zfile, "w", zipfile.ZIP_DEFLATED)
        if zbdir is None:
            zbdir = zfile.split("/")[-1][:-4]
    dopick = len(pfile) > 0
    if dopick:
        pick_data = dict()
    
    MAXSIZE = 5 * 1024 * 1024  # 5 MB
    subdirs = [ x for x in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, x)) ]
    for sd in subdirs:
        try:
            jobnum = int(sd[3:])
            path = os.path.join(basedir, sd)
            shortpath = os.path.join(zbdir,sd)
            for r, ds, fs in os.walk(path):
                for f in fs:
                    if os.path.getsize(os.path.join(r, f)) < MAXSIZE:
                        if dozip:
                            if f[-4:] == ".out" and ofiles:
                                zf.write(os.path.join(r, f), arcname=os.path.join(shortpath, f))
                            elif (f[-3:] == ".py" or f[-6:] == ".slurm") and jfiles:
                                zf.write(os.path.join(r, f), arcname=os.path.join(shortpath, f))
                            elif f[-6:] == ".trace" and traces:
                                zf.write(os.path.join(r, f), arcname=os.path.join(shortpath, f))
                            elif f[-5:] == ".html" and traces:
                                zf.write(os.path.join(r, f), arcname=os.path.join(shortpath, f))
                        if dopick and f == "results.pickle":
                            pick_data[jobnum] = pickle.load(open(os.path.join(r,f), "rb"))
                break # dont check __pycache__ etc
        except:
            pass

    if dopick:
        pickle.dump(pick_data, open(pfile, "wb"))
    if dozip:
        zf.write(pfile, arcname=os.path.join(zbdir, pfile))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir", type=str)
    parser.add_argument("-zfile", type=str, default="results.zip") #optional
    parser.add_argument("-pfile", type=str, default="results.pickle") #optional
    args = parser.parse_args()

    if args.zfile[-4:] != ".zip":
        args.zfile = args.zfile + ".zip"

    if args.pfile[-7:] != ".pickle":
        args.pfile = args.pfile + ".pickle"
    
    gather_results(basedir=args.basedir, zfile=args.zfile, pfile=args.pfile)
