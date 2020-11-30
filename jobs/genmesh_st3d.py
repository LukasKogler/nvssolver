import sys, os
import ngsolve as ngs

sys.path.append("/home/fs70654/lkogler/src/nvssolver/")

import sys, os
import ngsolve as ngs
import netgen as ng
import netgen.csg as csg
from utils import *

base_dir = "/gpfs/data/fs70654/lkogler/elamg/meshes/"
os.makedirs(base_dir, exist_ok=True)

mfd = open(os.path.join(base_dir, "mesh_data.txt"), "w")
mfd.close()

def gen_meshes(geo, maxhrefs, base_dir, name_mfile):
    os.makedirs(base_dir, exist_ok=True)
    ngs.ngsglobals.msg_level = 0
    with ngs.TaskManager():
        for maxh, maxref in maxhrefs:
            print("generate", maxh, 0, ", mesh_file =", os.path.join(base_dir, name_mfile(maxh, 0)))
            ngmesh = geo.GenerateMesh(maxh=maxh)
            ngs.ngsglobals.msg_level = 0
            ngmesh.Save(os.path.join(base_dir, name_mfile(maxh, 0)))
            mesh = ngs.Mesh(ngmesh)
            print(maxh, 0, mesh.nv, mesh.nedge, mesh.nface, mesh.ne)
            mfd = open(os.path.join(base_dir, "mesh_data.txt"), "a")
            mfd.write(str(maxh) + " " + str(0) + " " + str(mesh.nv) + " " + str(mesh.nedge) + " " + str(mesh.nface) + " " + str(mesh.ne) + "\n")
            mfd.close()
            for nref in range(1, maxref+1):
                print("generate", maxh, nref, ", mesh_file =", os.path.join(base_dir, name_mfile(maxh, nref)))
                ngs.ngsglobals.msg_level = 0
                ngmesh.Refine()
                ngs.ngsglobals.msg_level = 0
                ngmesh.Save(os.path.join(base_dir, name_mfile(maxh, nref)))
                mesh = ngs.Mesh(ngmesh)
                print(maxh, nref, mesh.nv, mesh.nedge, mesh.nface, mesh.ne)
                mfd = open(os.path.join(base_dir, "mesh_data.txt"), "a")
                mfd.write(str(maxh) + " " + str(nref) + " " + str(mesh.nv) + " " + str(mesh.nedge) + " " + str(mesh.nface) + " " + str(mesh.ne) + "\n")
                mfd.close()
                

base_dir = "/gpfs/data/fs70654/lkogler/stokes2/meshes/st3d"
H, W, L = 0.41, 0.41, 2.5
geo = geo_3dchannel(H=H, W=W, L=L, obstacle=True)
# maxhrefs = [ (0.2, 4), (0.1, 4), (0.05, 3), (0.025, 2), (0.015, 2), (0.01, 1), (0.005, 1) ]
maxhrefs = [ (0.075, 4), (0.05, 4), (0.035, 3), (0.025, 3), (0.015, 3), (0.01, 1), (0.005, 1) ]
gen_meshes(geo=geo, maxhrefs=maxhrefs, base_dir=base_dir, name_mfile = lambda h,r : "st3d_h" + str(h).replace(".", "_") + "_r" + str(r) + ".vol")
