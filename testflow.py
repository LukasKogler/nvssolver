from ctypes import CDLL, RTLD_GLOBAL
CDLL("/opt/intel/mkl/lib/intel64/libmkl_rt.so", RTLD_GLOBAL)

from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg
import sys

import pickle

from utils import *
from FlowTemplates import StokesTemplate
# from FlowTemplates import *


ngsglobals.msg_level = 1
ngsglobals.testout = "test.out"


# flow_settings = channel2d(maxh=0.1, nu=1e-3, L=1)
# pqr = 0

#flow_settings = ST_2d(maxh=0.1, nu=1e-3)
#flow_settings.mesh.Curve(3)
#pqr = 0

#flow_settings = channel2d(maxh=0.05, nu=1e-3, L=1, nref=1)
#pqr = 0

#flow_settings = channel2d(maxh=0.2, nu=1e-3, L=40)
pqr = 0

# flow_settings = ST_3d(maxh=0.1, nu=1e-0, symmetric=True)
flow_settings = ST_2d(maxh=0.04, nu=1e-0, symmetric=False)
# flow_settings.mesh.Curve(3)
# print(mpi_world.rank, "csg", flow_settings.mesh.GetCurveOrder())

# import netgen.occ
# occ_geo = netgen.occ.OCCGeometry("geometry/cylinder3D.step")
# mesh = gen_ref_mesh(geo=occ_geo, maxh=100, comm=mpi_world)
# mesh.ngmesh.SetGeometry(occ_geo)
# mesh.Curve(3)
# pqr = 0
# print(mpi_world.rank, "OCC", mesh.GetCurveOrder())

# quit()

# flow_settings = vortex2d(maxh=0.025, nu=1)
# pqr = 1e-6

if mpi_world.size == 1:
    print("verts " , flow_settings.mesh.nv)
    print("edges " , flow_settings.mesh.nedge)
    print("facets " , flow_settings.mesh.nfacet)
    print("els " , flow_settings.mesh.ne)

order = 2
disc_opts = { "order" : order,
              "hodivfree" : False,
              "truecompile" : False, #mpi_world.size == 1,
              "RT" : False,
              "compress" : True,
              #"divdivpen" : 0,
              "trace_sigma" : False,
              "divdivpen" : 0,
              "pq_reg" : pqr }

sol_opts = { "elint" : True,
             # "block_la" : False,
             "pc_ver" : "block", # "block", "direct"
             "pc_opts" : {
                 "a_opts" : {
                     #"type" : "direct",
                     # "inv_type" : "mumps",
                     # "type" : "auxfacet",
                     "type" : "auxh1", 
                     # "amg_package" : "petsc",
                     "amg_package" : "ngs_amg",
                     # "amg_package" : "direct", # direct solve in auxiliary space
                     "mlt_smoother" : True,
                     "el_blocks" : False,
                     # "type" : "stokesamg", 
                     "amg_opts" : {
                         "ngs_amg_max_coarse_size" : 2,
                         # "ngs_amg_max_levels" : 2,
                         "ngs_amg_sm_type" : "gs",
                         "ngs_amg_keep_grid_maps" : True,
                         "ngs_amg_n_levels_d2_agg" : 1,
                         "ngs_amg_ecw_geom" : True,
                         "ngs_amg_enable_sp" : True,
                         "ngs_amg_sp_max_per_row" : 4,
                         "ngs_amg_ecw_robust" : False,
                         "ngs_amg_enable_multistep" : False,
                         "ngs_amg_log_level" : "none",
                         "ngs_amg_print_log" : False,
                         "ngs_amg_do_test" : False,
                         }
                 }
             }
}

# SetNumThreads(1)
with TaskManager():#pajetrace = 50 * 2024 * 1024):

    tsup = Timer("solve")
    tsup.Start()
    stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )
    tsup.Stop()

    X = stokes.disc.X
    Q = stokes.disc.X
    print("X0 ndof", sum(X.components[0].FreeDofs(True)))
    print("X1 ndof", sum(X.components[1].FreeDofs(True)))
    print("X  ndof", sum(X.FreeDofs(True)))

    #if sol_opts["pc_ver"] == "block":
    stokes.la.TestBlock()

    quit()

    ts = Timer("solve")
    ts.Start()
    nits = stokes.Solve(tol=1e-6, ms = 500, solver = "gmres", use_sz = True, rel_err = False)
    ts.Stop()
    #input()
    if mpi_world.rank == 0:
        print("\n---\ntime setup", tsup.time)
        print("nits = ", nits)
        print("A dofs  ", stokes.disc.X.ndofglobal)
        print("Q dofs  ", stokes.disc.Q.ndofglobal)
        print("---")
        print("A+Q dofs", stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (tsup.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (tsup.time * mpi_world.size) / 1000, "K" ) 
        print("---\ntime solve", ts.time)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (ts.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (ts.time * mpi_world.size) / 1000, "K" ) 
        print("---\n")
    
    mesh = stokes.disc.X.mesh    
    V_vis_u = HDiv(mesh, order=order)
    u_vis = GridFunction(V_vis_u)
    u_vis.Set(stokes.velocity)
    V_vis_p = L2(mesh, order=order-1)
    p_vis = GridFunction(V_vis_p)
    p_vis.Set(stokes.pressure)
    import pickle
    netgen.meshing.SetParallelPickling(True)
    pickle_file_sol = "stokes_sol_st3d.pickle"
    my_pfs = pickle_file_sol.replace(".pickle", "_" + str(mpi_world.rank) + ".pickle")
    pickle.dump((u_vis, p_vis), open(my_pfs, "wb"))
    mpi_world.Barrier()
    if mpi_world.rank==0:
        solu, solp = pickle.load(open(my_pfs, "rb"))
        vtk = VTKOutput(ma=solu.space.mesh, coefs=[solu, solp], names=["vel", "pressure"], filename="sol_st3d", subdivision=1)
        vtk.Do()
    mpi_world.Barrier()

    '''
    sys.stdout.flush()

    sol_opts["elint"] = False
    disc_opts["divdivpen"] = 0
    sol_opts["pc_ver"] = "direct"
    # sol_opts["elint"] = False
    stokesex = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts)
    stokesex.Solve(tol=1e-6, ms = 500, solver = "apply_pc")

    err = GridFunction(stokes.disc.V)
    err.vec.data = stokes.velocity.vec - stokesex.velocity.vec
    norm_err = Norm(err.vec)
    norm_errl2 = abs(Integrate((stokes.velocity - stokesex.velocity)**2, flow_settings.mesh))**0.5
    print("norm err l2 = ", norm_err)
    print("norm err L2 = ", norm_errl2)

    import matplotlib.pyplot as plt
    plt.semilogy(list(k for k,x in enumerate(stokes.solver.errors)), stokes.solver.errors)
    plt.show()
    '''

    Vvel = HDiv(mesh = stokes.settings.mesh, order = disc_opts["order"])
    myvel = GridFunction(Vvel)
    myvel.vec.data = stokes.velocity.vec
    Draw(stokes.velocity, stokes.settings.mesh, "velocity")

    # stokes.la.TestBlock()

    #
    # quit()

    
    picklefile = open("myout.dat", "wb")
    pickle.dump(myvel, picklefile)
    picklefile.close()
    #print(stokes.velocity)


print("finished")
