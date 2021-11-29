import pickle
from ctypes import CDLL, RTLD_GLOBAL
CDLL('/opt/ohpc/pub/intel/mkl/lib/intel64/libmkl_rt.so', RTLD_GLOBAL)

from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg
import sys

import pickle

from utils import *
from FlowTemplates import StokesTemplate
# from FlowTemplates import *

ngsglobals.msg_level = 1


# flow_settings = channel2d(maxh=0.1, nu=1e-3, L=1)
# pqr = 0

#flow_settings = ST_2d(maxh=0.1, nu=1e-3)
#flow_settings.mesh.Curve(3)
#pqr = 0

#flow_settings = channel2d(maxh=0.05, nu=1e-3, L=1, nref=1)
#pqr = 0

#flow_settings = channel2d(maxh=0.2, nu=1e-3, L=40)
#pqr = 0

flow_settings = ST_3d(maxh=0.1, nu=1e-2, symmetric=True)
pqr = 0

# flow_settings = vortex2d(maxh=0.025, nu=1)
# pqr = 1e-6

print("verts " , flow_settings.mesh.nv)
print("edges " , flow_settings.mesh.nedge)
print("facets " , flow_settings.mesh.nfacet)
print("els " , flow_settings.mesh.ne)

disc_opts = { "order" : 1,
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
                     # "amg_package" : "ngs_amg",
                     "amg_package" : "direct", # direct solve in auxiliary space
                     "blk_smoother" : True,
                     "mlt_smoother" : True,
                     # "sm_el_blocks" : False,
                     "sm_nsteps" : 2,
                     "mpi_thread" : True,
                     "mpi_overlap" : True,
                     "shm" : mpi_world.size==1,
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
                         "ngs_amg_log_level" : "extra",
                         "ngs_amg_print_log" : True,
                         "ngs_amg_do_test" : True,
                         }
                 }
             }
}

# SetNumThreads(1)
with TaskManager(pajetrace = 50 * 2024 * 1024):

    tsup = Timer("solve")
    tsup.Start()
    stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )
    tsup.Stop()

    X = stokes.disc.X
    Q = stokes.disc.X
    print("X0 ndof", sum(X.components[0].FreeDofs(True)))
    print("X1 ndof", sum(X.components[1].FreeDofs(True)))
    print("X  ndof", sum(X.FreeDofs(True)))

    if sol_opts["pc_ver"] == "block":
       stokes.la.TestBlock()

    ts = Timer("solve")
    ts.Start()
    nits = stokes.Solve(tol=1e-12, ms = 500, solver = "gmres", use_sz = True, rel_err = False)
    ts.Stop()
    #input()
    '''
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
