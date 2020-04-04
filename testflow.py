from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg
import sys

from utils import *
from FlowTemplates import StokesTemplate
# from FlowTemplates import *

ngsglobals.msg_level = 1

flow_settings = ST_2d(maxh=0.1, nu=1e-3)
flow_settings.mesh.Curve(3)
pqr = 0

# flow_settings = channel2d(maxh=0.1, nu=1e-3, L=5)
# pqr = 0

# flow_settings = ST_3d(maxh=0.1, nu=1e-2)
# flow_settings.mesh.Curve(3)
# pqr = 0

# flow_settings = vortex2d(maxh=0.2, nu=1)
# pqr = 1e-6

print("verts " , flow_settings.mesh.nv)
print("edges " , flow_settings.mesh.nedge)
print("facets " , flow_settings.mesh.nfacet)
print("els " , flow_settings.mesh.ne)

disc_opts = { "order" : 2,
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
                     "type" : "direct",
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
                         "ngs_amg_max_coarse_size" : 50,
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
    nits = stokes.Solve(tol=1e-6, ms = 300, solver = "minres")
    ts.Stop()

    if mpi_world.rank == 0:
        print("\n---\ntime setup", tsup.time)
        print("nits = ", nits)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (tsup.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (tsup.time * mpi_world.size) / 1000, "K" ) 
    if mpi_world.rank == 0:
        print("\n---\ntime solve", ts.time)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (ts.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (ts.time * mpi_world.size) / 1000, "K" ) 

    
    Draw(stokes.velocity, stokes.settings.mesh, "velocity")
    # stokes.la.TestBlock()

sys.stdout.flush()
# quit()
