
## manually link mkl RT library so we don't need to mess around with preloads
from ctypes import CDLL, RTLD_GLOBAL
CDLL('/opt/ohpc/pub/intel/mkl/lib/intel64/libmkl_rt.so', RTLD_GLOBAL)

## okay, now we can import
from ngsolve import *
import sys
from utils import *
from FlowTemplates import StokesTemplate

ngsglobals.msg_level = 1 if mpi_world.rank == 0 else 0

exec(fs_cmd)
    
disc_opts = { "order" : 2,
              "hodivfree" : False,
              "truecompile" : False,
              "RT" : False,
              "compress" : True,
              "divdivpen" : 0,
              "pq_reg" : 0 }

sol_opts = { "elint" : True,
             "pc_ver" : "block", # "block", "direct"
             "pc_opts" : {
                 "a_opts" : {
                     "type" : "auxh1", 
                     "amg_package" : "ngs_amg",
                     "mlt_smoother" : True,
                     "el_blocks" : False,
                     "amg_opts" : {
                         "ngs_amg_max_coarse_size" : 100,
                         "ngs_amg_sm_type" : "gs",
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

def doit():
    tsup = Timer("solve")
    tsup.Start()
    stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )
    tsup.Stop()

    ts = Timer("solve")
    ts.Start()
    nits = stokes.Solve(tol=1e-6, ms = 3000, solver=solver)
    ts.Stop()

    sys.stdout.flush()

    kappaA, kappaS, evsA, evsS = stokes.la.TestBlock()

    sys.stdout.flush()

    if mpi_world.rank == 0:
        print("\n---\ntime setup", tsup.time)
        print("A dofs", stokes.disc.X.ndofglobal)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (tsup.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (tsup.time * mpi_world.size) / 1000, "K" ) 
    if mpi_world.rank == 0:
        print("\n---\ntime solve", ts.time)
        print("Q dofs", stokes.disc.Q.ndofglobal)
        print("A dofs/proc", stokes.disc.X.ndofglobal / mpi_world.size / 1000, "K" ) 
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (ts.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (ts.time * mpi_world.size) / 1000, "K" ) 

    sys.stdout.flush()

    ## pickle results 
    def getnd(V):
        return (V.ndofglobal, mpi_world.Max(V.ndof), mpi_world.Min(V.ndof))
    import pickle
    RD = dict()
    RD["ND"] = { "X" : getnd(stokes.disc.X), "Xc" : [ getnd(stokes.disc.X.components[0]), getnd(stokes.disc.X.components[1]) ], "Q" : getnd(stokes.disc.Q) }
    RD["tsetup"] = tsup.time
    RD["tsolve"] = ts.time
    RD["ctimers"] = { x['name'] : x['time'] for x in Timers() }
    RD["disc_opts"] = disc_opts
    RD["sol_opts"] = sol_opts
    RD["kappaA"] = kappaA
    RD["kappaS"] = kappaS
    RD["evsA"] = evsA
    RD["evsS"] = evsS
    RD["NP"] = mpi_world.size
    RD["nel"] = mpi_world.Sum(flow_settings.mesh.ne)
    RD["nits"] = nits
    if mpi_world.size == 1 or mpi_world.rank == 1:
        pickle.dump(RD, open(pickle_file, "wb"))

        
if mpi_world.size == 1 or mpi_world.rank == 1:
    SetNumThreads(1)
    with TaskManager(pajetrace=100 * 1024 * 1024):
        doit()
else:
    doit()
    
# Draw(stokes.velocity, stokes.settings.mesh, "velocity")
