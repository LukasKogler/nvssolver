
## manually link mkl RT library so we don't need to mess around with preloads
from ctypes import CDLL, RTLD_GLOBAL
CDLL('/opt/sw/vsc4/VSC/x86_64/glibc-2.17/skylake/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_rt.so', RTLD_GLOBAL)

## okay, now we can import
from ngsolve import *
import sys
from utils import *
from FlowTemplates import StokesTemplate

ngsglobals.msg_level = 1 if mpi_world.rank == 0 else 0

tmesh = Timer("solve")

mpi_world.Barrier()
tmesh.Start()
exec(__generated_code_to_execute)
H, W, L = 0.41, 0.41, 2.5
geo = geo_3dchannel(H=H, W=W, L=L, obstacle=True)
mesh = gen_ref_mesh(geo=geo, mesh_file=mesh_file, nref=nref, load=True, comm=mpi_world, maxh=maxh)
uin = CoefficientFunction( (2.25 * (2/H)**2 * (2/W)**2 * y * (H - y) * z * (W - z), 0, 0))
nu = 1e-3
flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "left", outlet = "right", wall_slip = "",
                            wall_noslip = "wall|obstacle", uin = uin, symmetric = False, vol_force = None)
ngsglobals.msg_level = 0 # curve message is printed by all ranks otherwise!
flow_settings.mesh.Curve(3)
ngsglobals.msg_level = 1
mpi_world.Barrier()
tmesh.Stop()

if mpi_world.rank == 0:
    print("\n---")
    print("time to load mesh:", tmesh.time)
    print("\n---")


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
                         "ngs_amg_max_coarse_size" : lambda V : min(1000, V.ndofglobal/1000),
                         "ngs_amg_min_coarse_size" : lambda V : min(200, V.ndofglobal/10000),
                         "ngs_amg_max_levels" : 100,
                         "ngs_amg_enable_multistep" : False,
                         "ngs_amg_ecw_geom" : False,
                         "ngs_amg_ecw_geom_spec" : [ 1, 1, 1 ], # 3 levels geometric, then harmonic
                         "ngs_amg_edge_thresh" : 0.03,
                         "ngs_amg_edge_thresh_spec" : [ 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04 ],
                         "ngs_amg_newsp" : True,
                         "ngs_amg_faflm" : True,
                         "sp_max_per_row" : 6,
                         "sp_max_per_row_spec" : [ 6 ],
                         "sp_max_per_row_classic" : 14,
                         # "sp_max_per_row_classic_spec" : [ 4, 4, 4, 6, 6, 8, 8, 8 ],
                         "ngs_amg_rd_min_nv_gl" : 1000,
                         "ngs_amg_rd_seq_nv" : 1000,
                         "ngs_amg_sm_type" : "gs",
                         # "ngs_amg_sm_type_spec" : ["gs", "gs"],
                         "ngs_amg_sm_mpi_overlap" : True,
                         "ngs_amg_sm_mpi_thread" : True,
                         # "ngs_amg_crs_alg" : "spw",
                         # "ngs_amg_spw_rounds" : 3,
                         # "ngs_amg_spw_rounds_spec" : [ 4, 4 ],
                         # "ngs_amg_spw_wo" : True,
                         # "ngs_amg_spw_pcwt" : "mmx", # "harm", "geom", "mmx"
                         # "ngs_amg_spw_pmmas" : "geom", # "min", "geom", "harm", "alg", "max"
                         # "ngs_amg_spw_cbs" : False,
                         "ngs_amg_log_level" : "extra",
                         "ngs_amg_log_level_pc" : "extra",
                         "ngs_amg_do_test" : True,
                         "ngs_amg_print_log" : True
                     }
                 }
             }
}

def doit():
    tsup = Timer("solve")
    mpi_world.Barrier()
    tsup.Start()
    stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts)
    mpi_world.Barrier()
    tsup.Stop()

    eff_size = mpi_world.size - 1 if mpi_world.size > 1 else 1

    if mpi_world.rank == 0:
        print("\n---")
        print(" A dofs", stokes.disc.X.ndofglobal / 1000, "K" ) 
        print(" A dofs/proc", stokes.disc.X.ndofglobal / eff_size / 1000, "K" ) 
        print(" Q dofs", stokes.disc.Q.ndofglobal / 1000, "K" ) 
        print(" Q dofs/proc", stokes.disc.Q.ndofglobal / eff_size / 1000, "K" )
        print(" A + Q dofs", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / 1000, "K" ) 
        print(" A + Q dofs / proc", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / eff_size / 1000, "K" ) 
        print("---\ntime setup", tsup.time)
        print(" A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (tsup.time * eff_size) / 1000, "K" ) 
        print(" A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (tsup.time * eff_size) / 1000, "K" )
        print("---\n")

    kappas = True
    if kappas:
        tkappa = Timer("solve")
        mpi_world.Barrier()
        tkappa.Start()
        kappaA, kappaS, evsA, evsS = stokes.la.TestBlock()
        mpi_world.Barrier()
        tkappa.Stop()

        if mpi_world.rank == 0:
            print("\n---")
            print("time for evs:", tkappa.time)
            print("\n---")

    sys.stdout.flush()

    ts = Timer("solve")
    mpi_world.Barrier()
    ts.Start()
    nits = stokes.Solve(tol = 1e-6, ms = 1000, rel_err = True, solver = "gmres", use_sz = True)
    mpi_world.Barrier()
    ts.Stop()

    sys.stdout.flush()

    if mpi_world.rank == 0:
        print("\n---")
        print("time to load mesh:", tmesh.time)
        if kappas:
            print("time for evs:", tkappa.time)
        print("\n---")
        print(" A dofs", stokes.disc.X.ndofglobal / 1000, "K" ) 
        print(" A dofs/proc", stokes.disc.X.ndofglobal / eff_size / 1000, "K" ) 
        print(" Q dofs", stokes.disc.Q.ndofglobal / 1000, "K" ) 
        print(" Q dofs/proc", stokes.disc.Q.ndofglobal / eff_size / 1000, "K" )
        print(" A + Q dofs", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / 1000, "K" ) 
        print(" A + Q dofs / proc", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / eff_size / 1000, "K" ) 
        print("---\ntime steup", tsup.time)
        print(" A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (tsup.time * eff_size) / 1000, "K" ) 
        print(" A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (tsup.time * eff_size) / 1000, "K" ) 
        print("\n---\ntime solve", ts.time)
        print(" A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (ts.time * eff_size) / 1000, "K" ) 
        print(" A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (ts.time * eff_size) / 1000, "K" ) 

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
    if kappas:
        RD["kappaA"] = kappaA
        RD["kappaS"] = kappaS
        RD["evsA"] = evsA
        RD["evsS"] = evsS
    RD["NP"] = mpi_world.size
    RD["nel"] = mpi_world.Sum(flow_settings.mesh.ne)
    RD["nits"] = nits
    if mpi_world.size == 1 or mpi_world.rank == 1:
        pickle.dump(RD, open(pickle_file, "wb"))

        
doit()

# if mpi_world.size == 1 or mpi_world.rank == 1:
#     SetNumThreads(1)
#     with TaskManager(pajetrace=100 * 1024 * 1024):
#         doit()
# else:
#     doit()
    
# Draw(stokes.velocity, stokes.settings.mesh, "velocity")
