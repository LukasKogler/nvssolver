from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg
import sys

from FlowTemplates import *

ngsglobals.msg_level = 5


geo = g2d.SplineGeometry()
geo.AddRectangle((0, 0), (1, 1), bcs=("wall1", "wall2", "wall3", "wall4"))
mesh = Mesh(geo.GenerateMesh(maxh=0.1))
nu = 1e-3
zeta = x**2*(1-x)**2*y**2*(1-y)**2
u_ex = CoefficientFunction((zeta.Diff(y),-zeta.Diff(x)))
Draw(u_ex, mesh, "u_ex")
p_ex = x**5+y**5-1/3
f_1 = -nu * (u_ex[0].Diff(x).Diff(x) + u_ex[0].Diff(y).Diff(y)) + p_ex.Diff(x)
f_2 = -nu * (u_ex[1].Diff(x).Diff(x) + u_ex[1].Diff(y).Diff(y)) + p_ex.Diff(y)
vol_force = CoefficientFunction((f_1,f_2))
wall_slip = ""
wall_noslip = ".*"
inlet = ""
outlet = ""
uin = CoefficientFunction((0, 0))
    
flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = inlet, outlet = outlet, wall_slip = wall_slip,
                            wall_noslip = wall_noslip, uin = uin, symmetric = False, vol_force = vol_force)

order = 3
VO = 1
cf = (1, 0)

disc_opts = { "order" : order,
              "hodivfree" : False,
              "truecompile" : False, #mpi_world.size == 1,
              "RT" : False,
              "compress" : True,
              "pq_reg" : 1e-6 }

sol_opts = { "elint" : False,
             # "block_la" : False,
             "pc_ver" : "block",
             "pc_opts" : {
                 "a_opts" : {
                     "type" : "direct",
                     #"type" : "auxfacet",
                     # "type" : "auxh1", 
                     # "inv_type" : "sparsecholesky",
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
with TaskManager():#pajetrace = 50 * 2024 * 1024):
    stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )

    ts = Timer("solve")
    ts.Start()
    # stokes.Solve(tol=1e-6, ms = 300)
    ts.Stop()

    Apre = ngs.Preconditioner(stokes.la.a, "ngs_amg.stokes_gg_2d", **sol_opts["pc_opts"]["a_opts"]["amg_opts"])
    stokes.la.a.Assemble()

    evs_A = list(ngs.la.EigenValues_Preconditioner(mat=stokes.la.a.mat, pre=Apre, tol=1e-10))
    print("\n----")
    print("min ev. preA\A:", evs_A[0])
    print("max ev. preA\A:", evs_A[-1])
    print("cond-nr preA\A:", evs_A[-1]/evs_A[0])
    print("----")

    # stokes.la.TestBlock()

    if mpi_world.rank == 0:
        print("\n---\ntime solve", ts.time)
        print("A dofs/(sec * proc)", stokes.disc.X.ndof / (ts.time * mpi_world.size) ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndof + stokes.disc.Q.ndof) / (ts.time * mpi_world.size) ) 

        # stokes.la.TestBlock()

sys.stdout.flush()
