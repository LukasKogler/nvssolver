from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg

from FlowTemplates import *

ngsglobals.msg_level = 1

case = 3

if case == 1:
    geo = g2d.unit_square
    mesh = Mesh(geo.GenerateMesh(maxh=1))
    inlet = "left"
    outlet = "right"
    wall_slip = ""
    wall_noslip = "top|bottom"
    uin = CoefficientFunction((y * (1-y), 0))
    nu = 1
    vol_force = CoefficientFunction( (0,0) )
elif case == 2:
    geo = g2d.SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bcs=("wall", "wall", "wall", "wall"))
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
elif case == 3:
    geo = g2d.SplineGeometry()
    geo.AddRectangle((0, 0), (2,0.41), bcs=("wall", "outlet", "wall", "inlet"))
    geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    ngm = geo.GenerateMesh(maxh=0.1)
    for l in range(3): # TODO: facet aux with refined mesh!!
        # raise "facet_aux with refined mesh does not work yet!! (check fine_facets ... )"
        ngm.Refine()
    mesh = Mesh(ngm)
    mesh.Curve(3)
    inlet = "inlet"
    outlet = "outlet"
    wall_noslip = "wall|cyl"
    wall_slip = ""
    # vol_force = CoefficientFunction( (0,0) )
    vol_force = None
    nu = 1e-3
    uin = CoefficientFunction( (4 * (2/0.41)**2 * y * (0.41 - y), 0))
    
flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = inlet, outlet = outlet, wall_slip = wall_slip,
                            wall_noslip = wall_noslip, uin = uin, symmetric = True, vol_force = vol_force)

disc_opts = { "order" : 3,
              "hodivfree" : False,
              "truecompile" : mpi_world.size == 1,
              "RT" : False,
              "compress" : True,
              "pq_reg" : 1e-6 if case == 2 else 0 }

sol_opts = { "elint" : False,
             # "block_la" : False,
             "pc_ver" : "block",
             "pc_opts" : {
                 "a_opts" : {
                     "type" : "direct",
                     "inv_type" : "umfpack",
                     "amg_opts" : {
                         "ngs_amg_max_coarse_size" : 50,
                         "ngs_amg_sm_type" : "bgs",
                         "ngs_amg_keep_grid_maps" : True,
                         "ngs_amg_ecw_geom" : True,
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
    stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )

    ts = Timer("solve")
    ts.Start()
    stokes.Solve(tol=1e-6, ms = 300)
    ts.Stop()

    if mpi_world.rank == 0:
        print("\n---\ntime solve", ts.time)
        print("A dofs/(sec * proc)", stokes.disc.X.ndof / (ts.time * mpi_world.size) ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndof + stokes.disc.Q.ndof) / (ts.time * mpi_world.size) ) 

        # stokes.la.TestBlock()
    

    
Draw(stokes.velocity, mesh, "velocity")
Draw(stokes.pressure, mesh, "pressure")

# Draw(stokes.velocity - u_ex, mesh, "error")


