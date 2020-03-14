from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg
import sys

from FlowTemplates import *

ngsglobals.msg_level = 1

case = 3

if case == 1:
    geo = g2d.unit_square
    ngm = geo.GenerateMesh(maxh=0.1)
    for l in range(2):
        ngm.Refine()
    mesh = Mesh(ngm)
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
    ngm = geo.GenerateMesh(maxh=0.1)
    for l in range(2):
        ngm.Refine()
    mesh = Mesh(ngm)
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
    H = 0.41
    L = 1
    geo.AddRectangle((0, 0), (L*H,H), bcs=("wall", "outlet", "wall", "inlet"))
    # geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
    ngm = geo.GenerateMesh(maxh=H*0.2)
    for l in range(0): # TODO: facet aux with refined mesh!!
        # raise "facet_aux with refined mesh does not work yet!! (check fine_facets ... )"
        ngm.Refine()
    mesh = Mesh(ngm)
    inlet = "inlet"
    outlet = "outlet"
    wall_noslip = "wall|cyl"
    wall_slip = ""
    # vol_force = CoefficientFunction( (0,0) )
    vol_force = None
    nu = 1e-3
    uin = CoefficientFunction( (4 * (2/0.41)**2 * y * (0.41 - y), 0))
elif case == 4:
    geo = csg.CSGeometry()
    channel = csg.OrthoBrick( csg.Pnt(-1, 0, 0), csg.Pnt(3, 0.41, 0.41) ).bc("wall")
    inlet = csg.Plane (csg.Pnt(0,0,0), csg.Vec(-1,0,0)).bc("inlet")
    outlet = csg.Plane (csg.Pnt(2.5, 0,0), csg.Vec(1,0,0)).bc("outlet")
    cyl = csg.Cylinder(csg.Pnt(0.5, 0.2,0), csg.Pnt(0.5,0.2,0.41), 0.05).bc("cyl")
    fluiddom = channel*inlet*outlet-cyl
    geo.Add(fluiddom)
    ngm = geo.GenerateMesh()
    for l in range(0):
        ngm.Refine()
    mesh = Mesh(ngm)
    inlet = "inlet"
    outlet = "outlet"
    wall_noslip = "wall|cyl"
    wall_slip = ""
    # vol_force = CoefficientFunction( (0,0) )
    vol_force = None
    nu = 1e-3
    uin = CoefficientFunction( (16*y*(0.41-y)*z*(0.41-z)/(0.41*0.41*0.41*0.41), 0, 0) )



ngsglobals.msg_level = 2
print("ok, have mesh ...")
sys.stdout.flush()

SetHeapSize(99999999)

# mesh.Curve(3)

flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = inlet, outlet = outlet, wall_slip = wall_slip,
                            wall_noslip = wall_noslip, uin = uin, symmetric = False, vol_force = vol_force)

order = 2

disc_opts = { "order" : order,
              "hodivfree" : False,
              "truecompile" : False, #mpi_world.size == 1,
              "RT" : False,
              "compress" : True,
              "pq_reg" : 1e-6 if case == 2 else 0 }

sol_opts = { "elint" : False,
             # "block_la" : False,
             "pc_ver" : "block", # "block", "direct"
             "pc_opts" : {
                 "a_opts" : {
                     # "type" : "direct",
                     "type" : "stokesamg",
                     # "type" : "auxfacet",
                     # "type" : "auxh1", 
                     "inv_type" : "umfpack",
                     "amg_opts" : {
                         "ngs_amg_max_coarse_size" : 50,
                         "ngs_amg_max_levels" : 2,
                         "ngs_amg_sm_type" : "gs",
                         "ngs_amg_keep_grid_maps" : True,
                         "ngs_amg_n_levels_d2_agg" : 1,
                         "ngs_amg_ecw_geom" : True,
                         "ngs_amg_enable_sp" : False,
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


stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )

a = stokes.la.a
Apre = stokes.la.Apre

from bftester_stokes import *
shape_test(stokes = stokes, aux_pc = Apre)
