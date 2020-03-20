from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg
import sys

from utils import *
from FlowTemplates import *

ngsglobals.msg_level = 1

ngsglobals.msg_level = 2
print("ok, have mesh ...")
sys.stdout.flush()

SetHeapSize(99999999)

# mesh.Curve(3)

flow_settings = vortex2d(maxh=0.4, nu=1)
pqr = 1e-6
order = 2

mesh = flow_settings.mesh

disc_opts = { "order" : order,
              "hodivfree" : False,
              "truecompile" : False, #mpi_world.size == 1,
              "RT" : False,
              "compress" : True,
              "pq_reg" : pqr }

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
                         "ngs_amg_clev" : "none",
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

X = stokes.disc.X
for fac in mesh.facets:
    print("X dofs facet", fac, " = ", list(X.GetDofNrs(fac)))
print("\n")
for k, el in enumerate(mesh.Elements()):
    print("X dofs el nr ", k, " = ", list(x for x in X.GetDofNrs(el) if x>=0))
print("\n")

a = stokes.la.a
Apre = stokes.la.Apre

if False:
    from bftester_stokes import *
    shape_test(stokes = stokes, aux_pc = Apre)
else:
    v = GridFunction(X)
    Apre.PoC(1, 0, v.vec)
    print("v0 vec", [ x for x in enumerate(v.components[0].vec) if abs(x[1])>1e-8], "\n\n")
    print("v1 vec", [ x for x in enumerate(v.components[1].vec) if abs(x[1])>1e-8], "\n\n")
    Draw(v.components[0], mesh, "Pconst")

    w = GridFunction(X)
    w.components[0].Set(CoefficientFunction( (1, 0) ))
    print("w0 vec", [ x for x in enumerate(w.components[0].vec) if abs(x[1])>1e-8], "\n\n")
    # print("w1 vec", [ x for x in enumerate(w.components[1].vec) if x[1]!=0.0])
    # Draw(w.components[0], mesh, "Pset")

    diff = GridFunction(X)
    diff.components[0].vec.data = w.components[0].vec - v.components[0].vec
    print("diff vec", [ x for x in enumerate(diff.components[0].vec) if abs(x[1])>1e-8], "\n\n")
    Draw(diff.components[0], mesh, "diff")


    Q = stokes.disc.Q
    B = stokes.la.B
    gfp = GridFunction(Q)
    gfp.vec.data = B * v.vec
    Draw(gfp, mesh, "div")

    # gfpd = GridFunction(Q)
    # gfpd.vec.data = B * diff.vec
    # Draw(gfpd, mesh, "divdiff")
