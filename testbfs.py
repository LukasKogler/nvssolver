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

# flow_settings = vortex2d(maxh=0.2, nu=1)
# flow_settings.wall_noslip = ""
# #flow_settings.wall_slip = ""
# #flow_settings.inlet = ""
# #flow_settings.outlet = ".*"
# flow_settings.l2_coef = 1e-4
# pqr = 1e-6

flow_settings = ST_2d(maxh=0.025, nu=1e-3)
# flow_settings = channel2d(maxh=0.1, nu=1)
# flow_settings.uin = CoefficientFunction((1, 0))
# flow_settings.inlet = "left|top|bottom"
# flow_settings.wall_noslip = ""

mesh = flow_settings.mesh

disc_opts = { "order" : 2,
              "hodivfree" : True,
              "truecompile" : False, #mpi_world.size == 1,
              "RT" : False,
              "compress" : True,
              "divdivpen" : 1e6,
              "pq_reg" : 0 }

sol_opts = { "elint" : True,
             # "block_la" : False,
             "pc_ver" : "block", # "block", "direct"
             "pc_opts" : {
                 "a_opts" : {
                     #"type" : "direct",
                     "type" : "stokesamg",
                     #"type" : "auxfacet",
                     #"type" : "auxh1", 
                     "amg_package" : "ngs_amg", 
                     "inv_type" : "umfpack",
                     "amg_opts" : {
                         "ngs_amg_max_coarse_size" : 2,
                         "ngs_amg_max_levels" : 2,
                         "ngs_amg_sm_type" : "gs",
                         "ngs_amg_sm_ver" : 3,
                         "ngs_amg_keep_grid_maps" : True,
                         "ngs_amg_n_levels_d2_agg" : 1,
                         "ngs_amg_ecw_geom" : True,
                         "ngs_amg_enable_sp" : False,
                         "ngs_amg_sp_max_per_row" : 4,
                         "ngs_amg_clev" : "inv",
                         "ngs_amg_ecw_robust" : False,
                         "ngs_amg_enable_multistep" : False,
                         "ngs_amg_log_level" : "extra",
                         "ngs_amg_print_log" : True,
                         "ngs_amg_do_test" : False,
                         "ngs_amg_hpt_sm" : True,
                         "ngs_amg_aux_elmats" : False
                     }
                 }
             }
}


# print("ndofs X, V, Vhat", stokes.disc.X.ndof, stokes.disc.V.ndof, stokes.disc.Vhat.ndof)
# quit()

tsup = Timer("solve")
tsup.Start()
stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )
tsup.Stop()

def doit():
    ts = Timer("solve")
    ts.Start()
    nits = stokes.Solve(tol=1e-6, ms = 300, solver="minres")
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
        print("---\ntime solve", ts.time)
        print("Q dofs", stokes.disc.Q.ndofglobal)
        print("A dofs/proc", stokes.disc.X.ndofglobal / mpi_world.size / 1000, "K" ) 
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (ts.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (ts.time * mpi_world.size) / 1000, "K" )
        print("---\n")

    sys.stdout.flush()

    Draw(stokes.velocity, stokes.settings.mesh, "velocity")

    import matplotlib.pyplot as plt
    plt.semilogy(list(k for k,x in enumerate(stokes.solver.errors)), stokes.solver.errors)
    plt.show()

doit()


# quit()
# X = stokes.disc.X
# for fac in mesh.facets:
#     print("X dofs facet", fac, " = ", list(X.GetDofNrs(fac)))
# print("\n")
# for k, el in enumerate(mesh.Elements()):
#     print("X dofs el nr ", k, " = ", list(x for x in X.GetDofNrs(el) if x>=0))
# print("\n")

a = stokes.la.a
Apre = stokes.la.Apre

if False:
    from bftester_stokes import *
    shape_test(stokes = stokes, aux_pc = Apre)
elif False:
    v = GridFunction(X)
    Apre.PoC(1, 0, v.vec)
    print("v0 vec", [ x for x in enumerate(v.components[0].vec) if abs(x[1])>1e-8], "\n\n")
    print("v1 vec", [ x for x in enumerate(v.components[1].vec) if abs(x[1])>1e-8], "\n\n")
    Draw(v.components[0], mesh, "Pconst")

    w = GridFunction(X)
    w.components[0].Set(CoefficientFunction( (1, 0) ))
    print("w0 vec", [ x for x in enumerate(w.components[0].vec) if abs(x[1])>1e-8], "\n\n")
    # print("w1 vec", [ x for x in enumerate(w.components[1].vec) if x[1]!=0.0])
    Draw(w.components[0], mesh, "Pset")

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
elif False:
    v0 = GridFunction(X)
    v0.vec[:] = 0
    if flow_settings.uin is not None:
        v0.components[0].Set(flow_settings.uin, definedon=flow_settings.mesh.Boundaries(flow_settings.inlet))
        v0.components[1].Set(flow_settings.uin, definedon=flow_settings.mesh.Boundaries(flow_settings.inlet))

    rv = stokes.la.rhs_vec[0].CreateVector()
    rv.data = stokes.la.rhs_vec[0] - stokes.la.a2.mat * v0.vec

    v = GridFunction(X)

    # vaux = Apre.CreateAuxVector()
    # raux = Apre.CreateAuxVector()
    # raux.data = Apre.P.T * rv
    # auxi = Apre.aux_mat.Inverse(Apre.aux_freedofs, inverse="sparsecholesky")
    # vaux.data = auxi * raux
    # v.vec.data = Apre.P * vaux
    
    # Apre.CINV(sol=v.vec, rhs=rv)

    v.vec.data = Apre * rv

    v.vec.data += v0.vec
    Draw(v.components[0], mesh, "AUXINV_hd-vel")

    v2 = GridFunction(X)
    Apre.CINV(sol=v2.vec, rhs=rv)
    print("\nCINV sol vec: ", [x for x in enumerate(v2.vec)])
    v2.vec.data += v0.vec
    Draw(v2.components[0], mesh, "CINV_hd-vel")

    v3 = GridFunction(X)
    v3.vec.data = v.vec - v2.vec
    Draw(v3.components[0], mesh, "CA-diff-vel")
    
    vconst = GridFunction(X)
    Apre.PoC(1, 0, vconst.vec)
    vconst.vec.data += v0.vec
    Draw(vconst.components[0], mesh, "C-const")

    w = GridFunction(X)
    w.vec.data = stokes.la.a2.mat.Inverse(X.FreeDofs()) * rv
    print("\nex sol vec: ", [x for x in enumerate(w.vec)])
    w.vec.data += v0.vec
    Draw(w.components[0], mesh, "ex-hd-vel")

    w2 = GridFunction(X)
    # w2.components[0].Set(flow_settings.uin)
    # w2.components[0].Set(CoefficientFunction((y, 0)))
    w2.vec.data = w.vec - v.vec
    print("\ndiff sol vec: ", [x for x in enumerate(w2.vec) if abs(x[1])>1e-10])
    Draw(w2.components[0], mesh, "diff-hd-vel")

    Q = stokes.disc.Q
    B = stokes.la.B
    gfp = GridFunction(Q)
    gfp.vec.data = B * v.vec
    Draw(gfp, mesh, "div")
    gfp2 = GridFunction(Q)
    gfp2.vec.data = B * w.vec
    Draw(gfp2, mesh, "ex-div")
    gfp3 = GridFunction(Q)
    gfp3.vec.data = gfp.vec - gfp2.vec
    # gfp3.vec.data = B * w2.vec
    Draw(gfp3, mesh, "diff-div")
    
