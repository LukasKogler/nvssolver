from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg
import sys

from utils import *
from FlowTemplates import StokesTemplate
# from FlowTemplates import *

ngsglobals.msg_level = 1

# flow_settings = ST_2d(maxh=0.1, nu=1e-3)
# flow_settings.mesh.Curve(3)
# pqr = 0

flow_settings = channel2d(maxh=0.05, nu=1, L=20)
pqr = 0

# flow_settings = ST_3d(maxh=0.1, nu=1e-2)
# flow_settings.mesh.Curve(3)
# pqr = 0

# flow_settings = vortex2d(maxh=0.025, nu=1)
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
              "divdivpen" : 1e2,
              "pq_reg" : pqr }

ddp = disc_opts["divdivpen"]

d = -1

sol_opts = { "elint" : True,
             # "block_la" : False,
             "pc_ver" : "block", # "block", "direct"
             "pc_opts" : {
                 "a_opts" : {
                     "type" : "direct",
                     # "inv_type" : "mumps",
                     "type" : "stokesamg",
                     "p1aux" : True,
                     # "type" : "auxfacet",
                     #"type" : "auxh1", 
                     # "amg_package" : "petsc",
                     "amg_package" : "ngs_amg",
                     #"amg_package" : "direct", # direct solve in auxiliary space
                     "mlt_smoother" : True,
                     "el_blocks" : False,
                     "el_blocks" : False,
                     # "type" : "stokesamg", 
                     "amg_opts" : {
                         "ngs_amg_aux_elmats" : False,
                         "ngs_amg_max_coarse_size" : 1,
                         "ngs_amg_max_levels" : 30,
                         "ngs_amg_energy" : "alg",
                         "ngs_amg_n_levels_d2_agg" : 10,
                         "ngs_amg_ecw_geom" : False,
                         "ngs_amg_clev" : "none",
                         "ngs_amg_enable_sp" : False,
                         "ngs_amg_sp_max_per_row" : 3,
                         "ngs_amg_ecw_robust" : False,
                         "ngs_amg_enable_multistep" : False,
                         "ngs_amg_aaf" : 0.1,
                         "ngs_amg_first_aaf" : 0.2,
                         "ngs_amg_gs_ver" : 3,
                         "ngs_amg_mg_cycle" : "BS",
                         "ngs_amg_sm_type" : "bgs",
                         "ngs_amg_sm_shm" : True,
                         "ngs_amg_sm_sl2" : True,
                         "ngs_amg_sm_steps" : 1,
                         "ngs_amg_sm_symm" : False,
                         "ngs_amg_spec_sm_types" : ["bgs", "bgs"],
                         "ngs_amg_keep_grid_maps" : True,
                         "ngs_amg_hpt_sm" : True,
                         "ngs_amg_hpt_sm_blk" : False,
                         "ngs_amg_comp_sm" : True,
                         "ngs_amg_comp_sm_steps" : 1,
                         "ngs_amg_comp_sm_blocks" : False,
                         "ngs_amg_comp_sm_blocks_el" : False,
                         "ngs_amg_log_level" : "extra",
                         "ngs_amg_print_log" : True,
                         "ngs_amg_do_test" : True,
                         }
                 }
             }
}

tsup = Timer("setup")
ts = Timer("solve")

SetNumThreads(1)
with TaskManager(pajetrace = 100 * 1024 * 1024):

    tsup.Start()
    stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )
    tsup.Stop()

    X = stokes.disc.X
    Q = stokes.disc.X
    print("X0 ndof", sum(X.components[0].FreeDofs(True)))
    print("X1 ndof", sum(X.components[1].FreeDofs(True)))
    print("X  ndof", sum(X.FreeDofs(True)))

    ts.Start()
    nits = stokes.Solve(tol=1e-8/ddp, ms = 200, solver = "gmres", use_sz = True, rel_err = False, presteps = 0)
    # nits = stokes.Solve(tol=1e-8, ms = 200, solver = "apply_pc", use_sz = True, rel_err = False, presteps = 0)
    # nits = 0
    ts.Stop()

    # if stokes.la.block_la:
    #     stokes.la.TestBlock()

    if mpi_world.rank == 0:
        print("\n---\ntime setup", tsup.time)
        print("A dofs  ", stokes.disc.X.ndofglobal)
        print("Q dofs  ", stokes.disc.Q.ndofglobal)
        print("---")
        print("A+Q dofs", stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (tsup.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (tsup.time * mpi_world.size) / 1000, "K" ) 
        print("---\ntime solve", ts.time)
        print("nits = ", nits)
        print("A dofs  ", stokes.disc.X.ndofglobal)
        print("Q dofs  ", stokes.disc.Q.ndofglobal)
        print("A+Q dofs", stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (ts.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (ts.time * mpi_world.size) / 1000, "K" )
        print("---\n")

    sys.stdout.flush()

    # sol_opts["elint"] = False
    # disc_opts["divdivpen"] = 0
    # sol_opts["pc_ver"] = "direct"
    # # sol_opts["elint"] = False
    # stokesex = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts )
    # stokesex.Solve(tol=1e-6, ms = 500, solver = "minres", presteps=0)
    # err = GridFunction(stokes.disc.V)
    # err.vec.data = stokes.velocity.vec - stokesex.velocity.vec
    # norm_err = Norm(err.vec)
    # norm_errl2 = abs(Integrate((stokes.velocity - stokesex.velocity)**2, flow_settings.mesh))**0.5
    # print("norm err l2 = ", norm_err)
    # print("norm err L2 = ", norm_errl2)
    # Draw(err, stokes.settings.mesh, "ERROR")
    # Draw(stokes.velocity, stokes.settings.mesh, "APPROX")
    # Draw(stokesex.velocity, stokes.settings.mesh, "EX")
    # ifds = stokes.disc.X.GetDofs(stokes.settings.mesh.Boundaries(flow_settings.inlet))

    # print("\ninflow DOFs: ", [x[0] for x in enumerate(ifds) if x[1]])
    # print("diff sol vec: ")
    # for x in [ (*x, stokes.disc.X.CouplingType(x[0]))  for x in enumerate(err.vec) if abs(x[1])>1e-8]:
        # print(*x)
    # print("diff sol vec: ", [ stokes.disc.X.CouplingType(x[0]) for x in enumerate(err.vec) if abs(x[1])>1e-8])
    
    Draw(stokes.velocity, stokes.settings.mesh, "velocity")
    # stokes.la.TestBlock()

    if False:
        pcm = stokes.la.Apre @ stokes.la.A
        print("get evals!")
        evecs, evals = MatIter (pcm, n_vecs = 3, lam_max = 1, lam_min = 0, reverse = True, M = 1e3, startvec = None, tol=1e-6, freedofs = stokes.disc.X.FreeDofs(stokes.la.elint and stokes.la.it_on_sc))
        print("evals = ", evals)
        fP = ngs.Projector(stokes.disc.X.FreeDofs(stokes.la.elint), True)
        L = ngs.L2(stokes.settings.mesh, order=1)
        for k,v in enumerate(evecs):
            gf = GridFunction(X)
            gf.vec.data = v
            Draw(gf.components[0], stokes.settings.mesh, 'BAD'+str(k))
            gfd = ngs.GridFunction(L)
            gfd.Set(ngs.div(gf.components[0]))
            Draw(gfd, stokes.settings.mesh, "div(BAD"+str(k)+")")
            # Draw(gf.components[1], stokes.settings.mesh, 'Vhat_BAD'+str(k))
            tv = gf.vec.CreateVector()
            tv.data = stokes.la.A * gf.vec
            tv2 = gf.vec.CreateVector()
            tv2.data = fP * tv
            print('norms', Norm(tv), Norm(tv2))

    if True:
        sol_opts["elint"] = False
        disc_opts["divdivpen"] = 0
        sol_opts["pc_ver"] = "direct"
        # sol_opts["elint"] = False
        stokesex = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts)
        stokesex.Solve(tol=1e-6, ms = 200, solver = "apply_pc")

        err = GridFunction(stokes.disc.V)
        err.vec.data = stokes.velocity.vec - stokesex.velocity.vec
        norm_err = Norm(err.vec)
        norm_errl2 = abs(Integrate((stokes.velocity - stokesex.velocity)**2, flow_settings.mesh))**0.5
        print("norm err l2 = ", norm_err)
        print("norm err L2 = ", norm_errl2)

    import matplotlib.pyplot as plt
    plt.semilogy(list(k for k,x in enumerate(stokes.solver.errors)), stokes.solver.errors)
    # plt.semilogy(list(k for k,x in enumerate(stokes.solver.errors)), list(stokes.solver.errors[0]/(10**k) for k in range(len(stokes.solver.errors))))
    plt.semilogy(list(k for k,x in enumerate(stokes.solver.errors)), list(0.5**k * stokes.solver.errors[0] for k in range(len(stokes.solver.errors))))
    plt.title("StokesAMG")
    plt.show()

            
    if d == 1:
        from bftester_stokes import *
        shape_test(stokes = stokes, aux_pc = stokes.la.bApre)
    elif d == 2:
        mesh = stokes.settings.mesh
        X = stokes.disc.X
        Apre = stokes.la.bApre
        v0 = GridFunction(X)
        v0.vec[:] = 0
        if flow_settings.uin is not None:
            v0.components[0].Set(flow_settings.uin, definedon=flow_settings.mesh.Boundaries(flow_settings.inlet))
            v0.components[1].Set(flow_settings.uin, definedon=flow_settings.mesh.Boundaries(flow_settings.inlet))
        rv = stokes.la.rhs_vec[0].CreateVector()
        rv.data = stokes.la.rhs_vec[0] - stokes.la.a2.mat * v0.vec
        v = GridFunction(X)
        v.vec.data = Apre * rv
        # v.vec.data += v0.vec
        Draw(v.components[0], mesh, "AUXINV_hd-vel")
        v2 = GridFunction(X)
        Apre.CINV(sol=v2.vec, rhs=rv)
        # print("\nCINV sol vec: ", [x for x in enumerate(v2.vec)])
        # v2.vec.data += v0.vec
        Draw(v2.components[0], mesh, "CINV_hd-vel")
        w = GridFunction(X)
        w.vec.data = stokes.la.a2.mat.Inverse(X.FreeDofs()) * rv
        # print("\nex sol vec: ", [x for x in enumerate(w.vec)])
        # w.vec.data += v0.vec
        Draw(w.components[0], mesh, "ex-hd-vel")
        L = ngs.L2(mesh, order=1)
        gfd = ngs.GridFunction(L)
        gfd.Set(ngs.div(w.components[0]))
        Draw(gfd, mesh, "div(EX)")
    
        
sys.stdout.flush()
# quit()

