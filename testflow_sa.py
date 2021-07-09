import os, sys
from ctypes import CDLL, RTLD_GLOBAL
try:
    CDLL(os.path.join(os.environ["MKLROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
except:
    try:
        CDLL(os.path.join(os.environ["MKL_ROOT"], "lib/intel64/libmkl_rt.so"), RTLD_GLOBAL)
    except:
        pass
    
from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg


import pickle

import sys
import ngsolve as ngs
import netgen as ng

# from krylovspace_extension import BPCGSolver
from krylovspace_extension import BPCGSolver, GMResSolver, MinResSolver
from utils import *
from FlowTemplates import StokesTemplate
# from FlowTemplates import *

ngsglobals.msg_level = 1

ngsglobals.testout = "test.out"
# SetNumThreads(1)

# flow_settings = channel2d(maxh=0.01, nu=1e-3, L=1)
# pqr = 0

# flow_settings = vortex2d(maxh=0.05, nu=1)
# pqr = 1e-7

# flow_settings = ST_2d(maxh=0.05, nu=1e-3, symmetric=False)
# flow_settings.mesh.Curve(3)
# pqr = 0

# flow_settings = channel2d(maxh=0.1, nu=1e-3, L=5, nref=1, symmetric=True)
# pqr = 0

# flow_settings = channel2d(maxh=0.1, nu=1e-3, L=40)
# pqr = 0

flow_settings = ST_3d(maxh=0.2, nu=1, symmetric=False, L=2.5*10)
# # flow_settings.mesh.Curve(3)
pqr = 0
# flow_settings.wall_noslip = ""
# flow_settings.inlet = ""
# flow_settings.outlet = ""

# H, L, W = 1, 1, 1
# geo = geo_3dchannel(H=H, W=W, L=L, obstacle=False)
# mesh = gen_ref_mesh(geo, ngs.mpi_world, maxh=0.25, nref=0, save=False, load=False, mesh_file="AAAA")
# uin = ngs.CoefficientFunction( (2.25 * (2/H)**2 * (2/W)**2 * ngs.y * (H - ngs.y) * ngs.z * (W - ngs.z), 0, 0))
# flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = 1, inlet = "", outlet = "", wall_slip = "",
#                             wall_noslip = "", uin = uin, symmetric = True, vol_force = None)


# flow_settings = vortex2d(maxh=0.025, nu=1)
# pqr = 1e-6

# print("verts " , flow_settings.mesh.nv)
# print("edges " , flow_settings.mesh.nedge)
# print("facets " , flow_settings.mesh.nfacet)
# print("els " , flow_settings.mesh.ne)

ddp = 1e6

disc_opts = { "order" : 1,
              "hodivfree" : False,
              "truecompile" : False, #mpi_world.size == 1,
              "RT" : False,
              "compress" : True,
              "trace_sigma" : False,
              "divdivpen" : ddp,
              "pq_reg" : pqr }

sao = { "ngs_amg_enable_sp" : False,
        "ngs_amg_enable_sp_spec" : [ False ],
        "ngs_amg_sp_max_per_row" : 300,
        # "ngs_amg_enable_sp_spec" : [ True ],
        # "ngs_amg_check_kvecs" : True,
        "ngs_amg_energy" : "alg",
        "ngs_amg_max_coarse_size" : 100,
        "ngs_amg_max_levels" : 20,
        "ngs_amg_clev" : "inv",
        "ngs_amg_cinv_type" : "masterinverse",
        "ngs_amg_enable_redist" : False,
        "ngs_amg_rd_min_nv_gl" : 100,
        "ngs_amg_rd_min_nv_thresh" : 200,
        "ngs_amg_rd_loc_thresh" : 0.5,
        "ngs_amg_rd_loc_gl" : 0.7,
        "ngs_amg_rd_seq_nv" : 10,
        "ngs_amg_rd_pfac" : 2,
        "ngs_amg_enable_multistep" : False,
        "ngs_amg_mg_cycle" : "BS",
        "ngs_amg_sm_symm" : False,
        "ngs_amg_sm_steps" : 1,
        "ngs_amg_sm_type" : "gs",
        "ngs_amg_cinv_type" : "masterinverse",
        "ngs_amg_hpt_sm" : True, # use Hiptmair smoother 
        "ngs_amg_hpt_sm_ex" : False, # Hiptmair exact solve
        "ngs_amg_hpt_sm_bs" : False, # Hiptmair Braess-Sarazin
        "ngs_amg_d2_agg" : True,
        "ngs_amg_d2_agg_spec" : [ True ],
        "ngs_amg_cinv_type_loc" : "sparsecholesky",
        "ngs_amg_test_smoothers" : False, #len(flow_settings.wall_noslip)>0,
        "ngs_amg_test_levels" : False, #len(flow_settings.wall_noslip)>0,
        "ngs_amg_do_test" : True, #len(flow_settings.wall_noslip)>0,
        "ngs_amg_log_level" : "extra", # "debug"
        "ngs_amg_log_level_pc" : "extra", #"debug",
        "ngs_amg_agg_print_aggs" : False,
        "ngs_amg_agg_print_vmap" : False,
        "ngs_amg_print_log" : True }

sol_opts = { "elint" : True,
             # "elint" : disc_opts["order"]>1 or disc_opts["compress"]==False,
             # "block_la" : False,
             "pc_ver" : "block", # "block", "direct"
             "pc_opts" : {
                 "a_opts" : {
                     # "type" : "direct",
                     # "type" : "auxh1",
                     "type" : "stokesamg",
                     "mlt_smoother" : True,
                     "blk_smoother" : True,
                     "sm_el_blocks" : False,
                     "sm_nsteps" : 1,
                     "sm_symm" : False,
                     "sm_nsteps_loc" : 1,
                     "sm_symm_loc" : False,
                     "amg_opts" : sao,
                 }
             }
}

SetNumThreads(1)
with TaskManager():#pajetrace = 50 * 2024 * 1024):


    tsup = Timer("sup")
    tsup.Start()
    stokes = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts)
    tsup.Stop()

    X = stokes.disc.X
    Q = stokes.disc.X
    print("X0 ndof", sum(X.components[0].FreeDofs(False)), sum(X.components[0].FreeDofs(True)))
    print("X1 ndof", sum(X.components[1].FreeDofs(False)), sum(X.components[1].FreeDofs(True)))
    print("X  ndof", sum(X.FreeDofs(False)), sum(X.FreeDofs(True)))

    print(" X0 {} / free(F) {} / free(T) {}".format(X.components[0].ndof, sum(X.components[0].FreeDofs(False)), sum(X.components[0].FreeDofs(True))))
    print(" X0 {} / free(F) {} / free(T) {}".format(X.components[1].ndof, sum(X.components[1].FreeDofs(False)), sum(X.components[1].FreeDofs(True))))
    print(" X {} / free(F) {} / free(T) {}".format(X.ndof, sum(X.FreeDofs(False)), sum(X.FreeDofs(True))))
    
    if mpi_world.rank==0:
        print("\n\n====\n====\n\n")
        sys.stdout.flush()
    mpi_world.Barrier()

    # for el in stokes.settings.mesh.facets:
    #     print("V   ", el, stokes.disc.V.GetDofNrs(el))
    #     print("Vhat", el, stokes.disc.Vhat.GetDofNrs(el))
    #     break

    # for elnr in range(stokes.settings.mesh.ne):
    #     # el = NodeId(elnr, )
    #     el = ngs.NodeId(ngs.ELEMENT, elnr)
    #     print("V   ", el, stokes.disc.V.GetDofNrs(el))
    #     print("Vhat", el, stokes.disc.Vhat.GetDofNrs(el))
    #     quit()

    # for el in stokes.settings.mesh.Elements():
    #     print("V   ", el, stokes.disc.V.GetDofNrs(el))
    #     print("Vhat", el, stokes.disc.Vhat.GetDofNrs(el))
    #     break
    
    # cd = True
    # a1 = ngs.BilinearForm(stokes.disc.X, condense = cd, eliminate_hidden = True, store_inner = True)
    # a1 += stokes.disc.stokesA()
    # a1.Assemble()
    # a1inv = a1.mat.Inverse(stokes.disc.X.FreeDofs(cd), inverse="mumps")

    # if mpi_world.rank==0:
    #     print("\n\n====\n====\n\n")
    #     sys.stdout.flush()
    # mpi_world.Barrier()

    # quit()

    # if sol_opts["pc_ver"] == "block":
       # stokes.la.TestBlock()

    # quit()

SetNumThreads(1)
with TaskManager(pajetrace = 50 * 2024 * 1024):
    ts = Timer("solve")
    ts.Start()
    # nits = stokes.Solve(tol=1e-6, ms = 100, presteps = 0, solver = "minres", use_sz = False, rel_err = False, printrates = True)
    nits = stokes.Solve(tol=1e-6, ms = 500, presteps = 0, solver = "gmres", use_sz = True, rel_err = False, printrates = True)
    # nits = 0
    ts.Stop()

SetNumThreads(1)
with TaskManager():#pajetrace = 50 * 2024 * 1024):
    #input()
    if mpi_world.rank == 0:
        print("\n---")
        print("els ", stokes.disc.X.mesh.ne)
        print("A dofs  ", stokes.disc.X.ndofglobal)
        print("Q dofs  ", stokes.disc.Q.ndofglobal)
        print("nits = ", nits)
        print("---\ntime setup", tsup.time)
        print("A+Q dofs", stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (tsup.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (tsup.time * mpi_world.size) / 1000, "K" ) 
        print("---\ntime solve", ts.time)
        print("A dofs/(sec * proc)", stokes.disc.X.ndofglobal / (ts.time * mpi_world.size) / 1000, "K" ) 
        print("A+Q dofs/(sec * proc)", (stokes.disc.X.ndofglobal + stokes.disc.Q.ndofglobal) / (ts.time * mpi_world.size) / 1000, "K" ) 
        print("---\n")

    sys.stdout.flush()

    sol_opts["elint"] = False
    # disc_opts["divdivpen"] = 0
    sol_opts["pc_ver"] = "direct"
    sol_opts["pc_opts"]["inv_type"] = "mumps"
    # # sol_opts["elint"] = False

    stokesex = StokesTemplate(disc_opts = disc_opts, flow_settings = flow_settings, sol_opts = sol_opts)
    stokesex.Solve(tol=1e-6, ms = 500, solver = "apply_pc")

    # quit()

    err = GridFunction(stokes.disc.V)
    err.vec.data = stokes.velocity.vec - stokesex.velocity.vec
    norm_err = Norm(err.vec)
    norm_errl2 = abs(Integrate((stokes.velocity - stokesex.velocity)**2, flow_settings.mesh))**0.5
    print("norm err l2 = ", norm_err)
    print("norm err L2 = ", norm_errl2)

    quit()

    if mpi_world.rank == 0:
        import matplotlib.pyplot as plt
        plt.semilogy(stokes.solver.errors)
        plt.show()


    # Vvel = HDiv(mesh = stokes.settings.mesh, order = disc_opts["order"])
    # myvel = GridFunction(Vvel)
    # myvel.vec.data = stokes.velocity.vec
    # Draw(stokes.velocity, stokes.settings.mesh, "velocity")
    # picklefile = open("myout.dat", "wb")
    # pickle.dump(myvel, picklefile)
    # picklefile.close()
    # print(stokes.velocity)


    print("finished")
