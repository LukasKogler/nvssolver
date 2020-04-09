import sys
import ngsolve as ngs
import netgen as ng

# from krylovspace_extension import BPCGSolver
from krylovspace_extension import BPCGSolver, GMResSolver, MinResSolver


_ngs_amg = True
try:
    import ngs_amg
except:
    _ngs_amg = False

_ngs_petsc = True
try:
    import ngs_petsc as petsc
except:
    _ngs_petsc = False

# _ngs_petsc = False
    
### Misc Utilities ###
class CVADD (ngs.BaseMatrix):
    def __init__(self, M, Mcv):
        super(CVADD, self).__init__()
        self.M = M
        self.Mcv = Mcv
    def IsComplex(self):
        return False
    def Height(self):
        return self.M.height
    def Width(self):
        return self.M.width
    def Mult(self, x, y):
        self.M.Mult(x, y)
    def MultAdd(self, scal, x, y):
        self.M.MultAdd(scal, x, y)
    def MultTrans(self, x, y):
        # self.M.MultTrans(x, y)
        self.M.T.Mult(x, y)
    def MultTransAdd(self, scal, x, y):
        self.M.MultTransAdd(scal, x, y)
    def CreateColVector(self):
        return self.Mcv.CreateColVector()
    def CreateRowVector(self):
        return self.Mcv.CreateRowVector()
    
class SPCST (ngs.BaseMatrix):
    def __init__(self, smoother, pc, mat, emb, swr = True):
        super(SPCST, self).__init__()
        self.swr = True # smooth with residuum
        self.A = mat
        self.S = smoother
        self.pc = pc
        self.E = emb
        self.ET = emb.T
        self.emb_pc = self.E @ self.pc @ self.ET
        self.xtemp = self.S.CreateColVector()
        self.res = self.S.CreateColVector()
        self.baux = self.E.CreateColVector()
        self.xaux = self.E.CreateColVector()
    def IsComplex(self):
        return False
    def Height(self):
        return self.S.height
    def Width(self):
        return self.S.width
    def CreateColVector(self):
        return self.S.CreateColVector()
    def CreateRowVector(self):
        return self.S.CreateRowVector()
    def MultAdd(self, scal, b, x):
        self.Mult(b, self.xtemp)
        x.data += scal * self.xtemp
    def MultTransAdd(self, scal, b, x):
        self.MultAdd(scal, b, x)
    def MultTrans(self, b, x):
        self.Mult(b, x)
    def Mult(self, b, x):
        x[:] = 0
        if self.swr: # Forward smoothing - update residual
            self.res.data = b
            self.S.Smooth(x, b, self.res, x_zero = True, res_updated = True, update_res = True)
        else:
            self.S.Smooth(x, b)
            self.res.data = b - self.A * x

        self.emb_pc.MultAdd(1.0, self.res, x)
        
        if self.swr: # Backward smoothing - no need to update residual
            self.S.SmoothBack(x, b, self.res, False, False, False)
        else:
            self.S.SmoothBack(x, b)
        

### END Misc Utilities ###


### FlowOptions ###

class FlowOptions:
    """
    A collection of parameters for Stokes/Navier-Stokes computations. Collects Boundary-conditions
    """
    def __init__(self, mesh, geom = None, nu = 1, inlet = "", outlet = "", wall_slip = "", wall_noslip = "",
                 uin = None, symmetric = True, vol_force = None, l2_coef = None):
        # geom/mesh
        self.geom = geom
        self.mesh = mesh

        # physical parameters
        self.nu = nu
        self.vol_force = vol_force
        self.l2_coef = l2_coef

        # BCs
        self.inlet = inlet
        self.uin = uin
        self.outlet = outlet
        self.wall_slip = wall_slip
        self.wall_noslip = wall_noslip

        # grad:grad or eps:eps?
        self.sym = symmetric
        
### END FlowOptions ###


### MCS Discretization ###

class MCS:
    def __init__(self, settings = None, order = 2, RT = False, hodivfree = False, compress = True, truecompile = False, pq_reg = 0, divdivpen = 0, trace_sigma = False):

        self.settings = settings
        self.order = order
        self.RT = RT
        self.hodivfree = hodivfree
        self.compress = compress
        self.sym = settings.sym
        self.truecompile = truecompile
        self.pq_reg = pq_reg
        self.ddp = divdivpen
        self.trace_sigma = trace_sigma

        # Spaces
        self.V = ngs.HDiv(settings.mesh, order = self.order, RT = self.RT, hodivfree = self.hodivfree, \
                          dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.wall_slip)
        if self.RT:
            if False: # "correct" version
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = order, \
                                                       dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.outlet)
            else: # not "correct", but works with facet-aux
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = order, \
                                                       dirichlet = settings.inlet + "|" + settings.wall_noslip + "|")
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = order, GGBubbles = True, discontinuous = True, ordertrace = self.order if self.trace_sigma else -1)
            # self.Sigma = ngs.HCurlDiv(mesh, order=order + 1, discontinuous=True, ordertrace=order) # slower I think
            self.Q = ngs.L2(settings.mesh, order = 0 if self.hodivfree else order)
            raise Exception("AAA")
        else:
            if True: # "correct" version
                # self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       # dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.outlet)
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.outlet)
            else: # works with facet-aux
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       dirichlet = settings.inlet + "|" + settings.wall_noslip)
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True, ordertrace = self.order-1 if self.trace_sigma else -1)
            # self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True)
            # self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True)
            self.Q = ngs.L2(settings.mesh, order = 0 if self.hodivfree else order-1)

        if self.sym:
            if settings.mesh.dim == 2:
                self.S = ngs.L2(settings.mesh, order=self.order if self.RT else order-1)
            else:
                self.S = ngs.VectorL2(settings.mesh, order=self.order if self.RT else order-1)
        else:
            self.S = None

        if self.compress:
            self.Sigma.SetCouplingType(ngs.IntRange(0, self.Sigma.ndof), ngs.COUPLING_TYPE.HIDDEN_DOF)
            self.Sigma = ngs.Compress(self.Sigma)
            if self.sym:
                self.S.SetCouplingType(ngs.IntRange(0, self.S.ndof), ngs.COUPLING_TYPE.HIDDEN_DOF)
                self.S = ngs.Compress(self.S)


    def Compile (self, term):
        return term.Compile(realcompile = self.truecompile, wait = True)

    
    def SetUpForms(self, compound = False):

        self.dS = ngs.dx(element_vb = ngs.BND)

        self.h = ngs.specialcf.mesh_size
        self.n = ngs.specialcf.normal(self.settings.mesh.dim)
        self.normal = lambda u : (u * self.n) * self.n
        self.tang = lambda u : u - (u * self.n) * self.n
        if self.settings.mesh.dim == 2:
            self.Skew2Vec = lambda m :  m[1, 0] - m[0, 1]
        else:
            self.Skew2Vec = lambda m : ngs.CoefficientFunction((m[0, 1] - m[1, 0], m[2, 0] - m[0, 2], m[1, 2] - m[2, 1]))

        if self.sym:
            self.X = ngs.FESpace([self.V, self.Vhat, self.Sigma, self.S])
        else:
            self.X = ngs.FESpace([self.V, self.Vhat, self.Sigma])

        if compound:
            self.Xext = ngs.FESpace([self.X, self.Q])
            if self.sym:
                (u, uhat, sigma, W), p = self.Xext.TrialFunction()
                (v, vhat, tau, R), q = self.Xext.TestFunction()
            else:
                (u, uhat, sigma), p = self.Xext.TrialFunction()
                (v, vhat, tau), q = self.Xext.TestFunction()
        else:
            self.Xext = None
            if self.sym:
                u, uhat, sigma, W = self.X.TrialFunction()
                v, vhat, tau, R = self.X.TestFunction()
            else:
                u, uhat, sigma = self.X.TrialFunction()
                v, vhat, tau = self.X.TestFunction()
            p,q = self.Q.TnT()

        nu = 2 * self.settings.nu if self.settings.sym else self.settings.nu
        self.a_vol = -1 / nu * ngs.InnerProduct(sigma, tau) \
                     + ngs.div(sigma) * v \
                     + ngs.div(tau) * u
        if not self.trace_sigma:
            self.a_vol += nu * ngs.div(u) * ngs.div(v)
        if self.sym:
           self.a_vol += ngs.InnerProduct(W, Skew2Vec(tau)) \
                         + ngs.InnerProduct(R, Skew2Vec(sigma))
        self.a_bnd = - ((sigma * self.n) * self.n) * (v * self.n) \
                       - ((tau * self.n) * self.n) * (u * self.n) \
                       - (sigma * self.n) * self.tang(vhat) \
                       - (tau * self.n) * self.tang(uhat)
        if self.settings.l2_coef is not None:
            self.a_vol += self.settings.l2_coef * ngs.InnerProduct(u, v)
            self.a_bnd += self.settings.l2_coef * ngs.InnerProduct(self.tang(uhat), self.tang(vhat))
        self.b_vol = ngs.div(u) * q
        self.bt_vol = ngs.div(v) * p

        # This is useful in some cases.
        self.divdiv = ngs.div(u) * ngs.div(v)
        self.uv = ngs.InnerProduct(u, v)
        self.uhvh = ngs.InnerProduct(self.tang(uhat), self.tang(vhat))

        if self.pq_reg != 0:
            self.c_vol = -self.pq_reg * nu * p * q
        else:
            self.c_vol = None

        if self.settings.vol_force is not None:
            self.f_vol = self.settings.vol_force * v
        else:
            self.f_vol = None
            
        self._stokesM = None
        self._stokesA = None
        self._stokesB = None
        self._stokesBT = None
        self._stokesBBT = None
        self._stokesf = None
        self._stokesC = None
        

    def stokesA(self):
        if self._stokesA == None:
            self._stokesA = self.Compile(self.a_vol) * ngs.dx + self.Compile(self.a_bnd) * self.dS
        return self._stokesA

    def stokesB(self):
        if self._stokesB == None:
            self._stokesB = self.Compile(self.b_vol) * ngs.dx
        return self._stokesB

    def stokesBT(self):
        if self._stokesB == None:
            self._stokesB = self.Compile(self.bt_vol) * ngs.dx
        return self._stokesB

    def stokesC(self):
        if self._stokesC == None and self.c_vol is not None:
            self._stokesC = self.Compile(self.c_vol) * ngs.dx
        return self._stokesC

    def stokesBBT(self):
        if self._stokesB == None:
            if self.c_vol == None:
                self._stokesB = self.Compile(self.b_vol + self.bt_vol) * ngs.dx
            else:
                self._stokesB = self.Compile(self.b_vol + self.bt_vol + self.c_vol) * ngs.dx
        return self._stokesB

    def stokesM(self):
        if self._stokesM == None:
            if self.c_vol == None:
                self._stokesM = self.Compile(self.a_vol + self.b_vol + self.bt_vol) * ngs.dx + self.Compile(self.a_bnd) * self.dS
            else:
                self._stokesM = self.Compile(self.a_vol + self.b_vol + self.bt_vol + self.c_vol) * ngs.dx + self.Compile(self.a_bnd) * self.dS
        return self._stokesM

    def stokesf(self):
        if self._stokesf == None and self.f_vol is not None:
            self._stokesf = self.Compile(self.f_vol) * ngs.dx
        return self._stokesf

### END MCS Discretization ###


### Stokes Template ###

class StokesTemplate():

    class LinAlg():
        def __init__(self, stokes, pc_ver = "aux", pc_opts = dict(), elint = False):

            self.comm = stokes.settings.mesh.comm

            self.elint = elint
            ## with static condensation + hodivfree, we can iterate on the schur complement
            self.it_on_sc = self.elint and stokes.disc.hodivfree

            self.block_la = pc_ver != "direct"

            self.pc_avail = { "direct"      : lambda astokes, opts : self.SetUpDirect(astokes, **opts),
                              "block"       : lambda astokes, opts : self.SetUpBlock(astokes, **opts),
                              "none"        : lambda astokes, opts : self.SetUpDummy(astokes) }

            self.pc_a_avail = { "direct"    : lambda astokes, opts : self.SetUpADirect(astokes, **opts),
                                "auxh1"     : lambda astokes, opts : self.SetUpAAux(astokes, **opts),
                                "auxfacet"  : lambda astokes, opts : self.SetUpAFacet(astokes, **opts),
                                "stokesamg" : lambda astokes, opts : self.SetUpStokesAMG(astokes, **opts) }

            stokes.disc.SetUpForms(compound = not self.block_la)
            
            if self.block_la:
                self.gfu = ngs.GridFunction(stokes.disc.X)
                self.velocity = self.gfu.components[0]
                self.velhat = self.gfu.components[1]
                self.p = ngs.GridFunction(stokes.disc.Q)
                self.pressure = self.p
                self.sol_vec = ngs.BlockVector([self.gfu.vec,
                                                self.pressure.vec])
            else:
                self.gfu = ngs.GridFunction(stokes.disc.Xext)
                self.velocity = self.gfu.components[0].components[0]
                self.velhat = self.gfu.components[0].components[1]
                self.pressure = self.gfu.components[1]
                self.sol_vec = self.gfu.vec
            
            self._to_assemble = []

            self.SetUpFWOps(stokes)
            self.SetUpRHS(stokes)

            self.Assemble()

            self.need_bp_scale = True

            if self.block_la:
                ## Even when using static condensation, we still need to iterate on the entire A matrix,
                ## except when we are also using hodivfree
                self.A = self.a.mat
                if self.elint and not self.it_on_sc:
                    Ahex, Ahext, Aii  = self.a.harmonic_extension, self.a.harmonic_extension_trans, self.a.inner_matrix
                    Id = ngs.IdentityMatrix(self.A.height)
                    self.A = (Id - Ahext) @ (self.A.local_mat + Aii) @ (Id - Ahex)
                    if self.a.space.mesh.comm.size > 1:
                        self.A = ngs.ParallelMatrix(self.A, row_pardofs = self.a.mat.row_pardofs, \
                                                    col_pardofs = self.a.mat.col_pardofs, op = ngs.ParallelMatrix.C2D)

                self.B = self.b.mat
                # self.BT = self.B.T
                self.BT = self.B.local_mat.CreateTranspose()
                if self.comm.size > 1:
                    self.BT = ngs.ParallelMatrix(self.BT, row_pardofs = self.B.col_pardofs, \
                                                 col_pardofs = self.B.row_pardofs, op = ngs.ParallelMatrix.C2D)
                self.C = None if self.c is None else self.c.mat

                self.M = ngs.BlockMatrix( [ [self.A, self.BT],
                                            [self.B, self.C] ] )

                self.rhs_vec = ngs.BlockVector ( [self.f.vec,
                                                  self.g.vec] )
            else:
                self.M = self.m.mat
                if self.elint and not self.it_on_sc:
                    Mhex, Mhext, Mii  = self.m.harmonic_extension, self.m.harmonic_extension_trans, self.m.inner_matrix
                    Id = ngs.IdentityMatrix(self.M.height)
                    self.M = (Id - Mhext) @ (self.M.local_mat + Mii) @ (Id - Mhex)
                    if self.m.space.mesh.comm.size > 1:
                        self.M = ngs.ParallelMatrix(self.Mext, row_pardofs = self.m.mat.row_pardofs, \
                                                       col_pardofs = self.m.mat.col_pardofs, op = ngs.ParallelMatrix.C2D)
                self.rhs_vec = self.f.vec

            if not pc_ver in self.pc_avail:
                raise Exception("invalid PC version!")
            else:
                self.pc_avail[pc_ver](stokes, pc_opts)

                
        def SetUpFWOps(self, stokes):
            # Forward operators
            if self.block_la:
                self.a = ngs.BilinearForm(stokes.disc.X, condense = self.elint, eliminate_hidden = stokes.disc.compress, \
                                          store_inner = self.elint and not self.it_on_sc)
                self.a += stokes.disc.stokesA()
                # self.a += 1e2 * stokes.settings.nu * stokes.disc.divdiv * ngs.dx

                if stokes.disc.ddp != 0:
                    self.a2 = ngs.BilinearForm(stokes.disc.X, condense = self.elint, eliminate_hidden = stokes.disc.compress, \
                                               store_inner = self.elint and not self.it_on_sc)
                    self.a2 += stokes.disc.stokesA()
                    self.a2 += stokes.disc.ddp * stokes.settings.nu * stokes.disc.divdiv * ngs.dx
                else:
                    self.a2 = self.a
                # self.a2 += 1e-4 * stokes.settings.nu * stokes.disc.uv * ngs.dx
                # self.a2 += 1e-4 * stokes.settings.nu * stokes.disc.uhvh * ngs.dx(element_vb=ngs.BND)

                # self.a = self.a2

                self.b = ngs.BilinearForm(trialspace = stokes.disc.X, testspace = stokes.disc.Q)
                self.b += stokes.disc.stokesB()

                if stokes.disc.stokesC() is not None:
                    self.c = ngs.BilinearForm(stokes.disc.Q)
                    self.c += stokes.disc.stokesC()
                else:
                    self.c = None

                self._to_assemble += [ self.a, self.a2, self.b, self.c ]

            else:
                if self.elint:
                    if not stokes.disc.hodivfree:
                        raise Exception("Static condensation for entire system only with hodivfree!")
                    stokes.disc.Q.SetCouplingType(ngs.IntRange(0, stokes.disc.Q.ndof), ngs.COUPLING_TYPE.WIREBASKET_DOF)
                    stokes.disc.Q.FinalizeUpdate()
                    stokes.disc.Xext.SetCouplingType(ngs.IntRange(stokes.disc.X.ndof, stokes.disc.Xext.ndof), ngs.COUPLING_TYPE.WIREBASKET_DOF)
                    stokes.disc.Xext.FinalizeUpdate()
                self.m = ngs.BilinearForm(stokes.disc.Xext, condense = self.elint, eliminate_hidden = stokes.disc.compress,
                                          tore_inner = self.elint and not self.it_on_sc)
                self.m += stokes.disc.stokesM()

                self._to_assemble += [ self.m ]
                self.a = self.m
                self.a2 = self.m
                
                
        def SetUpRHS(self, stokes):
            # Right hand side
            if self.block_la:
                self.f = ngs.LinearForm(stokes.disc.X)
                if stokes.disc.stokesf() is not None:
                    self.f += stokes.disc.stokesf()

                self.g = ngs.LinearForm(stokes.disc.Q)

                self._to_assemble += [ self.f, self.g ]

            else:
                self.f = ngs.LinearForm(stokes.disc.Xext)
                if stokes.disc.stokesf() is not None:
                    self.f += stokes.disc.stokesf()

                self._to_assemble += [ self.f ]


        def Assemble(self):
            for x in self._to_assemble:
                if x is not None:
                    x.Assemble()

        def PrepRHS(self, rhs_vec):
            if self.it_on_sc:
                rv = rhs_vec[0] if self.block_la else rhs_vec
                rv.Distribute()
                rv.local_vec.data += self.a.harmonic_extension_trans * rv.local_vec

        def ExtendSol(self, sol_vec, rhs_vec):
            if self.it_on_sc:
                sv = sol_vec[0] if self.block_la else sol_vec 
                rv = rhs_vec[0] if self.block_la else rhs_vec
                rv.Distribute()
                sv.Cumulate()
                sv.local_vec.data += self.a.inner_solve * rv.local_vec
                sv.local_vec.data += self.a.harmonic_extension * sv.local_vec

        def SetUpDirect(self, stokes, inv_type = None, **kwargs):
            # Direct inverse
            if self.block_la:
                raise Exception("Cannot invert block matrices!")
            else:
                itype = "umfpack" if inv_type is None else inv_type
                self.Mpre = self.m.mat.Inverse(stokes.disc.Xext.FreeDofs(self.elint), inverse = itype)
                if self.elint and not self.it_on_sc:
                    Mhex, Mhext, Miii  = self.m.harmonic_extension, self.m.harmonic_extension_trans, self.m.inner_solve
                    Id = ngs.IdentityMatrix(self.M.height)
                    if self.m.space.mesh.comm.size == 1:
                        self.Mpre = ((Id + Mhex) @ (self.Mpre) @ (Id + Mhext)) + Miii
                    else:
                        Ihex = ngs.ParallelMatrix(Id + Mhex, row_pardofs = self.m.mat.row_pardofs,
                                                  col_pardofs = self.m.mat.row_pardofs, op = ngs.ParallelMatrix.C2C)
                        Ihext = ngs.ParallelMatrix(Id + Mhext, row_pardofs = self.m.mat.row_pardofs,
                                                   col_pardofs = self.m.mat.row_pardofs, op = ngs.ParallelMatrix.D2D)
                        Isolve = ngs.ParallelMatrix(Miii, row_pardofs = self.m.mat.row_pardofs,
                                                    col_pardofs = self.m.mat.row_pardofs, op = ngs.ParallelMatrix.D2C)
                        self.Mpre = ( Ihex @ self.Mpre @ Ihext ) + Isolve
            
        def SetUpBlock(self, stokes, a_opts = { "type" : "direct" } , **kwargs):
            if not self.block_la:
                raise Exception("block-PC with big compond space todo")
            else:
                p,q = stokes.disc.Q.TnT()
                self.massp = ngs.BilinearForm(stokes.disc.Q)
                if stokes.disc.stokesC() is not None:
                    self.massp += -1 * stokes.disc.stokesC()
                if stokes.disc.ddp == 0:
                    self.massp +=  1/stokes.settings.nu * p * q * ngs.dx
                else:
                    self.massp +=  1/(stokes.settings.nu * stokes.disc.ddp) * p * q * ngs.dx
                # self.Spre = ngs.Preconditioner(self.massp, "direct")
                self.Spre = ngs.Preconditioner(self.massp, "local")

                self.massp.Assemble()

                aver = a_opts["type"] if "type" in a_opts else "direct"
                if aver in self.pc_a_avail:
                    self.pc_a_avail[aver](stokes, a_opts)
                else:
                    raise Exception("invalid pc type for A block!")

                self.ASpre = self.Apre

                if self.elint and not self.it_on_sc:
                    Ahex, Ahext, Aiii  = self.a2.harmonic_extension, self.a2.harmonic_extension_trans, self.a2.inner_solve
                    # print("Ahex ", Ahex)
                    # print("Ahext", Ahext)
                    # print("Aiii ", Aiii)
                    Id = ngs.IdentityMatrix(self.A.height)
                    if self.a.space.mesh.comm.size == 1:
                        self.Apre = ((Id + Ahex) @ (self.Apre) @ (Id + Ahext)) + Aiii
                    else:
                        Ihex = ngs.ParallelMatrix(Id + Ahex, row_pardofs = self.a.mat.row_pardofs,
                                                  col_pardofs = self.a.mat.row_pardofs, op = ngs.ParallelMatrix.C2C)
                        Ihext = ngs.ParallelMatrix(Id + Ahext, row_pardofs = self.a.mat.row_pardofs,
                                                   col_pardofs = self.a.mat.row_pardofs, op = ngs.ParallelMatrix.D2D)
                        Isolve = ngs.ParallelMatrix(Aiii, row_pardofs = self.a.mat.row_pardofs,
                                                    col_pardofs = self.a.mat.row_pardofs, op = ngs.ParallelMatrix.D2C)
                        self.Apre = ( Ihex @ self.Apre @ Ihext ) + Isolve
               
                self.Mpre = ngs.BlockMatrix( [ [self.Apre, None],
                                               [None, self.Spre] ] )

        def SetUpADirect(self, stokes, inv_type = None, **kwargs):
            if inv_type is None:
                if ngs.mpi_world.size > 1:
                    ainvt = "mumps"
                else:
                    ainvt = "sparsecholesky" if stokes.disc.compress else "umfpack"
            else:
                ainvt = inv_type
            self.Apre = self.a2.mat.Inverse(self.a.space.FreeDofs(self.elint), inverse = ainvt)
            # self.Apre = self.a.mat.Inverse(self.a.space.FreeDofs(self.elint), inverse = ainvt)
            self.need_bp_scale = False


        def SetUpAAux(self, stokes, amg_package = "petsc", amg_opts = dict(), mpi_thrad = False, mpi_overlap = False, shm = None,
                      bsblocks = None, multiplicative = True, el_blocks = False, mlt_smoother = True, **kwargs):
            if stokes.disc.hodivfree:
                raise Exception("Sorry, Auxiliary space not available with hodivfree (dual shapes not implemented) !")
            use_petsc = amg_package == "petsc"
            aux_direct = amg_package == "direct"
            if use_petsc:
                if not _ngs_petsc:
                    raise Exception("NGs-PETSc interface not available!")
            elif not aux_direct:
                if not _ngs_amg:
                    raise Exception("NGsAMG not available!")

            # Auxiliary space
            if True:
                V = ngs.H1(stokes.settings.mesh, order = 1, dirichlet = stokes.settings.wall_noslip + "|" + stokes.settings.inlet, \
                           dim = stokes.settings.mesh.dim)
            else:
                V = ngs.VectorH1(stokes.settings.mesh, order = 1, dirichlet = stokes.settings.wall_noslip + "|" + stokes.settings.inlet)
                # V = ngs.VectorH1(stokes.settings.mesh, order = 1, dirichletx = stokes.settings.wall_noslip + "|" + stokes.settings.inlet,
                                 # dirichlety = stokes.settings.wall_noslip + "|" + stokes.settings.inlet + "|" + stokes.settings.outlet)

            # print("V free", sum(V.FreeDofs()), "of", V.ndof)

            # Auxiliary space Problem
            u, v = V.TnT()
            a_aux = ngs.BilinearForm(V)
            if stokes.settings.sym:
                eps = lambda U : 0.5 * (ngs.grad(U) + ngs.grad(U).trans)
                a_aux += 2 * stokes.settings.nu * ngs.InnerProduct(eps(u), eps(v)) * ngs.dx
                if not use_petsc:
                    amg_cl = ngs_amg.elast_2d if stokes.settings.mesh.dim == 2 else ngs_amg.elast_3d
            else:
                a_aux += stokes.settings.nu * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * ngs.dx
                if not use_petsc:
                    amg_cl = ngs_amg.h1_2d if stokes.settings.mesh.dim == 2 else ngs_amg.h1_3d

            if stokes.settings.l2_coef is not None:
                a_aux += stokes.settings.l2_coef * ngs.InnerProduct(u, v) * ngs.dx
            if len(stokes.settings.outlet) > 0:
                a_aux += 1.0/stokes.disc.h * ngs.InnerProduct(stokes.disc.tang(u), stokes.disc.tang(v)) * ngs.ds(definedon=stokes.settings.mesh.Boundaries(stokes.settings.outlet))
            if len(stokes.settings.wall_slip) > 0:
                a_aux += 1.0/stokes.disc.h * ngs.InnerProduct(stokes.disc.normal(u), stokes.disc.normal(v)) * ngs.ds(definedon=stokes.settings.mesh.Boundaries(stokes.settings.wall_slip))

                
            ## nodalp2 experiment
            # amg_opts["ngs_amg_crs_alg"] = "ecol"
            # amg_opts["ngs_amg_enable_multistep"] = True
            # amg_opts["ngs_amg_lo"] = False
            # amg_opts["ngs_amg_subset"] = "free"
            # aux_pre = ngs.Preconditioner(a_aux, "bddc", coarsetype = "ngs_amg.h1_2d")
            # aux_pre = ngs.Preconditioner(a_aux, "bddc")

            # aux_pre = ngs.Preconditioner(a_aux, "direct")
            if not aux_direct:
                if not use_petsc:
                    aux_pre = amg_cl(a_aux, **amg_opts)
                    a_aux.Assemble()
                else:
                    if stokes.settings.sym:
                        raise Exception("symmetric gradient + PETSc-AMG TODO! (need to set RBMs)")
                    a_aux.Assemble()
                    mat_convert = petsc.PETScMatrix(a_aux.mat, freedofs=V.FreeDofs(), format=petsc.PETScMatrix.BAIJ)
                    aux_pre = petsc.PETSc2NGsPrecond(mat=mat_convert, name="aux_amg", petsc_options = amg_opts)
            else:
                a_aux.Assemble()
                aux_pre = a_aux.mat.Inverse(V.FreeDofs(), inverse="mumps" if ngs.mpi_world.size>1 else "sparsecholesky")

            # print("aux free ", sum(V.FreeDofs()), len(V.FreeDofs()))
            # evs_Aa = list(ngs.la.EigenValues_Preconditioner(mat=a_aux.mat, pre=aux_pre, tol=1e-10))
            # # evs_Aa = list(ngs.la.EigenValues_Preconditioner(mat=a_aux.mat, pre=ngs.Projector(V.FreeDofs(), True), tol=1e-10))
            # if self.a.space.mesh.comm.rank == 0:
            #     print("\n----")
            #     print("min ev. preAa\Aa:", evs_Aa[:5])
            #     print("max ev. preAa\Aa:", evs_Aa[-5:])
            #     print("cond-nr preAa\Aa:", evs_Aa[-1]/evs_Aa[0])
            #     print("----")
            
                
            # Embeddig Auxiliary space -> MCS space
            emb1 = ngs.comp.ConvertOperator(spacea = V, spaceb = stokes.disc.V, localop = True, parmat = False, bonus_intorder_ab = 2,
                                            range_dofs = stokes.disc.V.FreeDofs(self.elint))
            tc1 = ngs.Embedding(stokes.disc.X.ndof, stokes.disc.X.Range(0)) # to-compound
            emb2 = ngs.comp.ConvertOperator(spacea = V, spaceb = stokes.disc.Vhat, localop = True, parmat = False, bonus_intorder_ab = 2,
                                            range_dofs = stokes.disc.Vhat.FreeDofs(self.elint))
            tc2 = ngs.Embedding(stokes.disc.X.ndof, stokes.disc.X.Range(1)) # to-compound
            embA = (tc1 @ emb1) + (tc2 @ emb2)
            embA0 = embA
            # embA = ngs.Projector(stokes.disc.X.FreeDofs(self.elint), True) @ embA
            # print("embA 1 dims ", embA.height, embA.width)
            if ngs.mpi_world.size > 1:
                embA = ngs.ParallelMatrix(embA, row_pardofs = V.ParallelDofs(), col_pardofs = stokes.disc.X.ParallelDofs(),
                                          op = ngs.ParallelMatrix.C2C)
            # else:
                # embA = CVADD(embA, embA0)

            # print("embA 2 dims ", embA.height, embA.width)
                
            # Block-Smoother to combine with auxiliary PC    
            if mlt_smoother and V.mesh.comm.size>1 and not _ngs_amg:
                raise Exception("MPI-parallel multiplicative block-smoothers only available with NgsMPI!")
            X = stokes.disc.X
            x_free = X.FreeDofs(self.elint)
            sm_blocks = list()
            if el_blocks:
                if V.mesh.comm.size>1:
                    raise Exception("Element-Blocks are not possible with MPI!")
                for elem in stokes.settings.mesh.Elements():
                    block = list( dof for dof in X.GetDofNrs(elem) if dof>=0 and x_free[dof])
                    if len(block):
                        sm_blocks.append(block)
            else:
                for facet in stokes.settings.mesh.facets:
                    block = list( dof for dof in X.GetDofNrs(facet) if dof>=0 and x_free[dof])
                    if len(block):
                        sm_blocks.append(block)
                if not self.elint: # if elint, no need to check these - len(block) will be 0!
                    for elnr in range(stokes.settings.mesh.ne):
                        block = list( dof for dof in X.GetDofNrs(ngs.NodeId(ngs.ELEMENT, elnr)) if dof>=0 and x_free[dof])
                        if len(block):
                            sm_blocks.append(block)

            # The entire PC
            if mlt_smoother:
                if _ngs_amg:
                    bsmoother = ngs_amg.CreateHybridBlockGSS(mat = self.a.mat, blocks = sm_blocks, shm = ngs.mpi_world.size == 1)
                    self.Apre = SPCST(smoother = bsmoother, mat = self.a.mat, pc = aux_pre, emb = embA, swr = True)
                elif mpi_world.size == 1:
                    bsmoother = self.a.mat.local_mat.CreateBlockSmoother(sm_blocks)
                    self.Apre = SPCST(smoother = bsmoother, mat = self.a.mat, pc = aux_pre, emb = embA, swr = False)
                else:
                    raise Exception("Parallel Multiplicative block-smoothers only available with NgsAMG!")
            else:
                bsmoother = self.a.mat.local_mat.CreateBlockSmoother(blocks = sm_blocks, parallel=True)
                self.Apre = bsmoother + embA @ aux_pre @ embA.T

            # v1 = self.Apre.CreateColVector()
            # v2 = aux_pre.CreateColVector()
            # print("vec lens", len(v1), len(v2))
            # print("embA dims", embA.height, embA.width)
            # v3 = aux_pre.CreateColVector()
            # v4 = aux_pre.CreateRowVector()
            # print("2vec lens", len(v3), len(v4))
            # v3 = aux_pre.local_mat.CreateColVector()
            # v4 = aux_pre.local_mat.CreateRowVector()
            # print("2vec lens", len(v3), len(v4))
            
            ## END SetUpAAux ##
            
        def SetUpAFacet(self, stokes, amg_opts = dict(), **kwargs):
            if not _ngs_amg:
                raise Exception("Facet Auxiliary space Preconditioner only available with NgsAMG!")
            if stokes.settings.sym:
                if stokes.settings.mesh.dim == 2:
                    self.Apre = ngs_amg.mcs_epseps_2d(self.a, **amg_opts)
                else:
                    self.Apre = ngs_amg.mcs_epseps_3d(self.a, **amg_opts)
            else:
                if stokes.settings.mesh.dim == 2:
                    self.Apre = ngs_amg.mcs_gg_2d(self.a, **amg_opts)
                else:
                    self.Apre = ngs_amg.mcs_gg_3d(self.a, **amg_opts)
            self.a.Assemble() # <- TODO: is this necessary ??
            # print("AUX MAT: ", self.Apre.aux_mat)
            ## END SetUpAFacet ##

        def SetUpStokesAMG(self, stokes, amg_opts = dict(), **kwargs):
            if not _ngs_amg:
                raise Exception("Stokes AMG only available with NgsAMG!")
            # self.Apre = ngs_amg.stokes_gg_2d(self.a, **amg_opts)
            # self.Apre = ngs.Preconditioner(self.a, "ngs_amg.stokes_gg_2d", **amg_opts)
            blf = self.a2
            # self.Apre = ngs.Preconditioner(blf, "direct")
            self.Apre = ngs_amg.stokes_gg_2d(blf, **amg_opts)
            blf.Assemble()

        def SetUpDummy(self, stokes, **kwargs):
            self.Mpre = ngs.Projector(stokes.disc.Xext.FreeDofs(self.elint), True)
                
                
        def TestBlock(self, exai = False):
            o_ms_l = ngs.ngsglobals.msg_level
            ngs.ngsglobals.msg_level = 0

            evs_A = list(ngs.la.EigenValues_Preconditioner(mat=self.A, pre=self.Apre, tol=1e-10))
            if self.a.space.mesh.comm.rank == 0:
                print("\n----")
                print("Block-PC Condition number test")
                print("--")
                print("EVs for A block")
                print("min ev. preA\A:", evs_A[:5])
                print("max ev. preA\A:", evs_A[-5:])
                print("cond-nr preA\A:", evs_A[-1]/evs_A[0])

            # if self.elint and not self.it_on_sc:
            #     evs_AS = list(ngs.la.EigenValues_Preconditioner(mat=self.a.mat, pre=self.ASpre, tol=1e-10))
            #     if self.a.space.mesh.comm.rank == 0:
            #         print("--")
            #         print("EVs for condensed A block")
            #         print("min ev. preA\A:", evs_AS[:5])
            #         print("max ev. preA\A:", evs_AS[-5:])
            #         print("cond-nr preA\A:", evs_AS[-1]/evs_AS[0])
            
            if exai:
                if self.elint:
                    raise Exception("ex a inv for S test todo")
                ainv = self.a.mat.Inverse(self.a.space.FreeDofs(self.elint), inverse = "umfpack")
                S = self.B @ ainv @ self.B.T
            else:
                S = self.B @ self.Apre @ self.B.T

            evs_S = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=self.Spre, tol=1e-14))
            # evs_S = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=ngs.IdentityMatrix(S.height), tol=1e-14))
            evs0 = evs_S[0] if evs_S[0] > 1e-4 else evs_S[1]

            if self.a.space.mesh.comm.rank == 0:
                print("--")
                print("EVs for Schur Complement")
                print("min ev. preS\S:", evs_S[0:5])
                print("max ev. preS\S:", evs_S[-5:])
                print("cond-nr preS\S:", evs_S[-1]/(evs0))
                print("----\n")

            ngs.ngsglobals.msg_level = o_ms_l

            return evs_A[-1]/evs_A[0], evs_S[-1]/evs0, evs_A, evs_S

        
    def __init__(self, flow_settings = None, flow_opts = None, disc = None, disc_opts = None, sol_opts = None):

        # mesh, geometry, physical parameters 
        if flow_settings is not None:
            self.settings = flow_settings
        elif flow_opts is not None:
            self.settings = FlowOptions(**flow_opts)
        else:
            raise Exception("need either flow_settings or flow_opts!")

        # spaces, forms
        if disc is not None:
            self.disc = disc
        elif disc_opts is not None:
            self.disc = MCS(settings = self.settings, **disc_opts)
        else:
            raise Exception("need either disc or disc_opts!")

        # linalg, preconditioner
        self.InitLinAlg(sol_opts)

        self.velocity = self.la.velocity
        self.velhat = self.la.velhat
        self.pressure = self.la.pressure
            
    def InitLinAlg(self, sol_opts = None):
        if sol_opts is None:
            sol_opts = dict()
        self.la = self.LinAlg(self, **sol_opts)
        
    def AssembleLinAlg(self):
        self.la.Assemble()

    def Solve(self, tol = 1e-8, ms = 1000, solver = "minres"):

        homogenize = len(self.settings.inlet)>0 and self.settings.uin is not None

        rhs_vec = self.la.rhs_vec.CreateVector()
        rhs_vec.data = self.la.rhs_vec
        self.la.PrepRHS(rhs_vec = rhs_vec)
        if homogenize:
            sol_vec = self.la.sol_vec.CreateVector()
            self.velocity.Set(self.settings.uin, definedon=self.settings.mesh.Boundaries(self.settings.inlet))
            self.velhat.Set(self.settings.uin, definedon=self.settings.mesh.Boundaries(self.settings.inlet))
            rhs_vec.data -= self.la.M * self.la.sol_vec 
        else:
            sol_vec = self.la.sol_vec

        self.la.PrepRHS(rhs_vec = rhs_vec)

        if solver == "bp":
            if not self.la.block_la:
                raise Exception("For BPCG, use block-PC!")
            bp_cg = BPCGSolver(M = self.la.M, Mhat = self.la.Mpre, maxsteps=ms, tol=tol,
                               printrates = ngs.mpi_world.rank==0, rel_err=True)
            if self.la.need_bp_scale:
                bp_cg.ScaleAhat(tol=1e-10)#, scal = 1.0/1.35)
            sol_vec.data = bp_cg * rhs_vec
            nits = bp_cg.iterations
            self.solver = bp_cg
        elif solver == "gmres":
            # ngs.solvers.GMRes(A = self.la.M, b = rhs_vec, x = sol_vec, pre = self.la.Mpre,
            #                   tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps=ms ) 
            # note: to compare, use rel_err=False
            gmres = GMResSolver(M = self.la.M, Mhat = self.la.Mpre, maxsteps=ms, tol=tol,
                                printrates = ngs.mpi_world.rank==0, rel_err=True)
            sol_vec.data = gmres * rhs_vec
            nits = gmres.iterations
            self.solver = gmres
        elif solver == "apply_pc":
            sol_vec.data = self.la.Mpre * rhs_vec
            nits = 1
        elif solver == "minres":
            # note: to compare, use rel_err=False
            # ngs.solvers.MinRes(mat = self.la.M, rhs = rhs_vec, sol = sol_vec, pre = self.la.Mpre,
                               # tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps=ms)
            # nits = -1
            minres = MinResSolver(M = self.la.M, Mhat = self.la.Mpre, maxsteps=ms, tol=tol,
                                  printrates = ngs.mpi_world.rank==0, rel_err=True)
            sol_vec.data = minres * rhs_vec
            nits = minres.iterations
            self.solver = minres
        else:
            raise Exception("Use bp, gmres or minres as Solver!")


        if homogenize:
            self.la.sol_vec.data += sol_vec

        self.la.ExtendSol(sol_vec = self.la.sol_vec, rhs_vec = rhs_vec)

        # for k, comp in enumerate(self.la.gfu.components):
        #     print("SOL comp", k)
        #     print(self.la.gfu.components[k].vec)

        return nits

### END Stokes Template ###
