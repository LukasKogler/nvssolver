import sys
import ngsolve as ngs
import netgen as ng

# from krylovspace_extension import BPCGSolver
from krylovspace_extension import BPCGSolver, GMResSolver, MinResSolver


_ngsAMG = True
try:
    import NgsAMG
except:
    _ngsAMG = False

_ngs_petsc = True
try:
    import ngs_petsc as petsc
except:
    _ngs_petsc = False

# _ngs_petsc = False

def myPrint(*args, masterOnly=True, sync=False):
    if ( not masterOnly ) or ( ngs.mpi_world.rank == 0 ):
        print(*args)
        sys.stdout.flush()

def MakeFacetBlocks(V, freedofs=None):
    blocks = []
    if freedofs is not None:
        for facet in V.mesh.facets:
            block = list( dof for dof in V.GetDofNrs(facet) if dof>=0 and freedofs[dof])
            if len(block):
                blocks.append(block)
    else:
        for facet in V.mesh.facets:
            block = list( dof for dof in V.GetDofNrs(facet) if dof>=0)
            if len(block):
                blocks.append(block)
    return blocks

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

class CumulateOp (ngs.BaseMatrix):
    def __init__(self, M):
        super(CumulateOp, self).__init__()
        self.M = M
    def IsComplex(self):
        return False
    def Height(self):
        return self.M.height
    def Width(self):
        return self.M.width
    def CreateColVector(self):
        return self.M.CreateColVector()
    def CreateRowVector(self):
        return self.M.CreateRowVector()
    def Mult(self, b, x):
        self.M.Mult(b, x)
        x.Cumulate()
    def MultAdd(self, scal, b, x):
        self.M.MultAdd(scal, b, x)
        x.Cumulate()

class SPCST (ngs.BaseMatrix):
    def __init__(self, smoother, pc, mat, emb, embT = None, swr=True, steps=1):
        super(SPCST, self).__init__()
        self.swr = swr # smooth with residuum
        self.A = mat
        self.S = smoother
        self.pc = pc
        self.E = emb
        self.ET = emb.T if embT is None else embT
        self.steps = steps
        if self.pc is not None:
            self.emb_pc = self.E @ self.pc @ self.ET
        self.xtemp = self.S.CreateColVector()
        self.res = self.S.CreateColVector()
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
        x[:] = 0.0
        if self.swr: # Forward smoothing - update residual
            self.res.data = b
            self.S.SmoothK(self.steps, x, b, self.res, True, True, True)
        else:
            for l in range(self.steps):
                self.S.Smooth(x, b)
            self.res.data = b - self.A * x
        if self.pc is not None:
            # self.emb_pc.MultAdd(1.0, self.res, x)
            self.emb_pc.Mult(self.res, x)
        if self.swr: # Backward smoothing - no need to update residual
            self.S.SmoothBackK(self.steps, x, b, self.res, False, False, False)
        else:
            for l in range(self.steps):
                self.S.SmoothBack(x, b)

class AuxiliarySpacePreconditioner(ngs.BaseMatrix):
    def __init__(self, blf, aux_pre, embedding, sm_blocks = ["facet"], embeddingT = None, multiplicative = True,
                 sm_nsteps = 1, sm_nsteps_loc = 1, sm_symm = False, sm_symm_loc = False,
                 elint = False, mpi_thread = True):
        super(AuxiliarySpacePreconditioner, self).__init__()
        self.multiplicative = multiplicative
        self.blf = blf
        self.aux_pre = aux_pre
        self.embedding  = embedding
        self.embeddingT = embeddingT if embeddingT is not None else self.embedding.T
        # self.comm = self.blf.spacee.comm
        self.elint = elint
        self.freedofs = self.blf.space.FreeDofs(self.elint)
        self.swr = False
        self.op = None
        self._block_makers = { "F"   : self.MakeFacetBlocks,    # facets
                               "C"   : self.MakeCellBlocks,     # cells
                               "F+C" : self.MakeFCBlocks,       # facets + appended cells
                               "EL"  : self.MakeElementBlocks } # elements
        self.SetUpSmoother(sm_blocks, sm_nsteps, sm_nsteps_loc,
                           sm_symm, sm_symm_loc, mpi_thread)
        self.Finalize()

    def Finalize(self):
        if self.multiplicative:
            self.op = SPCST(smoother = self.smoother, mat = self.blf.mat, pc = self.aux_pre,
                            emb = self.embedding, embT = self.embeddingT, swr = self.swr, steps = 1)
        else:
            self.op = self.smoother + self.embedding @ self.aux_pre @ self.embeddingT

    def SetUpSmoother(self, block_codes, sm_nsteps, sm_nsteps_loc,
                      sm_symm, sm_symm_loc, mpi_thread):
        sm_blocks = self.CalcBlocks(block_codes)
        if self.blf.space.mesh.comm.size == 1:
            sm_nsteps = sm_nsteps_loc*sm_nsteps
            sm_symm = sm_symm or sm_symm_loc
        if self.multiplicative: # multiplicative: smooth, AUX, moothback
            if _ngsAMG: # use MPI-parallel multiplicative smoothers from NgsAMG
                self.swr = True
                if sm_blocks is None:
                    print("CreateHybridGSS")
                    self.smoother = NgsAMG.CreateHybridGSS(mat = self.blf.mat, freedofs = self.freedofs, mpi_overlap = True,
                                                            mpi_thread = mpi_thread, nsteps = sm_nsteps, symm = sm_symm, pinv = False,
                                                            nsteps_loc = sm_nsteps_loc, symm_loc = sm_symm_loc)
                else:
                    self.smoother = NgsAMG.CreateHybridBlockGSS(mat = self.blf.mat, blocks = sm_blocks, shm = False, # new bs not shm-par!
                                                                 mpi_overlap = True, mpi_thread = mpi_thread, pinv = False,
                                                                 bs2 = True, blocks_no = False,
                                                                 nsteps = sm_nsteps, symm = sm_symm,
                                                                 nsteps_loc = sm_nsteps_loc, symm_loc = sm_symm_loc)
            else: # shm-parallel multiplicative smoothers from NGSolve (no MPI)
                self.swr = False
                if sm_blocks is None:
                    self.smoother = self.blf.mat.local_mat.CreateSmoother(self.freedofs)
                else:
                    self.smoother = self.blf.mat.local_mat.CreateBlockSmoother(sm_blocks)
        else: # additive: smooth + AUX
            if sm_blocks is None:
                if comm.size > 1:
                    # annoying - need to define a Preconditioner("local") BEFORE assemble...
                    # need to work with a blocksmoother..
                    raise Exception("this is annoying")
                else:
                    self.smoother = self.blf.mat.CreateSmoother(self.freedofs)
            else:
                self.smoother = self.blf.mat.local_mat.CreateBlockSmoother(sm_blocks, parallel=self.blf.space.mesh.comm.size>1)
                if self.blf.space.mesh.comm.size > 1: # EVIL HACK to get this to work as a D->C operation
                    # ParallelMatrix calls mult with local mat and .local_vec
                    # input is distributed, output set to distributed
                    self.smoother = ngs.ParallelMatrix(self.smoother, self.blf.mat.row_pardofs,
                                                       self.blf.mat.row_pardofs, ngs.ParallelMatrix.D2D)
                    # cumulateop cumulates output vector
                    self.smoother = CumulateOp(self.smoother)

    def MakeFacetBlocks(self):
        blocks = []
        V = self.blf.space
        for facet in V.mesh.facets:
            block = list( dof for dof in V.GetDofNrs(facet) if dof>=0 and self.freedofs[dof])
            if len(block):
                blocks.append(block)
        return blocks

    def MakeFCBlocks(self):
        blocks = []
        V = self.blf.space
        for facet in V.mesh.facets:
            block = list( dof for dof in V.GetDofNrs(facet) if dof>=0 and self.freedofs[dof])
            if not self.elint: # if elint, no need to check these - len(block) will be 0!
                for elid in facet.elements:
                    cellid = ngs.NodeId(ngs.ELEMENT, elid.nr)
                    block = list( dof for dof in V.GetDofNrs(cellid) if dof>=0 and self.freedofs[dof])
            if len(block):
                blocks.append(block)
        return blocks

    def MakeCellBlocks(self):
        blocks = []
        V = self.blf.space
        for elnr in range(V.mesh.ne):
            block = list( dof for dof in V.GetDofNrs(ngs.NodeId(ngs.ELEMENT, elnr)) if dof>=0 and self.freedofs[dof])
            if len(block):
                blocks.append(block)
        return blocks

    def MakeElementBlocks(self):
        blocks = []
        V = self.blf.space
        for elem in V.mesh.Elements():
            block = list( dof for dof in V.GetDofNrs(elem) if dof>=0 and self.freedofs[dof])
            if len(block):
                blocks.append(block)
        return blocks

    def CalcBlocks(self, sm_blocks):
        if len(sm_blocks) > 0:
            allblocks = []
            for bt in sm_blocks:
                if bt in self._block_makers:
                    allblocks += self._block_makers[bt]()
            return allblocks
        else:
            return None

    def IsComplex(self):
        return self.blf.mat.IsComplex()

    def Height(self):
        return self.blf.mat.height

    def Width(self):
        return self.blf.mat.width

    def CreateColVector(self):
        return self.blf.mat.CreateColVector()

    def CreateRowVector(self):
        return self.blf.mat.CreateRowVector()

    def Mult(self, x, y):
        y.data = self.op * x

    def MultTrans(self, x, y):
        y.data = self.op * x

    def MultAdd(self, scal, x, y):
        y.data += scal * self.op * x

    def MultTransAdd(self, scal, x, y):
        y.data += scal * self.op * x

    def GetOp():
        return self.op

# Schoeberl-Zulehner PC
class SZPC(ngs.BaseMatrix):
    def __init__(self, M, Ahat, Shat):
        super(SZPC, self).__init__()
        self.M = M
        self.A, self.B, self.BT = M[0,0], M[1,0], M[0,1]
        # lams1 = list(ngs.la.EigenValues_Preconditioner(mat=self.A, pre=Ahat, tol=1e-6))
        # self.Ahat = 1.0/(1.01 * lams1[-1]) * Ahat
        self.Ahat = Ahat
        # S = self.B @ self.Ahat @ self.BT
        # lams2 = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=Shat, tol=1e-6))
        # self.Shat = 1.0/(lams2[-1] * 1.01) * Shat
        self.Shat = Shat
        self.mBTSg2 = self.A.CreateColVector()
        self.g2 = self.Shat.CreateColVector()
        self.xtemp = self.M.CreateColVector()
    def IsComplex(self):
        return False
    def Height(self):
        return self.M.height
    def Width(self):
        return self.M.width
    def CreateColVector(self):
        return self.M.CreateColVector()
    def CreateRowVector(self):
        return self.M.CreateRowVector()
    def Mult(self, b, x):
        f, g = b[0], b[1]
        x[0].data = self.Ahat * f
        self.g2.data = g - self.B * x[0]
        x[1].data = - self.Shat * self.g2
        self.mBTSg2.data = self.BT * x[1]
        x[0].data -= self.Ahat * self.mBTSg2
    def MultTrans(self, b, x):
        self.Mult(b, x)
    def MultAdd(self, scal, b, x):
        self.Mult(b, self.xtemp)
        x.data += scal * self.xtemp
    def MultTransAdd(self, scal, b, x):
        self.MultAdd(scal, b, x)

class PSZPC(ngs.BaseMatrix):
    def __init__(self, M, Ahat, Shat):
        super(PSZPC, self).__init__()
        self.M = M
        self.A, self.B, self.BT = M[0,0], M[1,0], M[0,1]
        # lams1 = list(ngs.la.EigenValues_Preconditioner(mat=self.A, pre=Ahat, tol=1e-6))
        # self.Ahat = 1.0/(1.01 * lams1[-1]) * Ahat
        self.Ahat = Ahat
        # S = self.B @ self.Ahat @ self.BT
        # lams2 = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=Shat, tol=1e-6))
        # self.Shat = 1.0/(lams2[0] * 1.01) * Shat
        self.Shat = Shat
        self.mBTSg2 = self.A.CreateColVector()
        self.g2 = self.Shat.CreateColVector()
        self.xtemp = self.M.CreateColVector()
    def IsComplex(self):
        return False
    def Height(self):
        return self.M.height
    def Width(self):
        return self.M.width
    def CreateColVector(self):
        return self.M.CreateColVector()
    def CreateRowVector(self):
        return self.M.CreateRowVector()
    def Mult(self, b, x):
        # A = ngs.BlockMatrix([ [ self.la.Apre, self.la.Apre @ self.la.BT @ self.la.Spre ],
                              # [ None, - self.la.Spre ] ])
        f, g = b[0], b[1]
        x[1].data = - self.Shat * g
        self.mBTSg2.data = f - self.BT * x[1]
        x[0].data = self.Ahat * self.mBTSg2.data
    def MultTrans(self, b, x):
        self.Mult(b, x)
    def MultAdd(self, scal, b, x):
        self.Mult(b, self.xtemp)
        x.data += scal * self.xtemp
    def MultTransAdd(self, scal, b, x):
        self.MultAdd(scal, b, x)
### END Misc Utilities ###


### FlowOptions ###

class FlowOptions:
    """
    A collection of parameters for Stokes/Navier-Stokes computations. Collects Boundary-conditions
    """
    def __init__(self, mesh, geom = None, nu = 1, inlet = "", outlet = "", wall_slip = "", wall_noslip = "",
                 uin = None, symmetric = True, vol_force = None, l2_coef = None, fluid_domain = None,
                 vel_outlet_f = None, velhat_outlet_f = None):
        # geom/mesh
        self.geom = geom
        self.mesh = mesh

        # physical parameters
        self.nu = nu
        self.vol_force = vol_force
        self.l2_coef = l2_coef
        self.vel_outlet_f = vel_outlet_f  # RHS for outlet: (sigma + p I)n = vel_outflow_f
        self.velhat_outlet_f = velhat_outlet_f  # RHS for outlet: (sigma + p I)n = vel_outflow_f

        # domains
        self.fluid_domain = ".*" if fluid_domain is None else fluid_domain

        # BCs
        self.inlet = inlet
        self.uin = uin
        self.outlet = outlet
        self.wall_slip = wall_slip
        self.wall_noslip = wall_noslip

        # grad:grad or eps:eps?
        self.sym = symmetric

### END FlowOptions ###

### FlowDiscretization ###
class FlowDiscretization:
    def __init__(self, settings = None, order = 2, compress = True, truecompile = False, \
                 pq_reg = 0, divdivpen = 0, bonus_intorder_rhs = 0):
        self.settings = settings
        self.order = order
        self.compress = compress
        self.truecompile = truecompile
        self.pq_reg = pq_reg
        self.ddp = divdivpen
        self.bonus_intorder_rhs = bonus_intorder_rhs

        self.sym = settings.sym

        self.nueff = (2 * self.settings.nu) if self.settings.sym else self.settings.nu

        self._stokesM = None
        self._stokesA = None
        self._stokesB = None
        self._stokesBT = None
        self._stokesBBT = None
        self._stokesf = None
        self._stokesC = None

    def Compile (self, term):
        return term.Compile(realcompile = self.truecompile, wait = True)


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
        if self._stokesf == None:
            if self.f_vol is not None:
                self._stokesf = self.Compile(self.f_vol) * ngs.dx(bonus_intorder=self.bonus_intorder_rhs)
            if self.rhs_outlet is not None:
                if self._stokesf == None:
                    self._stokesf = self.Compile(self.rhs_outlet) * ngs.ds(bonus_intorder=self.bonus_intorder_rhs,
                                                                           definedon=self.settings.outlet)
                else:
                    self._stokesf += self.Compile(self.rhs_outlet) * ngs.ds(bonus_intorder=self.bonus_intorder_rhs,
                                                                            definedon=self.settings.outlet)
            # self._stokesf = self.Compile(self.f_vol) * ngs.dx
            # self._stokesf = self.f_vol * ngs.dx(bonus_intorder=self.bonus_intorder_rhs)
        # self._stokesf = self.Compile(self.f_bnd) * ngs.ds
        return self._stokesf

### END FlowDiscretization ###


### HDG Discretization ###

class HDG(FlowDiscretization):
    def __init__(self, settings = None, order = 2, hodivfree = False, compress = True, truecompile = False, \
                 pq_reg = 0, divdivpen = 0, bonus_intorder_rhs = 0, stab_alpha = 6.0, vhat_outlet_diri = False):

        super(HDG, self).__init__(settings, order, compress, truecompile, pq_reg, divdivpen, bonus_intorder_rhs)

        self.hodivfree = hodivfree
        self.alpha = stab_alpha
        self.vhat_outlet_diri = vhat_outlet_diri

        if not self.sym:
            raise Exception("HDG with full grad not implemented! (need to remove W space)")

        # for eps-eps order 1, take W/R in RT0 and add
        self.WHDiv = self.sym and (self.order == 1)

        # BDM
        self.V = ngs.HDiv(settings.mesh, order = self.order, RT = False, hodivfree = self.hodivfree, \
                          dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.wall_slip, \
                          definedon = self.settings.fluid_domain)

        vh_diri = settings.inlet + "|" + settings.wall_noslip
        if self.vhat_outlet_diri:
            vh_diri = vh_diri + "|" + settings.outlet
        self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order, dirichlet = vh_diri,
                                               definedon = self.settings.fluid_domain, highest_order_dc = True)

        self.Q = ngs.L2(settings.mesh, order = 0 if self.hodivfree else order-1, \
                        definedon = self.settings.fluid_domain)

        if self.WHDiv:
            self.W = ngs.HDiv(settings.mesh, order = 0, definedon = self.settings.fluid_domain,
                              dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.outlet)
        else:
            self.W = None

        if self.compress and self.settings.fluid_domain != ".*":
                self.V = ngs.Compress(self.V)
                self.Vhat = ngs.Compress(self.Vhat)
                self.Q = ngs.Compress(self.Q)

    def SetUpForms(self, compound = False):

        self.dS = ngs.dx(element_vb = ngs.BND)

        self.h = ngs.specialcf.mesh_size
        self.n = ngs.specialcf.normal(self.settings.mesh.dim)
        self.normal = lambda u : (u * self.n) * self.n
        self.tang = lambda u : u - (u * self.n) * self.n
        if self.settings.mesh.dim == 2:
            self.Skew2Vec = lambda m :  m[1, 0] - m[0, 1]
        else:
            # then W = curl(u)
            self.Skew2Vec = lambda m : -0.5 * ngs.CoefficientFunction( (m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0]) )
        self.Eps  = lambda u : 0.5 * (ngs.Grad(u) + ngs.Grad(u).trans)
        self.Curl = lambda u : ngs.CoefficientFunction( (ngs.Grad(u)[2,1]-ngs.Grad(u)[1,2], \
                                                         ngs.Grad(u)[0,2]-ngs.Grad(u)[2,0], \
                                                         ngs.Grad(u)[1,0]-ngs.Grad(u)[0,1]) )

        if self.sym:
            self.X = ngs.ProductSpace(self.V, self.Vhat, self.W)
        else:
            self.X = ngs.ProductSpace(self.V, self.Vhat)

        if compound:
            self.Xext = ngs.ProductSpace(self.X, self.Q)
            if self.WHDiv:
                (u, uhat, W), p = self.Xext.TrialFunction()
                (v, vhat, R), q = self.Xext.TestFunction()
            else:
                (u, uhat), p = self.Xext.TrialFunction()
                (v, vhat), q = self.Xext.TestFunction()
        else:
            self.Xext = None
            if self.WHDiv:
                u, uhat, omega = self.X.TrialFunction()
                v, vhat, eta = self.X.TestFunction()
            else:
                u, uhat = self.X.TrialFunction()
                v, vhat = self.X.TestFunction()
            p,q = self.Q.TnT()

        Du = self.Eps(u) if self.sym else ngs.Grad(u)
        Dv = self.Eps(v) if self.sym else ngs.Grad(v)
        self.a_vol = self.nueff * ngs.InnerProduct(Du,Dv)
        self.a_bnd = self.nueff * ( self.alpha/self.h *  ngs.InnerProduct(self.tang(uhat-u), self.tang(vhat-v)) \
                                    + ngs.InnerProduct(Du * self.n, self.tang(vhat-v)) \
                                    + ngs.InnerProduct(Dv * self.n, self.tang(uhat-u)) )
        if self.WHDiv:
            self.a_bnd += self.nueff * self.h *  ngs.InnerProduct( (self.Curl(u) - omega) * self.n , (self.Curl(v) - eta) * self.n )
        if self.settings.l2_coef is not None:
            slef.a_vol += self.settings.l2_coef * u * v * dx
        self.b_vol = -ngs.div(u) * q
        self.bt_vol = -ngs.div(v) * p
        self.divdiv = ngs.div(u) * ngs.div(v)
        self.uv = ngs.InnerProduct(u, v)
        self.uhvh = ngs.InnerProduct(self.tang(uhat), self.tang(vhat))

        if self.pq_reg != 0:
            self.c_vol = -self.pq_reg * p * q
        else:
            self.c_vol = None

        if self.settings.vol_force is not None:
            self.f_vol = self.settings.vol_force * v
        else:
            self.f_vol = None

        self.rhs_outlet = None

        if self.settings.vel_outlet_f is not None:
            self.rhs_outlet = ngs.InnerProduct(self.settings.vel_outlet_f, v.Trace())

        if self.settings.vel_outlet_f is not None:
            if self.rhs_outlet == None:
                self.rhs_outlet  = ngs.InnerProduct(self.settings.velhat_outlet_f, vhat.Trace())
            else:
                self.rhs_outlet += ngs.InnerProduct(self.settings.velhat_outlet_f, vhat.Trace())

        # self.f_bnd = self.settings.uin * v
        self._mass_int = u*v*ngs.dx + self.h * uhat*vhat*self.dS

    # utility
    def XMass(self):
        return self._mass_int

    def GetPhysicalQuantities(self, gfs):
        if type(gfs) != list:  # ( (V,Vh,W), Q ) ->  (V,Vh,W), Q
            return self.GetPhysicalQuantities([gfs.components[0], gfs.components[1]])
        gfu = gfs[0]
        gfp = gfs[1]
        vel = gfu.components[0]
        vh = gfu.components[1]
        sigma = None
        eta = gfu.components[2] if self.sym else None
        p = gfp
        return vel, vh, sigma, eta, p

    def H1AEmbedding(self, V, elint):
        emb0 = ngs.comp.ConvertOperator(spacea = V, spaceb = self.V, \
                                        range_dofs = self.V.FreeDofs(elint), \
                                        localop = True, parmat = False, bonus_intorder_ab = 2)
        tc0 = self.X.Embedding(0).local_mat
        emb1 = ngs.comp.ConvertOperator(spacea = V, spaceb = self.Vhat, \
                                        range_dofs = self.Vhat.FreeDofs(elint), \
                                        localop = True, parmat = False, bonus_intorder_ab = 2)
        tc1 = self.X.Embedding(1).local_mat
        embA = tc0 @ emb0 + tc1 @ emb1
        if self.WHDiv:
            emb2 = ngs.comp.ConvertOperator(spacea = V, spaceb = self.W, trial_cf = self.Curl(V.TrialFunction()), \
                                            range_dofs = self.W.FreeDofs(elint), localop = True, \
                                            parmat = False, bonus_intorder_ab = 2)
            tc2 = self.X.Embedding(2).local_mat
            embA = embA + tc2 @ emb2
        if V.mesh.comm.size > 1:
            embA = ngs.ParallelMatrix(embA, row_pardofs = V.ParallelDofs(), col_pardofs = self.X.ParallelDofs(), \
                                      op = ngs.ParallelMatrix.C2C)
        return embA

### END HDG Discretization ###


### MCS Discretization ###

class MCS(FlowDiscretization):
    def __init__(self, settings = None, order = 2, RT = False, hodivfree = False, compress = True, truecompile = False, \
                 pq_reg = 0, divdivpen = 0, trace_sigma = False, bonus_intorder_rhs = 0, vhat_outlet_diri = False):
        super(MCS, self).__init__(settings, order, compress, truecompile, pq_reg, divdivpen, bonus_intorder_rhs)

        self.hodivfree = hodivfree
        self.RT = RT
        self.trace_sigma = trace_sigma
        self.vhat_outlet_diri = vhat_outlet_diri

        # for eps-eps order 1, take W/R in RT0 and add
        # h**2 div(W)*div(R) to the BLF to achieve stability
        self.WHDiv = self.sym and (self.order == 1)

        # !! EITHER vhat=0 OR sigma_nt=0 on outlet!!
        vh_diri = settings.inlet + "|" + settings.wall_noslip
        if self.vhat_outlet_diri:
            vh_diri = vh_diri + "|" + settings.outlet

        # Spaces
        self.V = ngs.HDiv(settings.mesh, order = self.order, RT = self.RT, hodivfree = self.hodivfree, \
                          dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.wall_slip, \
                          definedon = self.settings.fluid_domain)
        if self.RT:
            if True: # "correct" version
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = order, dirichlet = vh_diri, \
                                                       definedon = self.settings.fluid_domain)
            else: # not "correct", but works with facet-aux
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = order, dirichlet = vh_diri, \
                                                       definedon = self.settings.fluid_domain)
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = order, GGBubbles = True, discontinuous = True, \
                                      ordertrace = self.order if self.trace_sigma else -1, \
                                      definedon = self.settings.fluid_domain)
            # self.Sigma = ngs.HCurlDiv(mesh, order=order + 1, discontinuous=True, ordertrace=order) # slower I think
            self.Q = ngs.L2(settings.mesh, order = 0 if self.hodivfree else order, \
                            definedon = self.settings.fluid_domain)
            raise Exception("AAA")
        else:
            if True: # "correct" version
                # self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       # dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.outlet)
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, dirichlet = vh_diri, \
                                                       definedon = self.settings.fluid_domain)
            else: # works with facet-aux
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, dirichlet = vh_diri, \
                                                       definedon = self.settings.fluid_domain)
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, \
                                      discontinuous = True, ordertrace = self.order-1 if self.trace_sigma else -1, \
                                        definedon = self.settings.fluid_domain)
            # self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True)
            # self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True)
            self.Q = ngs.L2(settings.mesh, order = 0 if self.hodivfree else order-1, \
                            definedon = self.settings.fluid_domain)

        if self.sym:
            if self.WHDiv:
                self.W = ngs.HDiv(settings.mesh, order = 0, definedon = self.settings.fluid_domain,
                                  dirichlet = settings.inlet + "|" + settings.wall_noslip)
            else:
                if settings.mesh.dim == 2:
                    self.W = ngs.L2(settings.mesh, order=self.order if self.RT else order-1, \
                                definedon = self.settings.fluid_domain)
                else:
                    self.W = ngs.VectorL2(settings.mesh, order=self.order if self.RT else order-1, \
                                definedon = self.settings.fluid_domain)
        else:
            self.W = None

        if self.compress:
            self.Sigma.SetCouplingType(ngs.IntRange(0, self.Sigma.ndof), ngs.COUPLING_TYPE.HIDDEN_DOF)
            self.Sigma = ngs.Compress(self.Sigma)
            if self.sym and not self.WHDiv:
                self.W.SetCouplingType(ngs.IntRange(0, self.W.ndof), ngs.COUPLING_TYPE.HIDDEN_DOF)
                self.W = ngs.Compress(self.W)
            if self.settings.fluid_domain != ".*":
                self.V = ngs.Compress(self.V)
                self.Vhat = ngs.Compress(self.Vhat)
                self.Q = ngs.Compress(self.Q)

    def SetUpForms(self, compound = False):

        self.dS = ngs.dx(element_vb = ngs.BND)

        self.h = ngs.specialcf.mesh_size
        self.n = ngs.specialcf.normal(self.settings.mesh.dim)
        self.normal = lambda u : (u * self.n) * self.n
        self.tang = lambda u : u - (u * self.n) * self.n
        if self.settings.mesh.dim == 2:
            self.Skew2Vec = lambda m :  m[1, 0] - m[0, 1]
        else:
            # then W = curl(u)
            self.Skew2Vec = lambda m : -0.5 * ngs.CoefficientFunction( (m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0]) )
        self.Eps  = lambda u : 0.5 * (ngs.Grad(u) + ngs.Grad(u).trans)
        self.Curl = lambda u : ngs.CoefficientFunction( (ngs.Grad(u)[2,1]-ngs.Grad(u)[1,2], \
                                                         ngs.Grad(u)[0,2]-ngs.Grad(u)[2,0], \
                                                         ngs.Grad(u)[1,0]-ngs.Grad(u)[0,1]) )

        if self.sym:
            self.X = ngs.ProductSpace(self.V, self.Vhat, self.Sigma, self.W)
        else:
            self.X = ngs.ProductSpace(self.V, self.Vhat, self.Sigma)

        if compound:
            self.Xext = ngs.ProductSpace(self.X, self.Q)
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
                if False:
                    # this way W, R scale like sigma, tau (for numerical stability)
                    W = 1/self.h**self.settings.mesh.dim * W
                    R = 1/self.h**self.settings.mesh.dim * R
            else:
                u, uhat, sigma = self.X.TrialFunction()
                v, vhat, tau = self.X.TestFunction()
            p,q = self.Q.TnT()

        self.a_vol = -1.0 / self.nueff * ngs.InnerProduct(sigma, tau) \
                     + ngs.div(sigma) * v \
                     + ngs.div(tau) * u
        if self.ddp != 0.0:
            self.a_vol += self.ddp * self.nueff * ngs.div(u)*ngs.div(v)
        if not self.trace_sigma:
            self.a_vol += 1.0/self.settings.mesh.dim * self.nueff * ngs.div(u) * ngs.div(v)
        if self.sym:
           self.a_vol += ngs.InnerProduct(W, self.Skew2Vec(tau)) \
                         + ngs.InnerProduct(R, self.Skew2Vec(sigma))
           # makes the formulation stable for order 1
           if self.WHDiv:
               self.a_vol += self.nueff * self.h**2 * ngs.div(W) * ngs.div(R)
        self.a_bnd = - ((sigma * self.n) * self.n) * (v * self.n) \
                       - ((tau * self.n) * self.n) * (u * self.n) \
                       - (sigma * self.n) * self.tang(vhat) \
                       - (tau * self.n) * self.tang(uhat)
        if self.settings.l2_coef is not None:
            raise Exception("Sorry, not sure l2 coef is implemented correctly???")
            self.a_vol += self.settings.l2_coef * ngs.InnerProduct(u, v)
            self.a_bnd += self.settings.l2_coef * ngs.InnerProduct(self.tang(uhat), self.tang(vhat))
        self.b_vol = -ngs.div(u) * q
        self.bt_vol = -ngs.div(v) * p

        # This is useful in some cases.
        self.divdiv = ngs.div(u) * ngs.div(v)
        self.uv = ngs.InnerProduct(u, v)
        self.uhvh = ngs.InnerProduct(self.tang(uhat), self.tang(vhat))

        if self.pq_reg != 0:
            self.c_vol = -self.pq_reg * p * q
        else:
            self.c_vol = None

        if self.settings.vol_force is not None:
            self.f_vol = self.settings.vol_force * v
        else:
            self.f_vol = None

        self.rhs_outlet = None

        if self.settings.vel_outlet_f is not None:
            self.rhs_outlet = ngs.InnerProduct(self.settings.vel_outlet_f, v.Trace())

        if self.settings.vel_outlet_f is not None:
            if self.rhs_outlet == None:
                self.rhs_outlet  = ngs.InnerProduct(self.settings.velhat_outlet_f, vhat.Trace())
            else:
                self.rhs_outlet += ngs.InnerProduct(self.settings.velhat_outlet_f, vhat.Trace())

        # self.f_bnd = self.settings.uin * v
        self._mass_int = u*v*ngs.dx + self.h * uhat*vhat*self.dS + ngs.InnerProduct(sigma, tau) * ngs.dx

    # utility
    def XMass(self):
        return self._mass_int

    def GetPhysicalQuantities(self, gfs):
        if type(gfs) != list:  # ( (V,Vh,S,W), Q ) ->  (V,Vh,S,W), Q
            return self.GetPhysicalQuantities([gfs.components[0], gfs.components[1]])
        gfu = gfs[0]
        gfp = gfs[1]
        vel = gfu.components[0]
        vh = gfu.components[0]
        sigma = gfu.components[2]
        eta = gfu.components[3] if self.sym else None
        p = gfp
        return vel, vh, sigma, eta, p

    def H1AEmbedding(self, V, elint):
        emb0 = ngs.comp.ConvertOperator(spacea = V, spaceb = self.V, \
                                        range_dofs = self.V.FreeDofs(elint), \
                                        localop = True, parmat = False, bonus_intorder_ab = 2)
        tc0 = self.X.Embedding(0).local_mat
        emb1 = ngs.comp.ConvertOperator(spacea = V, spaceb = self.Vhat, \
                                        range_dofs = self.Vhat.FreeDofs(elint), \
                                        localop = True, parmat = False, bonus_intorder_ab = 2)
        tc1 = self.X.Embedding(1).local_mat
        embA = tc0 @ emb0 + tc1 @ emb1
        if not self.compress:
            sig = -self.nueff * self.Eps(V.TrialFunction())
            emb2 = ngs.comp.ConvertOperator(spacea = V, spaceb = self.Sigma, trial_cf = sig, \
                                            range_dofs = self.Sigma.FreeDofs(elint), \
                                            localop = True, parmat = False, bonus_intorder_ab = 2)
            tc2 = self.X.Embedding(2).local_mat
            embA = embA + tc2 @ emb2
        if self.WHDiv:
            emb3 = ngs.comp.ConvertOperator(spacea = V, spaceb = self.W, trial_cf = self.Curl(V.TrialFunction()), \
                                            range_dofs = self.W.FreeDofs(elint), localop = True, \
                                            parmat = False, bonus_intorder_ab = 2)
            tc3 = self.X.Embedding(3).local_mat
            embA = embA + tc3 @ emb3
        if V.mesh.comm.size > 1:
            embA = ngs.ParallelMatrix(embA, row_pardofs = V.ParallelDofs(), col_pardofs = self.X.ParallelDofs(), \
                                      op = ngs.ParallelMatrix.C2C)
        return embA



### END MCS Discretization ###


### Stokes Template ###

class StokesTemplate():

    class LinAlg():
        def __init__(self, stokes, pc_ver = "aux", pc_opts = dict(), elint = False):

            self.comm = stokes.settings.mesh.comm

            self.elint = elint
            ## with static condensation + hodivfree, we can iterate on the schur complement
            self.it_on_sc = self.elint and stokes.disc.hodivfree
            # self.it_on_sc = True# self.elint and stokes.disc.hodivfree

            self.need_bp_scale = True

            self.block_la = pc_ver != "direct"

            self.pc_avail = { "direct"      : lambda astokes, opts : self.PCSetUpDirect(astokes, **opts),
                              "block"       : lambda astokes, opts : self.PCSetUpBlock(astokes, **opts),
                              "none"        : lambda astokes, opts : self.PCSetUpDummy(astokes) }

            # preconditionsers that set up from BLF
            self.pc_a_avail_pre = { "stokesAMG" : lambda astokes, opts : self.SetUpHDivStokesAMG(astokes, **opts) }

            # preconditionsers that set up from assembled matrix
            self.pc_a_avail_post = { "direct"       : lambda astokes, opts : self.SetUpADirect(astokes, **opts),
                                     "auxh1"        : lambda astokes, opts : self.SetUpAAux(astokes, **opts),
                                     "auxStokesAMG"    : lambda astokes, opts : self.SetUpAuxStokesAMG(astokes, **opts), }

            # make sure input is OK
            if self.block_la:
                if not "a_opts" in pc_opts:
                    pc_opts["a_opts"] = dict
                if not "type" in pc_opts["a_opts"]:
                    pc_opts["a_opts"]["type"] = direct

                aver = pc_opts["a_opts"]["type"]

                if not aver in self.pc_a_avail_pre and not aver in self.pc_a_avail_post:
                    raise Exception(f"Invalid A-block preconditioner type {aver}")

            # define forms & physical quantities 
            self.preSetUp(stokes)

            # define A-preconditioner if it is set up from unassembled BLF
            if self.block_la and ( aver in self.pc_a_avail_pre ):
                    self.pc_a_avail_pre[aver](stokes, pc_opts)
                    
            # assemble stuff
            self.Assemble()

            # harmonic extension for assembled operators
            self.HexFWOps()

            # preconditioner setup
            if not pc_ver in self.pc_avail:
                raise Exception("invalid PC version!")
            else:
                self.pc_avail[pc_ver](stokes, pc_opts)

            # harmonic extenstion for precond
            self.HexBWOps()

        def preSetUp(self, stokes):
            # forms
            stokes.disc.SetUpForms(compound = not self.block_la)

            # physical quantities
            if self.block_la:
                self.gfu = ngs.GridFunction(stokes.disc.X)
                self.gfp = ngs.GridFunction(stokes.disc.Q)
                self.velocity, self.velhat, self.sigma, \
                    self.eta, self.pressure = stokes.disc.GetPhysicalQuantities([self.gfu, self.gfp])
                self.sol_vec = ngs.BlockVector([self.gfu.vec, \
                                                self.gfp.vec])
            else:
                self.gfu = ngs.GridFunction(stokes.disc.Xext)
                self.velocity, self.velhat, self.sigma, \
                    self.eta, self.pressure = stokes.disc.GetPhysicalQuantities(self.gfu)
                self.sol_vec = self.gfu.vec

            self._to_assemble = []

            self.SetUpFWOps(stokes)
            self.SetUpRHS(stokes)


        def SetUpFWOps(self, stokes):
            # Forward operators
            if self.block_la:
                self.a = ngs.BilinearForm(stokes.disc.X, condense = self.elint, eliminate_hidden = stokes.disc.compress, \
                                          store_inner = self.elint and not self.it_on_sc, elmatev = False)
                self.a += stokes.disc.stokesA()
                # self.a += 1e2 * stokes.settings.nu * stokes.disc.divdiv * ngs.dx

                if False and stokes.disc.ddp != 0: # ddp is now in a_vol already
                    u, uhat, sigma = stokes.disc.X.TrialFunction()
                    v, vhat, tau = stokes.disc.X.TestFunction()
                    self.a2 = ngs.BilinearForm(stokes.disc.X, condense = self.elint, eliminate_hidden = stokes.disc.compress, \
                                               store_inner = self.elint and not self.it_on_sc)
                    self.a2 += stokes.disc.stokesA()
                    self.a2 += stokes.disc.ddp * stokes.settings.nu * stokes.disc.divdiv * ngs.dx
                    # what is thjis... this does not even make sense, it needs a surface-int
                    # self.a2 += 1e2/stokes.disc.h * stokes.settings.nu * ngs.InnerProduct(stokes.disc.normal(u), stokes.disc.normal(v)) * ngs.dx
                    self._to_assemble += [ ("a2", self.a2)  ]
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

                self._to_assemble += [ ("a",self.a), ("b",self.b), ("c",self.c) ]

                # if stokes.settings.sym:
                #     u, uhat, _, _ = stokes.disc.X.TrialFunction()
                #     v, vhat, _, _ = stokes.disc.X.TestFunction()
                # else:
                #     u, uhat, _ = stokes.disc.X.TrialFunction()
                #     v, vhat, _ = stokes.disc.X.TestFunction()
                # self.rblf_x = ngs.BilinearForm(stokes.disc.X)
                # self.rblf_x += ngs.InnerProduct(u,v) * ngs.dx
                # self.rblf_x += 1/stokes.disc.h * ngs.InnerProduct(uhat,vhat) * ngs.dx(element_vb=ngs.BND)
                # self._to_assemble.append(self.rblf_x)
                # p, q = stokes.disc.Q.TnT()
                # self.rblf_q = ngs.BilinearForm(stokes.disc.Q)
                # self.rblf_q += ngs.InnerProduct(p,q) * ngs.dx
                # self._to_assemble.append(self.rblf_q)

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

                self._to_assemble += [ ("m", self.m) ]
                self.a = self.m
                self.a2 = self.m


        def SetUpRHS(self, stokes):
            # Right hand side
            if self.block_la:
                self.f = ngs.LinearForm(stokes.disc.X)
                if stokes.disc.stokesf() is not None:
                    self.f += stokes.disc.stokesf()

                self.g = ngs.LinearForm(stokes.disc.Q)

                self._to_assemble += [ ("f", self.f), ("g", self.g) ]

            else:
                self.f = ngs.LinearForm(stokes.disc.Xext)
                if stokes.disc.stokesf() is not None:
                    self.f += stokes.disc.stokesf()

                self._to_assemble += [ ("F", self.f) ]


        def Assemble(self):
            for name, x in self._to_assemble:
                if x is not None:
                    if ngs.mpi_world.rank == 0:
                        print("\n===\nassemble {}\n===".format(name))
                        sys.stdout.flush()
                    t = ngs.Timer("assemble {}".format(name))
                    x.space.mesh.comm.Barrier()
                    t.Start()
                    x.Assemble()
                    x.space.mesh.comm.Barrier()
                    t.Stop()
                    if ngs.mpi_world.rank == 0:
                        print("\n===\nassembling {} took {} sec\n===".format(name, t.time))
                        sys.stdout.flush()


        def HexFWOps(self):
            if self.block_la:
                ## Even when using static condensation, we still need to iterate on the entire A matrix,
                ## except when we are also using hodivfree
                self.A = self.a.mat

                if self.elint and not self.it_on_sc:
                    Ahex, Ahext, Aii  = self.a.harmonic_extension.local_mat, self.a.harmonic_extension_trans.local_mat, \
                                        self.a.inner_matrix.local_mat
                    # print("Ahex  \n", Ahex)
                    # print("Ahext \n", Ahext)
                    # print("Aii   \n", Aii)
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
                    Mhex, Mhext, Mii  = self.m.harmonic_extension.local_mat, self.m.harmonic_extension_trans.local_mat, \
                                        self.m.inner_matrix.local_mat
                    Id = ngs.IdentityMatrix(self.M.height)
                    self.M = (Id - Mhext) @ (self.M.local_mat + Mii) @ (Id - Mhex)
                    if self.m.space.mesh.comm.size > 1:
                        self.M = ngs.ParallelMatrix(self.Mext, row_pardofs = self.m.mat.row_pardofs, \
                                                       col_pardofs = self.m.mat.col_pardofs, op = ngs.ParallelMatrix.C2D)
                self.rhs_vec = self.f.vec

        def HexBWOps(self):
            if self.block_la:
                if self.elint and not self.it_on_sc:
                    Ahex, Ahext, Aiii  = self.a2.harmonic_extension.local_mat, self.a2.harmonic_extension_trans.local_mat, \
                                         self.a2.inner_solve.local_mat
                    # print("II Ahex  \n", Ahex)
                    # print("II Ahext \n", Ahext)
                    # print("II Aiii   \n", Aiii)
                    Id = ngs.IdentityMatrix(self.A.height)
                    # Id = ngs.Projector(self.a2.space.FreeDofs(self.elint), True)
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

                # lams = list(ngs.la.EigenValues_Preconditioner(mat=self.A, pre=self.Apre, tol=1e-6))
                # self.Apre = 1.0/(lams2[1] * 1.1) * self.Apre

                # S = self.B @ self.Apre @ self.BT
                # lams2 = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=self.Spre, tol=1e-6))
                # self.Spre = 1.0/(lams2[0] * 0.9) * self.Spre


        def PrepRHS(self, rhs_vec):
            if self.elint and self.it_on_sc:
                rv = rhs_vec[0] if self.block_la else rhs_vec
                rv.Distribute()
                rv.local_vec.data += self.a.harmonic_extension_trans.local_mat * rv.local_vec


        def ExtendSol(self, sol_vec, rhs_vec):
            if self.elint and self.it_on_sc:
                sv = sol_vec[0] if self.block_la else sol_vec
                rv = rhs_vec[0] if self.block_la else rhs_vec
                rv.Distribute()
                sv.Cumulate()
                sv.local_vec.data += self.a.inner_solve.local_mat * rv.local_vec
                sv.local_vec.data += self.a.harmonic_extension.local_mat * sv.local_vec


        def PCSetUpDirect(self, stokes, inv_type = None, **kwargs):
            # Direct inverse
            if self.block_la:
                raise Exception("Cannot invert block matrices!")
            else:
                itype = "umfpack" if inv_type is None else inv_type
                self.Mpre = self.m.mat.Inverse(stokes.disc.Xext.FreeDofs(self.elint), inverse = itype)
                if self.elint and not self.it_on_sc:
                    Mhex, Mhext, Miii  = self.m.harmonic_extension.local_mat, self.m.harmonic_extension_trans.local_mat, \
                                         self.m.inner_solve
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


        def PCSetUpBlock(self, stokes, a_opts = { "type" : "direct" } , **kwargs):
            if not self.block_la:
                raise Exception("block-PC with big compond space todo")
            else:
                p,q = stokes.disc.Q.TnT()
                self.massp = ngs.BilinearForm(stokes.disc.Q)
                if stokes.disc.stokesC() is not None:
                    self.massp += -1 * stokes.disc.stokesC()
                if stokes.disc.ddp == 0:
                    self.massp +=  1/stokes.disc.nueff * p * q * ngs.dx
                else:
                    self.massp +=  1/(stokes.disc.nueff * (1 + stokes.disc.ddp)) * p * q * ngs.dx
                # self.Spre = ngs.Preconditioner(self.massp, "direct")
                self.Spre = ngs.Preconditioner(self.massp, "local")

                self.massp.Assemble()

                aver = a_opts["type"] if "type" in a_opts else "direct"
                if aver in self.pc_a_avail_post:
                    self.pc_a_avail_post[aver](stokes, a_opts)
                else:
                    raise Exception("invalid pc type for A block!")

                self.ASpre = self.Apre


        def PCSetUpDummy(self, stokes, **kwargs):
            self.Mpre = ngs.Projector(stokes.disc.Xext.FreeDofs(self.elint), True)


        def SetUpADirect (self, stokes, inv_type = None, **kwargs):
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


        def SetUpAuxSpaceBLF (self, stokes, V, ddp = 0.0):
            # Auxiliary space Problem
            u, v = V.TnT()
            a_aux = ngs.BilinearForm(V)
            if stokes.settings.sym:
                eps = lambda U : 0.5 * (ngs.Grad(U) + ngs.Grad(U).trans)
                a_aux += 2 * stokes.settings.nu * ngs.InnerProduct(eps(u), eps(v)) * ngs.dx
            else:
                a_aux += stokes.settings.nu * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * ngs.dx
            if stokes.settings.l2_coef is not None:
                a_aux += stokes.settings.l2_coef * ngs.InnerProduct(u, v) * ngs.dx
            # only add a penalty if uhat is dirichlet, otherwise natural BCs apply!
            if len(stokes.settings.outlet) > 0 and stokes.disc.vhat_outlet_diri:
                a_aux += 5*10*stokes.disc.order**2 * stokes.settings.nu/stokes.disc.h * \
                         ngs.InnerProduct(stokes.disc.tang(u), stokes.disc.tang(v)) * \
                         ngs.ds(definedon=stokes.settings.mesh.Boundaries(stokes.settings.outlet))
            if len(stokes.settings.wall_slip) > 0:
                a_aux += 5*10*stokes.settings.nu/stokes.disc.h * \
                         ngs.InnerProduct(stokes.disc.normal(u), stokes.disc.normal(v)) * \
                         ngs.ds(definedon=stokes.settings.mesh.Boundaries(stokes.settings.wall_slip))
            if ddp != 0.0:
                if stokes.settings.mesh.dim == 2:
                    my_div = lambda X : ngs.Grad(X)[0,0] + ngs.Grad(X)[1,1]
                else:
                    my_div = lambda X : ngs.Grad(X)[0,0] + ngs.Grad(X)[1,1] + ngs.Grad(X)[2,2]
                a_aux += ddp * stokes.settings.nu * my_div(u) * my_div(v) * ngs.dx
            return a_aux


        def AuxToMCSEmbedding (self, stokes, Vaux, localop=True, allops=False, projectN=False, asSparse=False, projectToFree=True):
            u, v = stokes.disc.V.TnT()

            if ngs.mpi_world.rank == 0:
                print("\n===\nCONVERT 0")
                sys.stdout.flush()

            useLocalOp = not asSparse

            if projectN and (stokes.disc.order > 0):
                print(" -> PROJECT V-AUX TO ORDER 0!")
                sys.stdout.flush()

                Vlo = ngs.HDiv(stokes.settings.mesh, order=0, dirichlet = stokes.settings.wall_noslip + "|" + stokes.settings.inlet)

                aux_lo = ngs.comp.ConvertOperator(spacea = Vaux, spaceb = Vlo, localop = useLocalOp, parmat = False, bonus_intorder_ab = 2,
                                                  range_dofs = Vlo.FreeDofs() if projectToFree else None)
                lo_ho = ngs.comp.ConvertOperator(spacea = Vlo, spaceb = stokes.disc.V, localop = useLocalOp, parmat = False, bonus_intorder_ab = 2,
                                                  range_dofs = stokes.disc.V.FreeDofs(self.elint) if projectToFree else None)
                emb0 =  lo_ho @ aux_lo
            else:
                emb0 = ngs.comp.ConvertOperator(spacea = Vaux, spaceb = stokes.disc.V, localop = localop and useLocalOp, parmat = False, bonus_intorder_ab = 2,
                                                range_dofs = stokes.disc.V.FreeDofs(self.elint) if projectToFree else None)

            tc0 = stokes.disc.X.Embedding(0).local_mat
            if ngs.mpi_world.rank == 0:
                print("\n===\nCONVERT 1")
                sys.stdout.flush()
            emb1 = ngs.comp.ConvertOperator(spacea = Vaux, spaceb = stokes.disc.Vhat, localop = localop and useLocalOp, parmat = False, bonus_intorder_ab = 2,
                                            range_dofs = stokes.disc.Vhat.FreeDofs(self.elint) if projectToFree else None)
            tc1 = stokes.disc.X.Embedding(1).local_mat
            embA = tc0 @ emb0 + tc1 @ emb1

            if not stokes.disc.compress:
                G2E = lambda G : 0.5 * (G + G.trans)
                sig = -stokes.disc.nueff * G2E(ngs.Grad(Vaux.TrialFunction()))
                if ngs.mpi_world.rank == 0:
                    print("\n===\nCONVERT 2")
                    sys.stdout.flush()
                emb2 = ngs.comp.ConvertOperator(spacea = Vaux, spaceb = stokes.disc.Sigma, localop = useLocalOp, parmat = False, bonus_intorder_ab = 2,
                                                range_dofs = stokes.disc.Sigma.FreeDofs(self.elint) if projectToFree else None, trial_cf = sig)
                tc2 = stokes.disc.X.Embedding(2).local_mat
                print("ADD EMB2")
                embA = embA + tc2 @ emb2
            if stokes.disc.WHDiv:
                n = ngs.specialcf.normal(3)
                tang = lambda u : u - (u * n) * n
                # Skew2Vec = lambda m : -0.5 * ngs.CoefficientFunction( (m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0]) )
                G2Cu = lambda G : ngs.CoefficientFunction( (G[2,1]-G[1,2], G[0,2]-G[2,0], G[1,0]-G[0,1]) )
                # G2Cu = lambda G : - 2 * Skew2Vec(G)
                cu = G2Cu(ngs.Grad(V.TrialFunction()))
                # V is H1, so curl is in HDiv -> localop=True and C2C is fine!
                if ngs.mpi_world.rank == 0:
                    print("\n===\nCONVERT 3")
                    sys.stdout.flush()
                emb3 = ngs.comp.ConvertOperator(spacea = Vaux, spaceb = stokes.disc.S, localop = useLocalOp, parmat = False, bonus_intorder_ab = 2,
                                                range_dofs = stokes.disc.S.FreeDofs(self.elint) if projectToFree else None, trial_cf = cu)
                tc3 = stokes.disc.X.Embedding(3).local_mat
                print("ADD EMB3")
                embA = embA + tc3 @ emb3
            if Vaux.mesh.comm.size > 1:
                embA = ngs.ParallelMatrix(embA, row_pardofs = Vaux.ParallelDofs(), col_pardofs = stokes.disc.X.ParallelDofs(),
                                          op = ngs.ParallelMatrix.C2C if localop else ngs.ParallelMatrix.C2D)
            if ngs.mpi_world.rank == 0:
                print("\n===\nCONVERT DONE")
                sys.stdout.flush()
            if asSparse:
                embA = NgsAMG.ToSparseMatrix(embA)
            # embA = ngs.la.LoggingMatrix(embA, "par embA")
            if allops:
                return embA, tc1@emb1, tc2@emb2
            else:
                return embA


        def SetUpAAux (self, stokes, amg_package = "petsc", amg_opts = dict(), mpi_thrad = False, mpi_overlap = False, shm = None,
                       multiplicative = True, sm_el_blocks = False, aux_mlt = True, blk_smoother = True,
                       sm_nsteps = 1, sm_symm = False, sm_nsteps_loc = 1, sm_symm_loc = False, sm_mpi_thread = True, **kwargs):
            if stokes.disc.hodivfree:
                raise Exception("Sorry, Auxiliary space not available with hodivfree (dual shapes not implemented) !")
            use_petsc = amg_package == "petsc"
            aux_direct = amg_package == "direct"
            if use_petsc:
                if not _ngs_petsc:
                    raise Exception("NGs-PETSc interface not available!")
            elif not aux_direct:
                if not _ngsAMG:
                    raise Exception("NgsAMG not available!")

            # Auxiliary space
            if True:
                V = ngs.H1(stokes.settings.mesh, order = 1, dirichlet = stokes.settings.wall_noslip + "|" + stokes.settings.inlet, \
                           dim = stokes.settings.mesh.dim, definedon = stokes.settings.fluid_domain)
            else:
                V = ngs.VectorH1(stokes.settings.mesh, order = 1, dirichlet = stokes.settings.wall_noslip + "|" + stokes.settings.inlet, \
                                 definedon = stokes.settings.fluid_domain)
                # V = ngs.VectorH1(stokes.settings.mesh, order = 1, dirichletx = stokes.settings.wall_noslip + "|" + stokes.settings.inlet,
                                 # dirichlety = stokes.settings.wall_noslip + "|" + stokes.settings.inlet + "|" + stokes.settings.outlet)


            # Auxiliary space Problem
            a_aux = self.SetUpAuxSpaceBLF(stokes, V)

            # some options can be given as lambdas that depend on the space V!
            for opt, val in amg_opts.items():
                if callable(val):
                    amg_opts[opt] = val(V)

            # t = ngs.Timer("assemble AUX")
            # a_aux.space.mesh.comm.Barrier()
            # t.Start()
            # a_aux.Assemble()
            # a_aux.space.mesh.comm.Barrier()
            # t.Stop()
            # if ngs.mpi_world.rank == 0:
            #     print("\n===\nassembling NOPC AUX SPACE took {} sec\n===".format(t.time))
            #     sys.stdout.flush()

            t = ngs.Timer("AUX assemble+sup solver")
            a_aux.space.mesh.comm.Barrier()
            t.Start()

            if not aux_direct:
                if not use_petsc:
                    if stokes.settings.sym:
                        amg_cl = NgsAMG.elast_2d if stokes.settings.mesh.dim == 2 else NgsAMG.elast_3d
                    else:
                        amg_cl = NgsAMG.h1_2d if stokes.settings.mesh.dim == 2 else NgsAMG.h1_3d
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
                aux_pre = a_aux.mat.Inverse(V.FreeDofs(), inverse="mumps" if a_aux.space.mesh.comm.size>1 else "sparsecholesky")

            a_aux.space.mesh.comm.Barrier()
            t.Stop()
            if ngs.mpi_world.rank == 0:
                print("\n===\nassembling AUX SPACE took {} sec\n===".format(t.time))
                sys.stdout.flush()

            # Embeddig Auxiliary space -> MCS space
            # embA = self.AuxToMCSEmbedding(stokes, V)
            embA = stokes.disc.H1AEmbedding(V, self.elint)

            if aux_mlt:
                if V.mesh.comm.size>1 and not _ngsAMG:
                    raise Exception("MPI-parallel multiplicative block-smoothers only available with NgsMPI!")
                if sm_el_blocks:
                    raise Exception("multiplicative smoothing with element-blocks not availalbe with MPI!")
            if V.mesh.comm.size>1 and blk_smoother and sm_el_blocks:
                    raise Exception("element-blocks with MPI not implemented!")

            # Blocks for smoother to combine with auxiliary PC
            if blk_smoother:
                if V.mesh.comm.size==1 and sm_el_blocks:
                    sm_blocks = ["EL"]
                else:
                    sm_blocks = ["F", "C"][0:1 if self.elint else 2]
            else: # empty list -> GS smoother, not recommended
                sm_blocks = []

            # Auxiliary space PC (additive or multiplicative)
            self.Apre = AuxiliarySpacePreconditioner(self.a, aux_pre, embA, sm_blocks=[], multiplicative=True,
                                                     sm_nsteps=sm_nsteps, sm_nsteps_loc=sm_nsteps_loc,
                                                     sm_symm=sm_symm, sm_symm_loc=sm_symm_loc, elint=self.elint,
                                                     mpi_thread = sm_mpi_thread)


            ## END SetUpAAux ##


        def SetUpAFacet (self, stokes, amg_opts = dict(), **kwargs):
            if not _ngsAMG:
                raise Exception("Facet Auxiliary space Preconditioner only available with NgsAMG!")
            if stokes.settings.sym:
                if stokes.settings.mesh.dim == 2:
                    self.Apre = NgsAMG.mcs_epseps_2d(self.a, **amg_opts)
                else:
                    self.Apre = NgsAMG.mcs_epseps_3d(self.a, **amg_opts)
            else:
                if stokes.settings.mesh.dim == 2:
                    self.Apre = NgsAMG.mcs_gg_2d(self.a, **amg_opts)
                else:
                    self.Apre = NgsAMG.mcs_gg_3d(self.a, **amg_opts)
            self.a.Assemble() # <- TODO: is this necessary ??
            # print("AUX MAT: ", self.Apre.aux_mat)
            ## END SetUpAFacet ##


        def SetUpAuxStokesAMG (self, stokes, amg_opts = dict(), **kwargs):
            if not _ngsAMG:
                raise Exception("Stokes AMG only available with NgsAMG!")

            if not stokes.disc.hodivfree:
                raise Exception("Should set hodivfree=True with Stokes-AMG, hodivfree=False is untested!")

            dgj = False

            self.Vaux = NgsAMG.NoCoH1(stokes.settings.mesh, dim=stokes.settings.mesh.dim, \
                                        dirichlet = stokes.settings.wall_noslip + "|" + stokes.settings.inlet,
                                        dgjumps = dgj, definedon=stokes.settings.fluid_domain)
            if stokes.settings.fluid_domain != ".*":
                self.Vaux = ngs.Compress(self.Vaux)

            stokes.Vaux = self.Vaux


            # Whether to project the normal component to order 0 which makes the embedding facet-wise
            projectN = stokes.disc.order > 0
            useAuxSpaceBLF = True

            embA = self.AuxToMCSEmbedding(stokes, self.Vaux, localop=False, allops=False, projectN=projectN, asSparse=False, projectToFree=True)

            if True:
                amg_cl = NgsAMG.stokes_gg_2d if stokes.settings.mesh.dim == 2 else NgsAMG.stokes_gg_3d

                for opt, val in amg_opts.items():
                    if callable(val):
                        amg_opts[opt] = val(Vaux)

                if not "ngs_amg_energy" in amg_opts:
                    amg_opts["ngs_amg_energy"] = "alg"

                # Auxiliary space BLF without divergence penalty (for weights)
                use_nodiv = amg_opts["ngs_amg_energy"] == "alg" and stokes.disc.ddp != 0.0
                if use_nodiv:
                    a_aux_nodiv = self.SetUpAuxSpaceBLF(stokes, self.Vaux, ddp=0)
                    a_aux_nodiv.Assemble()

                if useAuxSpaceBLF: # assemble BLF in aux space

                    # Auxiliary space BLF
                    u, v = self.Vaux.TnT()
                    a_aux = self.SetUpAuxSpaceBLF(stokes, self.Vaux, ddp=stokes.disc.ddp)

                    # a_aux += 1/ngs.specialcf.mesh_size * ngs.InnerProduct(u - u.Other(), v - v.Other()) * ngs.dx(element_boundary=True)

                    n = ngs.specialcf.normal(stokes.settings.mesh.dim)
                    if dgj:
                        a_aux += 30 * stokes.settings.nu/ngs.specialcf.mesh_size * ( ((u - u.Other())*n) * ((v - v.Other())*n) ) * ngs.dx(element_boundary=True)

                    # a_aux += ngs.InnerProduct(u, v) * ngs.dx(element_boundary=True)
                    if use_nodiv:
                        aux_pre = amg_cl(a_aux, a_aux_nodiv.mat, **amg_opts)
                    else:
                        aux_pre = amg_cl(a_aux, **amg_opts)

                    t = ngs.Timer("aux-assemble")
                    a_aux.space.mesh.comm.Barrier()
                    t.Start()

                    a_aux.Assemble()

                    self.Vaux.mesh.comm.Barrier()
                    t.Stop()
                    if ngs.mpi_world.rank == 0:
                        print("\n===\nassembling AUX SPACE (STOKES AMG) took {} sec\n===".format(t.time))
                        sys.stdout.flush()

                    embAT = embA.T
                else: # project HDG/MCS matrix to aux space

                    # sparsify embedding
                    sparseEMb = NgsAMG.ToSparseMatrix(embA)

                    # also need this for low order because of Vhat
                    sparseEMb = NgsAMG.RestrictMatrixToBlocks(mat=sparseEMb,
                                                              row_blocks=MakeFacetBlocks(stokes.disc.X),
                                                              col_blocks=MakeFacetBlocks(self.Vaux),
                                                              tol=1e-6)

                    sparseEMbT = NgsAMG.ToSparseMatrix(sparseEMb.T)

                    auxMat = NgsAMG.ToSparseMatrix( sparseEMbT @ self.A @ sparseEMb)

                    aux_pre = amg_cl(fes=self.Vaux,
                                     freedofs=self.Vaux.FreeDofs(self.elint),
                                     mat=auxMat,
                                     weight_mat=a_aux_nodiv.mat,
                                     **amg_opts)

                    # This does not work with SPCST in AuxiliarySpacePreconditioner for some reason!
                    # embA  = sparseEMb
                    # embAT = sparseEMbT

                    embAT = embA.T
            else:
                aux_pre = a_aux.mat.Inverse(self.Vaux.FreeDofs(self.elint))

            # emb_c = ngs.comp.ConvertOperator(spacea=Vaux, spaceb=Vc, localop=False)
            # embA = self.AuxToMCSEmbedding(stokes, Vc, True) @ emb_c

            # embA, emb1, emb2 = self.AuxToMCSEmbedding(stokes, Vaux, False)
            # print("embA\n", embA)

            # massa = ngs.BilinearForm(stokes.disc.XMass()).Assemble()
            # u, v = Vaux.TnT()
            # massb = ngs.BilinearForm(u*v*ngs.dx).Assemble()
            # # massbi = massb.mat.CreateSmoother() # Inverse()
            # massbi = massb.mat.Inverse(inverse="sparsecholesky")
            # mpr = embA @ massbi @ embA.T
            # # mpr = 0.0 * emb1 @ massbi @ emb1.T + 0.2 * massa.mat.CreateSmoother() # Inverse()
            # # mpr = emb2 @ massbi @ emb2.T
            # # mpr = massa.mat.CreateSmoother()
            # evs_A = list(ngs.la.EigenValues_Preconditioner(mat=massa.mat, pre=mpr, tol=1e-14))
            # if self.a.space.mesh.comm.rank == 0:
            #     print("\n----")
            #     print("NC MASS TEST")
            #     print("--")
            #     print("min ev. preA\A:", evs_A[:5])
            #     print("max ev. preA\A:", evs_A[-5:])
            #     print("cond-nr preA\A:", evs_A[-1]/evs_A[0])
            # # quit()
            # quit()

            sm_blocks = ["facet", "cell"][0:1 if self.elint else 2]
            # sm_blocks = ["facet"]
            # self.Apre = embA @ aux_pre @ embA.T
            # self.Apre = embA@aux_pre@embA.T
            self.Apre = AuxiliarySpacePreconditioner(self.a2, aux_pre, embA, sm_blocks,
                                                     embeddingT=embAT, multiplicative=True,
                                                     elint=self.elint, sm_nsteps=2, sm_symm=False)


        def SetUpHDivStokesAMG (self, stoeks, amg_opts = dict(), **kwargs):
            if not _ngsAMG:
                raise Exception("Stokes AMG only available with NgsAMG!")

            if not stokes.disc.hodivfree:
                raise Exception("Should set hodivfree=True with Stokes-AMG, hodivfree=False is untested!")

            amg_cl = NgsAMG.stokes_hdiv_gg_2d if stokes.settings.mesh.dim == 2 else NgsAMG.stokes_hdiv_gg_3d

            amg_pc = amg_cl(self.a2, **amg_opts)

            self.Apre = amg_pc


        def TestBlock(self, exai = False):
            o_ms_l = ngs.ngsglobals.msg_level
            ngs.ngsglobals.msg_level = 0


            if self.elint and not self.it_on_sc:
                evs_AS = list(ngs.la.EigenValues_Preconditioner(mat=self.a.mat, pre=self.ASpre, tol=1e-8))
                if self.a.space.mesh.comm.rank == 0:
                    print("--")
                    print("Block-PC Condition number test")
                    print("EVs for condensed A block")
                    print("min ev. preA\A:", evs_AS[:5])
                    print("max ev. preA\A:", evs_AS[-5:])
                    print("cond-nr preA\A:", evs_AS[-1]/evs_AS[0])

            evs_A = list(ngs.la.EigenValues_Preconditioner(mat=self.A, pre=self.Apre, tol=1e-8))
            if self.a.space.mesh.comm.rank == 0:
                print("\n----")
                if not(self.elint and not self.it_on_sc):
                    print("Block-PC Condition number test")
                print("--")
                print("EVs for A block")
                print("min ev. preA\A:", evs_A[:5])
                print("max ev. preA\A:", evs_A[-5:])
                print("cond-nr preA\A:", evs_A[-1]/evs_A[0])
            # quit()

            if exai:
                if self.elint:
                    raise Exception("ex a inv for S test todo")
                ainv = self.a.mat.Inverse(self.a.space.FreeDofs(self.elint), inverse = "umfpack")
                S = self.B @ ainv @ self.B.T
            else:
                S = self.B @ self.Apre @ self.B.T

            evs_S = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=self.Spre, tol=1e-8))
            # evs_S = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=ngs.IdentityMatrix(S.height), tol=1e-14))
            evs0 = evs_S[0] if evs_S[0] > 1e-4 else evs_S[1]

            if self.a.space.mesh.comm.rank == 0:
                print("--")
                print("EVs for Schur Complement")
                print("min ev. preS\S:", evs_S[0:5])
                print("max ev. preS\S:", evs_S[-5:])
                print("cond-nr preS\S:", evs_S[-1]/(evs0))
                print("----\n")

            # self.Spre = 1.0/evs_S[-1] * self.Spre
            # self.Spre = 1 * self.Spre
            # self.Spre = 1.01/evs_S[0] * self.Spre
            # self.Spre = 0.99/evs_S[-1] * self.Spre
            # self.Spre = 0.5 * self.Spre

            evs_S = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=self.Spre, tol=1e-14))
            evs0 = evs_S[0] if evs_S[0] > 1e-4 else evs_S[1]

            if self.a.space.mesh.comm.rank == 0:
                print("--")
                print("EVs for Schur Complement")
                print("min ev. preS\S:", evs_S[0:5])
                print("max ev. preS\S:", evs_S[-5:])
                print("cond-nr preS\S:", evs_S[-1]/(evs0))
                print("----\n")
            # quit()

            ngs.ngsglobals.msg_level = o_ms_l

            return evs_A[-1]/evs_A[0], evs_S[-1]/evs0, evs_A, evs_S
            ## END TestBlock ##

        ## END LinAlg ##

    def __init__(self, flow_settings = None, flow_opts = None, disc = None, disc_opts = {}, sol_opts = None):

        # mesh, geometry, physical parameters
        if flow_settings is not None:
            self.settings = flow_settings
        elif flow_opts is not None:
            self.settings = FlowOptions(**flow_opts)
        else:
            raise Exception("need either flow_settings or flow_opts!")

        # spaces, forms
        avail_discs = { "mcs" : lambda settings, kwa : MCS(settings=settings, **kwa),
                        "hdg" : lambda settings, kwa : HDG(settings=settings, **kwa) }
        if disc is not None:
            self.disc = disc
        else:
            if "disc_type" in disc_opts:
                self.disc_type = disc_opts["disc_type"]
                del disc_opts["disc_type"]
            else:
                self.disc_type = "mcs" # default
            if self.disc_type in avail_discs:
                self.disc = avail_discs[self.disc_type](self.settings, disc_opts)
            else:
                raise Exception("Discretization \"{}\" not available!!".format(self.disc_type))

        # linalg, preconditioner
        self.InitLinAlg(sol_opts)

        self.velocity = self.la.velocity
        self.velhat = self.la.velhat
        self.sigma = self.la.sigma
        self.eta = self.la.eta
        self.pressure = self.la.pressure

    def InitLinAlg(self, sol_opts = None):
        if sol_opts is None:
            sol_opts = dict()
        self.la = self.LinAlg(self, **sol_opts)

    def AssembleLinAlg(self):
        self.la.Assemble()

    def Solve(self, tol = 1e-8, ms = 1000, rel_err = True, solver = "minres", presteps = 0, use_sz = False, printrates = None,
              restart = 10000):

        pr = ngs.mpi_world.rank==0 if printrates is None else printrates

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

        if hasattr(self.la, "rblf_x"): # apply pc hack workaround
            RMAT = ngs.BlockMatrix([[self.la.rblf_x.mat.local_mat.CreateDiagonal(), None],
                                    [None, self.la.rblf_q.mat.local_mat.CreateDiagonal()]])
            RMX = self.la.rblf_x.mat.local_mat.CreateDiagonal()
            RMQ = self.la.rblf_q.mat.local_mat.CreateDiagonal()
            if ngs.mpi_world.size > 1:
                RMX = ngs.ParallelMatrix(RMX, self.la.rblf_x.mat.row_pardofs, self.la.rblf_x.mat.row_pardofs, ngs.ParallelMatrix.D2D)
                RMQ = ngs.ParallelMatrix(RMQ, self.la.rblf_q.mat.row_pardofs, self.la.rblf_q.mat.row_pardofs, ngs.ParallelMatrix.D2D)
            RMAT = ngs.BlockMatrix([[RMX, None], [None, RMQ]])

            # RMAT = ngs.BlockMatrix([[self.la.rblf_x.mat, None], [None, self.la.rblf_q.mat]])

            tv = RMAT.CreateColVector()
            def nip(x, y):
                tv.data = RMAT * x
                return ngs.InnerProduct(tv, y)
        else:
            def nip(x, y):
                return ngs.InnerProduct(x, y)

        print("solve prep done")
        sys.stdout.flush()

        if solver == "bp":
            if not self.la.block_la:
                raise Exception("For BPCG, use block-PC!")
            bp_cg = BPCGSolver(M = self.la.M, Mhat = self.la.Mpre, maxsteps=ms, tol=tol,
                               printrates = pr, rel_err = rel_err)
            # if self.la.need_bp_scale:
            bp_cg.ScaleAhat(tol=1e-10)#, scal = 1.0/1.35)
            sol_vec.data = bp_cg * rhs_vec
            nits = bp_cg.iterations
            self.solver = bp_cg
        elif solver == "gmres":
            # ngs.solvers.GMRes(A = self.la.M, b = rhs_vec, x = sol_vec, pre = self.la.Mpre,
            #                   tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps=ms )
            # note: to compare, use rel_err=False
            # A = ngs.BlockMatrix([ [ self.la.Apre, self.la.Apre @ self.la.BT @ self.la.Spre ],
            #                       [ None, - self.la.Spre ] ])
            # B = ngs.BlockMatrix([ [ ngs.IdentityMatrix(self.la.A.height), None ],
            #                       [ -self.la.B @ self.la.Apre, ngs.IdentityMatrix(self.la.Spre.height) ] ])
            # szpre = A @ B

            def myCB (it, err):
                sys.stdout.flush()

            if use_sz:
                pc = SZPC(M = self.la.M, Ahat = self.la.Apre, Shat = self.la.Spre)
            else:
                pc = self.la.Mpre
            gmres = GMResSolver(M = self.la.M, Mhat = pc, maxsteps=ms, tol=tol,
                                printrates = pr, rel_err = rel_err, restart=restart, innerproduct = nip,
                                callback=myCB)
            # opts = {"ksp_type":"cg", "ksp_atol":1e-30, "ksp_rtol":1e-8,
            #         #"pc_view" : "",
            #         #"ksp_view" : "",
            #         "ksp_monitor" : "",
            #         "pc_type" : "gamg"}
            # pcM = petsc.FlatPETScMatrix(self.la.M)
            # pcpc = petsc.NGs2PETScPrecond(mat=self.la.M, pc=pc)
            # gmres = petsc.ksp(pcM, finalize=False, name="mygmres",
            #                   petsc_options = { "ksp_monitor" : "", "ksp_type" : "gmres", "ksp_rtol" : tol })
            # gmres.SetPC(pcpc)
            # gmres.Finalize()
            # sol_vec.data = gmres * rhs_vec
            # nits = gmres.iterations
            if presteps > 0:
                gmres.maxsteps = presteps
                gmres.tol = 0
                sol_vec[:] = 0
                gmres.Solve(sol = sol_vec, rhs = rhs_vec, initialize = False)
                nits = gmres.iterations
                gmres.maxsteps = ms
                gmres.tol = tol
                gmres.Solve(sol = sol_vec, rhs = rhs_vec, initialize = False)
                nits = nits + gmres.iterations
            else:
                print("GO INTO GMR")
                sys.stdout.flush()
                sol_vec.data = gmres * rhs_vec
                nits = gmres.iterations
            self.solver = gmres
            # print("used restarts = ", gmres.restarts)
        elif solver == "apply_pc":
            if use_sz:
                pc = SZPC(M = self.la.M, Ahat = self.la.Apre, Shat = self.la.Spre)
            else:
                pc = self.la.Mpre
            sol_vec.data = pc * rhs_vec
            nits = 1
        elif solver == "minres":
            # note: to compare, use rel_err=False
            # ngs.solvers.MinRes(mat = self.la.M, rhs = rhs_vec, sol = sol_vec, pre = self.la.Mpre,
                               # tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps=ms)
            # nits = -1
            minres = MinResSolver(M = self.la.M, Mhat = self.la.Mpre, maxsteps=ms, tol=tol,
                                  printrates = pr, rel_err = rel_err)#, innerproduct = nip)
            # pre-iterate
            if presteps > 0:
                minres.maxsteps = presteps
                minres.tol = 0
                sol_vec[:] = 0
                minres.Solve(sol = sol_vec, rhs = rhs_vec, initialize = False)
                nits = minres.iterations
                minres.maxsteps = ms
                minres.tol = tol
                minres.Solve(sol = sol_vec, rhs = rhs_vec, initialize = False)
                nits = nits + minres.iterations
            else:
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
