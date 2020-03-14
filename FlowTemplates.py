import ngsolve as ngs
import netgen as ng
import ngs_amg

### Misc Utilities ###

class SPCST (ngs.BaseMatrix):
    def __init__(self, smoother, pc, emb):
        super(SPCST, self).__init__()
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
        self.res.data = b
        # Forward smoothing - update residual
        self.S.Smooth(x, b, self.res, x_zero = True, res_updated = True, update_res = True)
        # x.data += self.emb_pc * self.res
        self.emb_pc.MultAdd(1.0, self.res, x)
        # Backward smoothing - no need to update residual
        self.S.SmoothBack(x, b, self.res, False, False, False)
        

### END Misc Utilities ###


### FlowOptions ###

class FlowOptions:
    """
    A collection of parameters for Stokes/Navier-Stokes computations. Collects Boundary-conditions
    """
    def __init__(self, mesh, geom = None, nu = 1, inlet = "", outlet = "", wall_slip = "", wall_noslip = "",
                 uin = None, symmetric = True, vol_force = None):
        # geom/mesh
        self.geom = geom
        self.mesh = mesh

        # physical parameters
        self.nu = nu
        self.vol_force = vol_force

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
    def __init__(self, settings = None, order = 2, RT = False, hodivfree = False, compress = True, truecompile = False, pq_reg = 0):

        self.settings = settings
        self.order = order
        self.RT = RT
        self.hodivfree = hodivfree
        self.compress = compress
        self.sym = settings.sym
        self.truecompile = truecompile
        self.pq_reg = pq_reg

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
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = order, GGBubbles = True, discontinuous = True, ordertrace = self.order)
            # self.Sigma = ngs.HCurlDiv(mesh, order=order + 1, discontinuous=True, ordertrace=order) # slower I think
        else:
            if False: # "correct" version
                # self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       # dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.outlet)
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.outlet)
            else: # works with facet-aux
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       dirichlet = settings.inlet + "|" + settings.wall_noslip)
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True, ordertrace = self.order-1)
            # self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True)

        if self.sym:
            if settings.mesh.dim == 2:
                self.S = ngs.L2(settings.mesh, order=self.order if self.RT else order-1)
            else:
                self.S = ngs.VectorL2(settings.mesh, order=self.order if self.RT else order-1)
        else:
            self.S = None

        self.Q = ngs.L2(settings.mesh, order = 0 if self.hodivfree else order-1)

        if self.compress:
            self.Sigma.SetCouplingType(ngs.IntRange(0, self.Sigma.ndof), ngs.COUPLING_TYPE.HIDDEN_DOF)
            self.Sigma = ngs.Compress(self.Sigma)
            if self.sym:
                self.S.SetCouplingType(ngs.IntRange(0, self.S.ndof), ngs.COUPLING_TYPE.HIDDEN_DOF)
                self.S = ngs.Compress(self.S)
            
        if self.sym:
            self.X = ngs.FESpace([self.V, self.Vhat, self.Sigma, self.S])
        else:
            self.X = ngs.FESpace([self.V, self.Vhat, self.Sigma])

    def Compile (self, term):
        return term.Compile(realcompile = self.truecompile, wait = True)
    
    def SetUpForms(self, compound = False):

        self.dS = ngs.dx(element_vb = ngs.BND)

        n = ngs.specialcf.normal(self.settings.mesh.dim)
        def tang(u):
            return u - (u * n) * n
        if self.settings.mesh.dim == 2:
            def Skew2Vec(m):
                return m[1, 0] - m[0, 1]
        else:
            def Skew2Vec(m):
                return ngs.CoefficientFunction((m[0, 1] - m[1, 0], m[2, 0] - m[0, 2], m[1, 2] - m[2, 1]))

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

        self.a_vol = -1 / self.settings.nu * ngs.InnerProduct(sigma, tau) \
                     + ngs.div(sigma) * v \
                     + ngs.div(tau) * u
        if self.sym:
           self.a_vol += ngs.InnerProduct(W, Skew2Vec(tau)) \
                         + ngs.InnerProduct(R, Skew2Vec(sigma))
        self.a_bnd = - ((sigma * n) * n) * (v * n) \
                       - ((tau * n) * n) * (u * n) \
                       - (sigma * n) * tang(vhat) \
                       - (tau * n) * tang(uhat)
        self.b_vol = ngs.div(u) * q
        self.bt_vol = ngs.div(v) * p

        if self.pq_reg != 0:
            self.c_vol = -self.pq_reg * self.settings.nu * p*q
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

    def stokesM(self):
        if self._stokesM == None:
            self._stokesM = self.Compile(self.a_vol + self.b_vol + self.bt_vol) * ngs.dx + self.Compile(self.a_bnd) * self.dS
        return self._stokesM

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
        def __init__(self, stokes, pc_ver = "aux", block_la = None, pc_opts = dict(),
                     elint = False):

            if block_la is None:
                self.block_la = pc_ver != "direct"
            else:
                self.block_la = block_la

            self.elint = elint

            self.pc_avail = { "direct" : lambda astokes, opts : self.SetUpDirect(astokes, **opts),
                              "block" : lambda astokes, opts : self.SetUpBlock(astokes, **opts),
                              "none" : lambda astokes, opts : self.SetUpDummy(astokes) }

            self.pc_a_avail = { "direct" : lambda astokes, opts : self.SetUpADirect(astokes, **opts),
                                "auxh1" : lambda astokes, opts : self.SetUpAAux(astokes, **opts),
                                "auxfacet" : lambda astokes, opts : self.SetUpAFacet(astokes, **opts),
                                "stokesamg" : lambda astokes, opts : self.SetUpStokesAMG(astokes, **opts) }

            
            if self.block_la and pc_ver == "direct":
                raise "For direct solve, use block_la = False!"

            stokes.disc.SetUpForms(compound = not self.block_la)
            
            if self.block_la:
                self.gfu = ngs.GridFunction(stokes.disc.X)
                self.velocity = self.gfu.components[0]
                self.p = ngs.GridFunction(stokes.disc.Q)
                self.pressure = self.p
                self.sol_vec = ngs.BlockVector([self.gfu.vec,
                                                self.pressure.vec])
            else:
                self.gfu = ngs.GridFunction(stokes.disc.Xext)
                self.velocity = self.gfu.components[0].components[0]
                self.pressure = self.gfu.components[1]
                self.sol_vec = self.gfu.vec
            
            self._to_assemble = []

            self.SetUpFWOps(stokes)
            self.SetUpRHS(stokes)

            self.Assemble()

            if self.block_la:
                self.A = self.a.mat
                self.B = self.b.mat
                # self.BT = self.B.T
                self.BT = self.B.CreateTranspose()
                self.C = None if self.c is None else self.c.mat

                self.M = ngs.BlockMatrix( [ [self.A, self.BT],
                                            [self.B, self.C] ] )

                self.rhs = ngs.BlockVector ( [self.f.vec,
                                              self.g.vec] )
            else:
                self.M = self.m.mat
                self.rhs = self.f.vec
                                           
            if not pc_ver in self.pc_avail:
                raise "invalid PC version!"
            else:
                self.pc_avail[pc_ver](stokes, pc_opts)
                
                
        def SetUpFWOps(self, stokes):
            # Forward operators
            if self.block_la:
                self.a = ngs.BilinearForm(stokes.disc.X, condense = self.elint, eliminate_hidden = stokes.disc.compress)
                self.a += stokes.disc.stokesA()

                self.b = ngs.BilinearForm(trialspace = stokes.disc.X, testspace = stokes.disc.Q)
                self.b += stokes.disc.stokesB()

                if stokes.disc.stokesC() is not None:
                    self.c = ngs.BilinearForm(stokes.disc.Q)
                    self.c += stokes.disc.stokesC()
                else:
                    self.c = None

                self._to_assemble += [ self.a, self.b, self.c ]

            else:
                self.m = ngs.BilinearForm(stokes.disc.Xext, condense = self.elint, eliminate_hidden = stokes.disc.compress)
                self.m += stokes.disc.stokesM()

                self._to_assemble += [ self.m ]
                
                
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

        def SetUpDirect(self, stokes, inv_type = None, **kwargs):
            # Direct inverse
            if self.block_la:
                raise "Cannot invert block matrices!"
            else:
                itype = "umfpack" if inv_type is None else inv_type
                self.Mpre = self.M.Inverse(stokes.disc.Xext.FreeDofs(self.elint), inverse = itype)
            
        def SetUpBlock(self, stokes, a_opts = { "type" : "direct" } , **kwargs):
            if not self.block_la:
                raise "block-PC with big compond space todo"
            else:
                p,q = stokes.disc.Q.TnT()
                self.massp = ngs.BilinearForm(stokes.disc.Q)
                if stokes.disc.stokesC() is not None:
                    self.massp += -1 * stokes.disc.stokesC()
                self.massp +=  1/stokes.settings.nu * p * q * ngs.dx
                self.Spre = ngs.Preconditioner(self.massp, "direct")
                # self.Spre = ngs.Preconditioner(self.massp, "local")

                self.massp.Assemble()

                aver = a_opts["type"] if "type" in a_opts else "direct"
                if aver in self.pc_a_avail:
                    self.pc_a_avail[aver](stokes, a_opts)
                else:
                    raise "invalid pc type for A block!"

                self.Mpre = ngs.BlockMatrix( [ [self.Apre, None],
                                               [None, self.Spre.mat] ] )


        def SetUpADirect(self, stokes, inv_type = None, **kwargs):
            if inv_type is None:
                ainvt = "sparsecholesky" if stokes.disc.compress else "umfpack"
            else:
                ainvt = inv_type
            self.Apre = self.a.mat.Inverse(self.a.space.FreeDofs(self.elint), inverse = ainvt)
                
                
        def SetUpAAux(self, stokes, amg_opts = dict(), mpi_thrad = False, mpi_overlap = False, shm = None,
                      bsblocks = None, multiplicative = True, **kwargs):
            # Auxiliary space problem/Preconditioner
            # V = ngs.H1(stokes.settings.mesh, order = 1, dirichlet = stokes.settings.wall_noslip + stokes.settings.inlet, dim = stokes.settings.mesh.dim)
            #V = ngs.VectorH1(stokes.settings.mesh, order = 2, dirichlet = stokes.settings.wall_noslip + stokes.settings.inlet, nodalp2 = True)
            # V = ngs.VectorH1(stokes.settings.mesh, order = 1, dirichlet = stokes.settings.wall_noslip + stokes.settings.inlet)
            V = ngs.VectorH1(stokes.settings.mesh, order = 1, dirichlet = stokes.settings.wall_noslip + stokes.settings.inlet)
            u, v = V.TnT()
            a_aux = ngs.BilinearForm(V)
            if stokes.settings.sym:
                eps = lambda U : 0.5 * (ngs.grad(U) + ngs.grad(U).trans)
                a_aux += stokes.settings.nu * ngs.InnerProduct(eps(u), eps(v)) * ngs.dx
                amg_cl = ngs_amg.elast_2d if stokes.settings.mesh.dim == 2 else ngs_amg.elast_3d
            else:
                a_aux += stokes.settings.nu * ngs.InnerProduct(ngs.Grad(u), ngs.Grad(v)) * ngs.dx
                amg_cl = ngs_amg.h1_2d if stokes.settings.mesh.dim == 2 else ngs_amg.h1_3d

            # amg_opts["ngs_amg_crs_alg"] = "ecol"
            # amg_opts["ngs_amg_enable_multistep"] = True
            #amg_opts["ngs_amg_lo"] = False
            #amg_opts["ngs_amg_subset"] = "free"
            # aux_pre = amg_cl(a_aux, **amg_opts)
            aux_pre = ngs.Preconditioner(a_aux, "direct")

            # aux_pre = ngs.Preconditioner(a_aux, "bddc", coarsetype = "ngs_amg.h1_2d")
            # aux_pre = ngs.Preconditioner(a_aux, "bddc")

            a_aux.Assemble()

            # aux_pre.Test()
            # quit()

            # Embeddig Auxiliary space -> MCS space
            emb1 = ngs.comp.ConvertOperator(spacea = V, spaceb = stokes.disc.V, localop = False, parmat = False)
            tc1 = ngs.Embedding(stokes.disc.X.ndof, stokes.disc.X.Range(0)) # to-compound
            emb2 = ngs.comp.ConvertOperator(spacea = V, spaceb = stokes.disc.Vhat, localop = False, parmat = False)
            tc2 = ngs.Embedding(stokes.disc.X.ndof, stokes.disc.X.Range(1)) # to-compound
            embA = (tc1 @ emb1) + (tc2 @ emb2)
            self.embA = embA
            if ngs.mpi_world.size > 1:
                embA = ngs.ParallelMatrix(embA, row_pardofs = V.ParallelDofs(), col_pardofs = stokes.disc.X.ParallelDofs(),
                                          op = ParallelMatrix.C2C)

            # Block-Smoother to combine with auxiliary PC    
            X = stokes.disc.X
            x_free = X.FreeDofs(self.elint)
            facet_blocks = list()
            if True:
                for facet in stokes.settings.mesh.facets:
                    block = list( dof for dof in X.GetDofNrs(facet) if dof >=0 and x_free[dof])
                    if len(block):
                        facet_blocks.append(block)
                for elnr in range(stokes.settings.mesh.ne):
                    block = list( dof for dof in X.GetDofNrs(ngs.NodeId(ngs.ELEMENT, elnr)) if dof >=0 and x_free[dof])
                    # print("block len", len(block))
                    if len(block):
                        facet_blocks.append(block)
            else:
                for elem in stokes.settings.mesh.Elements():
                    block = list( dof for dof in X.GetDofNrs(elem) if dof >=0 and x_free[dof])
                    if len(block):
                        facet_blocks.append(block)
                    
                
            # The entire PC
            if True:
                # input("AAA")
                bsmoother = ngs_amg.CreateHybridBlockGSS(mat = self.a.mat, blocks = facet_blocks, shm = ngs.mpi_world.size == 1)
                # input("AAA")
                # self.Apre = ngs_amg.S_PC_S(smoother = bsmoother, pre = aux_pre, emb = embA)
                self.Apre = SPCST(smoother = bsmoother, pc = aux_pre, emb = embA)
                # input("AAA")
                # self.Apre = SPCST(smoother = bsmoother, pc = aux_pre, emb = embA)
            else:
                bsmoother = self.a.mat.local_mat.CreateBlockSmoother(facet_blocks)
                # bsmoother = self.a.mat.local_mat.CreateSmoother(stokes.disc.X.FreeDofs(True))
                # self.Apre = bsmoother + embA @ aux_pre @ embA.T
                # self.Apre = bsmoother + embA @ aux_pre @ embA.T
                self.Apre = bsmoother + embA @ aux_pre @ embA.T

            self.Astuff = (bsmoother, embA, aux_pre, V)

            # evs_A = list(ngs.la.EigenValues_Preconditioner(mat=self.a.mat, pre=self.Apre, tol=1e-10))
            # print("\n----")
            # print("min ev. preA\A:", evs_A[0])
            # print("max ev. preA\A:", evs_A[-1])
            # print("cond-nr preA\A:", evs_A[-1]/evs_A[0])
            # print("----")

            
        def SetUpAFacet(self, stokes, amg_opts = dict(), **kwargs):
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
            
        def SetUpStokesAMG(self, stokes, amg_opts = dict(), **kwargs):
            # self.Apre = ngs_amg.stokes_gg_2d(self.a, **amg_opts)
            # self.Apre = ngs.Preconditioner(self.a, "ngs_amg.stokes_gg_2d", **amg_opts)
            self.Apre = ngs_amg.stokes_gg_2d(self.a, **amg_opts)
            self.a.Assemble()
            
        def SetUpDummy(self, stokes, **kwargs):
            self.Mpre = ngs.Projector(stokes.disc.Xext.FreeDofs(self.elint), True)
                
                
        def TestBlock(self):
            o_ms_l = ngs.ngsglobals.msg_level
            ngs.ngsglobals.msg_level = 0

            evs_A = list(ngs.la.EigenValues_Preconditioner(mat=self.a.mat, pre=self.Apre, tol=1e-10))
            print("\n----")
            print("min ev. preA\A:", evs_A[:5])
            print("max ev. preA\A:", evs_A[-5:])
            print("cond-nr preA\A:", evs_A[-1]/evs_A[0])
            print("----")
            
            S = self.B @ self.Apre @ self.B.T
            # evs_S = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=ngs.IdentityMatrix(S.height), tol=1e-17))
            evs_S = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=self.Spre, tol=1e-10))
            print("min ev. preS\S:", evs_S[0:5])
            print("max ev. preS\S:", evs_S[-5:])
            evs0 = evs_S[0] if evs_S[0] > 1e-4 else evs_S[1]
            print("cond-nr preS\S:", evs_S[-1]/(evs0))
            print("----\n")

            ngs.ngsglobals.msg_level = o_ms_l

            return evs_S[-1]/evs0, evs_A[-1]/evs_A[0]

        def Assemble(self):
            for x in self._to_assemble:
                if x is not None:
                    x.Assemble()

    def __init__(self, flow_settings = None, flow_opts = None, disc = None, disc_opts = None, sol_opts = None):

        # mesh, geometry, physical parameters 
        if flow_settings is not None:
            self.settings = flow_settings
        elif flow_opts is not None:
            self.settings = FlowOptions(**flow_opts)
        else:
            raise "need either flow_settings or flow_opts!"

        # spaces, forms
        if disc is not None:
            self.disc = disc
        elif disc_opts is not None:
            self.disc = MCS(settings = self.settings, **disc_opts)
        else:
            raise "need either disc or disc_opts!"

        # linalg, preconditioner
        self.InitLinAlg(sol_opts)

        self.velocity = self.la.velocity
        self.pressure = self.la.pressure
            
    def InitLinAlg(self, sol_opts = None):
        if sol_opts is None:
            sol_opts = dict()
        self.la = self.LinAlg(self, **sol_opts)
        
    def AssembleLinAlg(self):
        self.la.Assemble()

    def Solve(self, tol = 1e-8, ms = 1000):

        if self.settings.uin is not None and len(self.settings.inlet)>0:
            self.velocity.Set(self.settings.uin, definedon=self.settings.mesh.Boundaries(self.settings.inlet))
            self.la.rhs.data -= self.la.M * self.la.sol_vec
        
        sv2 = self.la.sol_vec.CreateVector()
        sv2[:] = 0
        
        #ngs.solvers.GMRes(A = self.la.M, b = self.la.rhs, x = sv2, pre = self.la.Mpre,
        #                  tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps=ms ) 

        ngs.solvers.MinRes(mat = self.la.M, rhs = self.la.rhs, sol = sv2, pre = self.la.Mpre,
                           tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps=ms ) 

        #ngs.solvers.GMRes(A = self.la.M, b = self.la.rhs, x = sv2, freedofs = self.disc.Xext.FreeDofs(),
        #                  tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps = ms)

        # sv2.data = self.la.Mpre * self.la.rhs
        
        self.la.sol_vec.data += sv2

        # for k, comp in enumerate(self.la.gfu.components):
        #     print("SOL comp", k)
        #     print(self.la.gfu.components[k].vec)


### END Stokes Template ###
