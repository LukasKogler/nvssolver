import ngsolve as ngs
import netgen as ng
import ngs_amg

### Misc Utilities ###

## H1 -> MCS embedding ##

def Lo2Ho ( X_ho, indsho, X_lo, indslo ):
    dim = X_ho.mesh.dim

    Vn_ho = X_ho.components[indsho[0]]
    Vt_ho = X_ho.components[indsho[1]]
    Vn_lo = X_lo.components[indslo[0]]
    Vt_lo = X_lo.components[indslo[1]]

    offn_ho = X_ho.Range(indsho[0]).start
    offt_ho = X_ho.Range(indsho[1]).start
    offn_lo = X_lo.Range(indslo[0]).start
    offt_lo = X_lo.Range(indslo[1]).start

    ind = X_lo.ndof * [0]

    hdiv_os = (Vn_ho.globalorder + 1) * (Vn_ho.globalorder) // 2
    tf_os = Vt_ho.globalorder

    useless_dnum = -1
    
    for f in mesh.facets:
        dn_ho = Vn_ho.GetDofNrs(f)
        dt_ho = Vt_ho.GetDofNrs(f)
        dn_lo = Vn_lo.GetDofNrs(f)
        dt_lo = Vt_lo.GetDofNrs(f)
        if len(dn_ho) > 0:
            ct = Vn_ho.GetDofCouplingType(dn_ho[0])
            if ct != ngs.COUPLING_TYPE.UNUSED_DOF:
                if dim == 2:
                    for dlo, dho in zip(dn_lo, dn_ho):
                        ind[offn_lo + dlo] = offn_ho + dho
                    for dlo, dho in zip(dt_lo, dt_ho):
                        ind[offt_lo + dlo] = offt_ho + dho
                else:
                    # HDiv DOFs
                    ind[offn_lo + dn_lo[0]] = offn_ho + dn_ho[0]
                    ind[offn_lo + dn_lo[1]] = offn_ho + dn_ho[1]
                    ind[offn_lo + dn_lo[2]] = offn_ho + dn_ho[1 + hdiv_offset]
                    # TangentialFacet DOFs
                    for dlo, dho in zip(dt_lo[:4], dt_ho[:4]):
                        ind[offt_lo + dlo] = offt_ho + dho
                    ind[offt_lo + dt_lo[4]] = offt_ho + dt_ho[2 * tf_os]
                    ind[offt_lo + dt_lo[5]] = offt_ho + dt_ho[2 * tf_os + 1]
            else: # hopefully no harm done here ...
                if useless_dnum == -1:
                    useless_dnum = dn_ho[0]
                for dof in dn_lo:
                    ind[offn_lo + dof] = useless_dnum
                for dof in dt_lo:
                    ind[offt_lo + dof] = useless_dnum
                
    lo_ho = ngs.PermutationMatrix(X.ndof, ind)
    return lo_ho
                
                
def H1Embedding (H1, X, inds = [0, 1]):
    mesh = X.mesh

    indv, indh = inds
    
    V_ho = X.components[indv]
    Vh_ho = X.components[indvh]

    V_lo = ngs.HDiv(mesh, order=1)
    Vh_lo = ngs.TangentialFacetFESpace(mesh, order=1)
    X_lo = ngs.FESpace([V, Vhat])
    
    ## VectorH1 -> [HDiv_lo, TangFacet_lo] ##
    u,v = H1vec.TnT()
    (un, ut), (vn, vt) = Xlo.TnT()     # normal
    vnd = vn.Operator("dual")
    
    mmix = BilinearForm(trialspace = Xlo, testspace = H1vec)
    mmix += u * vnd * ngs.dx(element_vb=BND)
    mmix += u * ( vt - (vt*n)*n) * ngs.dx(element_vb=BND)
    mmix.Assemble()

    m = BilinearForm(Xlo)
    m += un * vnd * ngs.dx(element_vb=BND)
    m += ut * ( vt - (vt*n)*n) * ngs.dx(element_vb=BND)
    m.Assemble()

    minv = ngs_amg.BDI(m.mat.local_mat, face_blocks)
    vec_Xlo = (minv @ m.mat.local_mat)

    Xlo_X = Lo2Ho(X_ho = X, indsho = [indv, indvh], X_lo = Xlo, indslo = [0,1])
    
    vec_X = vec_Xlo @ Xlo_X
    
    return vec_X

    
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
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       dirichlet = settings.inlet + "|" + settings.wall_noslip + "|" + settings.outlet)
            else: # works with facet-aux
                self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                       dirichlet = settings.inlet + "|" + settings.wall_noslip)
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True, ordertrace = self.order-1)

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
                                "auxfacet" : lambda astokes, opts : self.SetUpAFacet(astokes, **opts) }

            
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
                self.massp +=  1/stokes.settings.nu * p * q * ngs.dx
                # self.Spre = ngs.Preconditioner(self.massp, "direct")
                self.Spre = ngs.Preconditioner(self.massp, "local")

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
                
                
        def SetUpAAux(self, stokes, amg_opts = dict(), mpi_thrad = False, mpi_overlap = mpi_overlap, shm = None,
                      bsblocks = None, **kwargs):
            V = ngs.VectorH1(stokes.settings.mesh, order = 1, dirichlet = stokes.settings.wall_slip + stokes.settings.inflow)
            u,v = V.TnT()
            a_aux = ngs.BilinearForm(V)
            if stokes.settings.sym:
                eps = lambda U : 0.5 * (grad(U) + grad(U).trans)
                a_aux += stokes.settings.nu * eps(u) * eps(v) * ngs.dx
                if stokes.settings.mesh.dim == 2:
                    aux_pre = ngs_amg.elast_2d(a_aux, **amg_opts)
                else:
                    aux_pre = ngs_amg.elast_3d(a_aux, **amg_opts)
            else:
                a_aux += stokes.settings.nu * ngs.Grad(u) * ngs.Grad(v) * ngs.dx
                if stokes.settings.mesh.dim == 2:
                    aux_pre = ngs_amg.h1_2d(a_aux, **amg_opts)
                else:
                    aux_pre = ngs_amg.h1_3d(a_aux, **amg_opts)
            a_aux.Assemble()
            emb = H1Embedding(H1 = V, X = stokes.disc.X, inds = [0,1])
            if mpi_world.size > 1:
                emb = ngs.ParallelMatrix(emb, row_pardofs = V.ParalellDofs(),
                                         col_pardofs = X.ParalellDofs(), ParallelMatrix.C2C)

            bsblocks = ["CF"]       # cell + face
            bsblocks = ["FE"]       # face + edge
            bsblocks = ["F", "C"]   # face, cell
            bsblocks = ["E", "F"]   # edge, face
                
            if bsblocks is None:
                if self.elint or stokes.disc.V.globalorder < 3:
                    bsblocks = [ ngs.NT_FACET ]
            
            bsmoother = ngs_amg.CreateBlockSmoother(self.a.mat, NT_FACET, shm = shm, mpi_thread = mpi_thread,
                                                    mpi_overlap = mpi_overlap)
            self.Apre = ngs_amg.EmbeddedAMGMatrix(amg_mat = aux_pre.amg_mat, smoother = bsmoother, embedding = emb)
            
            
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
                
                
        def SetUpDummy(self, stokes, **kwargs):
            self.Mpre = ngs.Projector(stokes.disc.Xext.FreeDofs(self.elint), True)
                
                
        def TestBlock(self):
            o_ms_l = ngs.ngsglobals.msg_level
            ngs.ngsglobals.msg_level = 0

            evs_A = list(ngs.la.EigenValues_Preconditioner(mat=self.a.mat, pre=self.Apre, tol=1e-10))
            print("\n----")
            print("min ev. preA\A:", evs_A[0])
            print("max ev. preA\A:", evs_A[-1])
            print("cond-nr preA\A:", evs_A[-1]/evs_A[0])
            print("----")
            
            S = self.B @ self.Apre @ self.B.T
            evs_S = list(ngs.la.EigenValues_Preconditioner(mat=S, pre=self.Spre, tol=1e-10))
            print("min ev. preS\S:", evs_S[0])
            print("max ev. preS\S:", evs_S[-1])
            print("cond-nr preS\S:", evs_S[-1]/evs_S[0])
            print("----\n")

            ngs.ngsglobals.msg_level = o_ms_l


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
        
        ngs.solvers.GMRes(A = self.la.M, b = self.la.rhs, x = sv2, pre = self.la.Mpre,
                          tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps=ms ) 

        #ngs.solvers.GMRes(A = self.la.M, b = self.la.rhs, x = sv2, freedofs = self.disc.Xext.FreeDofs(),
        #                  tol = tol, printrates = ngs.mpi_world.rank == 0, maxsteps = ms)

        # sv2.data = self.la.Mpre * self.la.rhs
        
        self.la.sol_vec.data += sv2

        # for k, comp in enumerate(self.la.gfu.components):
        #     print("SOL comp", k)
        #     print(self.la.gfu.components[k].vec)


### END Stokes Template ###
