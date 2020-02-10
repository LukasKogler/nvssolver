import ngsolve as ngs
import netgen as ng
import ngs_amg

### FlowOptions ###

class FlowOptions:
    """
    A collection of parameters for Stokes/Navier-Stokes computations. Collects Boundary-conditions
    """
    def __init__(self, mesh, geom = None, nu = 1, inflow = "", outflow = "", wall_slip = "", wall_noslip = "",
                 uin = None, symmetric = True, vol_force = None):
        # geom/mesh
        self.geom = geom
        self.mesh = mesh

        # physical parameters
        self.nu = nu
        self.vol_force = vol_force

        # BCs
        self.inflow = inflow
        self.uin = 0 if uin is None else uin
        self.outflow = outflow
        self.wall_slip = wall_slip
        self.wall_noslip = wall_noslip

        # grad:grad or eps:eps?
        self.sym = symmetric

### END FlowOptions ###


### MCS Discretization ###

class MCS:
    def __init__(self, settings = None, order = 2, RT = False, hodivfree = False, compress = True, truecompile = False):

        self.order = order
        self.RT = RT
        self.hodivfree = hodivfree
        self.compress = compress
        self.sym = settings.sym
        self.truecompile = truecompile

        # Spaces
        self.V = ngs.HDiv(settings.mesh, order = self.order, RT = self.RT, hodivfree = self.hodivfree, \
                          dirichlet = settings.inflow + "|" + settings.wall_noslip + "|" + settings.wall_slip)
        if self.RT:
            self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = order, \
                                                   dirichlet = settings.inflow + "|" + settings.wall_noslip + "|" + settings.outflow)
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = order, GGBubbles = True, discontinuous = True, ordertrace = self.order)
            # self.Sigma = ngs.HCurlDiv(mesh, order=order + 1, discontinuous=True, ordertrace=order) # slower I think
        else:
            self.Vhat = ngs.TangentialFacetFESpace(settings.mesh, order = self.order-1, \
                                                   dirichlet = settings.inflow + "|" + settings.wall_noslip + "|" + settings.outflow)
            self.Sigma = ngs.HCurlDiv(settings.mesh, order = self.order-1, orderinner = self.order, discontinuous = True, ordertrace = self.order-1)

        if self.sym:
            if settings.mesh.dim == 2:
                self.S = ngs.L2(settings.mesh, order=self.order if self.RT else order - 1)
            else:
                self.S = ngs.VectorL2(settings.mesh, order=self.order if self.RT else order - 1)
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

        # Forms
        dS = ngs.dx(element_boundary=True)
        n = ngs.specialcf.normal(settings.mesh.dim)
        def tang(u):
            return u - (u * n) * n
        if settings.mesh.dim == 2:
            def Skew2Vec(m):
                return m[1, 0] - m[0, 1]
        else:
            def Skew2Vec(m):
                return ngs.CoefficientFunction((m[0, 1] - m[1, 0], m[2, 0] - m[0, 2], m[1, 2] - m[2, 1]))

        if self.sym:
            u, uhat, sigma, W = self.X.TrialFunction()
            v, vhat, tau, R = self.X.TestFunction()
        else:
            u, uhat, sigma = self.X.TrialFunction()
            v, vhat, tau = self.X.TestFunction()
        p, q = self.Q.TnT()

        def Compile (term):
            return term.Compile(realcompile = self.truecompile, wait = True)

        a_vol_cf = -1 / settings.nu * ngs.InnerProduct(sigma, tau) + ngs.div(sigma) * v + ngs.div(tau) * u
        if self.sym:
            a_vol_cf += (ngs.InnerProduct(W, Skew2Vec(tau)) + ngs.InnerProduct(R, Skew2Vec(sigma)))
        a_bnd_cf = (-((sigma * n) * n) * (v * n) - ((tau * n) * n) * (u * n)) + \
                   (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat))
        b_vol_cf = ngs.div(u)*q
        bt_vol_cf = ngs.div(v)*p

        self.stokesA = Compile(a_vol_cf) * ngs.dx + Compile(a_bnd_cf) * dS

        self.stokesB = Compile(b_vol_cf) * ngs.dx
        self.stokesBT = Compile(bt_vol_cf) * ngs.dx

        # self.stokesA = -1 / settings.nu * ngs.InnerProduct(sigma, tau) * ngs.dx + \
        #                (ngs.div(sigma) * v + ngs.div(tau) * u) * ngs.dx + \
        #                (-((sigma * n) * n) * (v * n) - ((tau * n) * n) * (u * n)) * dS + \
        #                (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS
        # if self.sym:
        #     self.stokesA += (ngs.InnerProduct(W, Skew2Vec(tau)) + ngs.InnerProduct(R, Skew2Vec(sigma))) * ngs.dx
        # self.stokesB = ngs.div(u) * q * ngs.dx
        # self.stokesBT = ngs.div(v) * p * ngs.dx

        if settings.vol_force is not None:
            self.stokesf = settings.vol_force * v * ngs.dx
        else:
            self.stokesf = None

        # for non-block matrices
        self.Xext = ngs.FESpace( [self.X, self.Q] )

        if self.sym:
            (u, uhat, sigma, W), p = self.Xext.TrialFunction()
            (v, vhat, tau, R), q = self.Xext.TestFunction()
        else:
            (u, uhat, sigma), p = self.Xext.TrialFunction()
            (v, vhat, tau), q = self.Xext.TestFunction()
            
        aex_vol_cf = -1 / settings.nu * ngs.InnerProduct(sigma, tau) + ngs.div(sigma) * v + ngs.div(tau) * u
        if self.sym:
            aex_vol_cf += (ngs.InnerProduct(W, Skew2Vec(tau)) + ngs.InnerProduct(R, Skew2Vec(sigma)))
        aex_bnd_cf = (-((sigma * n) * n) * (v * n) - ((tau * n) * n) * (u * n)) + \
                   (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat))
        bex_vol_cf = ngs.div(u)*q
        btex_vol_cf = ngs.div(v)*p

        self.stokesM = Compile(aex_vol_cf + bex_vol_cf + btex_vol_cf) * ngs.dx + Compile(aex_bnd_cf) * dS
            
### END MCS Discretization ###


### Stokes Template ###

class StokesTemplate():
            
    class LinAlg():
        def __init__(self, stokes, pc_ver = "aux", block_la = False, pc_opts = dict(),
                     elint = False):

            self.block_la = block_la
            self.elint = elint

            print("block_la ", self.block_la)
            
            if (self.block_la == True) and (pc_ver == "direct"):
                raise "For direct solve, use block_la = False!"

            self._to_assemble = []

            self.SetUpFWOps(stokes)
            self.SetUpRHS(stokes)

            self.Assemble()

            if self.block_la:
                self.A = self.a.mat
                self.B = self.b.mat
                # self.BT = self.B.T
                self.BT = self.B.CreateTranspose()

                self.M = ngs.BlockMatrix( [ [self.A, self.BT],
                                            [self.B, None] ] )

                self.rhs = ngs.BlockVector ( [self.f.vec,
                                              self.g.vec] )
            else:
                self.M = self.m.mat
                self.rhs = self.f.vec
                                           
            
            self.pc_avail = {"direct" : lambda astokes, opts : self.SetUpDirect(astokes, **opts),
                             "aux" : lambda astokes, opts: self.SetUpAux(astokes, **opts),
                             "facet_aux" : lambda astokes, opts: self.SetUpFacetAux(astokes, **opts) }
            if not pc_ver in self.pc_avail:
                raise "invalid PC version!"
            else:
                self.pc_avail[pc_ver](stokes, pc_opts)
            
        def SetUpFWOps(self, stokes):
            # Forward operators
            if self.block_la:
                self.a = ngs.BilinearForm(stokes.disc.X, eliminate_internal = self.elint)
                self.a += stokes.disc.stokesA

                self.b = ngs.BilinearForm(trialspace = stokes.disc.X, testspace = stokes.disc.Q,
                                          eliminate_internal = self.elint)
                self.b += stokes.disc.stokesB

                self._to_assemble += [self.a, self.b]

            else:
                self.m = ngs.BilinearForm(stokes.disc.Xext, eliminate_internal = self.elint)
                self.m += stokes.disc.stokesM

                self._to_assemble += [ self.m ]
                
        def SetUpRHS(self, stokes):
            # Right hand side
            if self.block_la:
                self.f = ngs.LinearForm(stokes.disc.X)
                if stokes.disc.stokesf is not None:
                    self.f += stokes.disc.stokesf

                self.g = ngs.LinearForm(stokes.disc.Q)

                self._to_assemble += [ self.f, self.g ]

            else:
                self.f = ngs.LinearForm(stokes.disc.Xext)
                if stokes.disc.stokesf is not None:
                    self.f += stokes.disc.stokesf

                self._to_assemble += [ self.f ]
                
        def SetUpDirect(self, stokes, inv_type = None):
            # Direct inverse
            if self.block_la:
                raise "Cannot invert block matrices!"
            else:
                print("MAT, ", self.m.mat)
                itype = "umfpack" if inv_type is None else inv_type
                self.precond = self.m.mat.Inverse(stokes.disc.Xext.FreeDofs(self.elint), inverse = itype)
            
        def SetUpAux(self, stokes):
            raise "TODO!"
            # Embedded H1 auxiliary space + Block-GS


            # Inverse mass matrix for Schur complement
            p,q = stokes.settings.Q.TnT()
            self.massp = ngs.BilinearForm(stokes.settings.Q)
            self.massp +=  1/(stokes.settins.nu) * p * q * ngs.dx
            self.pc_S = ngs.Preconditioner(self.massp, "local")
            self.Spre.mat = self.pc_S.mat

            self.Mpre = ngs.BlockMatrix( [ [self.Apre, None,
                                            None, self.Spre] ] )


        

        def SetUpFacetAux(self, stokes):
            raise "TODO!"
            # Facet wise auxiliary space AMG + Block-GS
            amg_opts = { "ngs_amg_max_coarse_size" : 50,
                         "ngs_amg_max_levels" : 10 }
            self.pc_A = ngs_amg.mcs3d(self.A, **amg_opts)

            # Inverse mass matrix for Schur complement
            p,q = stokes.settings.Q.TnT()
            self.massp = ngs.BilinearForm(stokes.settings.Q)
            self.massp +=  1/(stokes.settins.nu) * p * q * ngs.dx
            self.pc_S = ngs.Preconditioner(self.massp, "local")
            self.Spre.mat = self.pc_S.mat

            self.Mpre = ngs.BlockMatrix( [ [self.Apre, None,
                                            None, self.Spre] ] )
                

        def Assemble(self):
            for x in self._to_assemble:
                print("ass ", x)
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

            
    def InitLinAlg(self, sol_opts = None):
        if sol_opts is None:
            sol_opts = dict()
        self.linalg_stuff = self.LinAlg(self, **sol_opts)
        
    def AssembleLinAlg(self):
        self.linalg_stuff.Assemble()

### END Stokes Template ###
