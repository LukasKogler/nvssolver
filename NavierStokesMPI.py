import ngsolve as ngs
import netgen as ng
import ngs_amg

class FlowOptions:
    """
    A collection of parameters for Stokes/Navier-Stokes computations. Collects Boundary-conditions
    """
    def __init__(self, mesh, geom = None, nu = 1, inflow = "", outflow = "", wall_slip = "", wall_noslip = "",
                 uin = None, symmetric = True):
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


class MCS:
    def __init__(self, settings = None, order = 2, RT = False, hodivfree = False, compress = True):

        self.order = order
        self.RT = RT
        self.hodivfree = hodivfree
        self.compress = compress
        self.sym = settings.sym

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
            if mesh.dim == 2:
                S = ngs.L2(settings.mesh, order=self.order if self.RT else order - 1)
            else:
                S = ngs.VectorL2(settings.mesh, order=self.order if self.RT else order - 1)
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
        n = ngs.specialcf.normal(mesh.dim)
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

        self.stokesA = -1 / self.nu * ngs.InnerProduct(sigma, tau) * ngs.dx + \
                       (ngs.div(sigma) * v + ngs.div(tau) * u) * ngs.dx + \
                       (-((sigma * n) * n) * (v * n) - ((tau * n) * n) * (u * n)) * dS + \
                       (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS
        if self.sym:
            self.stokesA += (ngs.InnerProduct(W, Skew2Vec(tau)) + ngs.InnerProduct(R, Skew2Vec(sigma))) * ngs.dx

        self.stokesB = div(u) * q * ngs.dx

        self.stokesf = settings.vol_force * v * ngs.dx


class Stokes:
            
    class LinAlg():
        def __init__(self, stokes, pc_ver = "aux", pc_opts = dict()):

            self.block_la = pc_ver != "direct"

            self.SetUpFWOps(stokes)
            self.SetUpRHS(stokes)
            self.SetUpPC(stokes, pc_ver = pc_ver)

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
                self.A = BilinearForm(stokes.disc.X)
                self.A += stokes.disc.stokesA

                self.B = BilinearForm(trialspace = stokes.disc.X, testspace=disc.X.Q)
                self.B += stokes.disc.stokesB

                self.M = ngs.BlockMatrix( [ [self.A.mat, self.B.mat.T], \
                                            [self.B.mat, None] ] )

                self.Mpre = ngs.BlockMatrix( [ [self.Apre, None,
                                                None, self.Spre] ] )
            else:
                raise "not implemented"

        def SetUpRHS(self, stokes):
            if self.block_la:
                # Right hand side
                self.f = LinearForm(stokes.disc.X)
                self.f += stokes.stokesf

                self.g = LinearForm(stokes.disc.Q)

                self.rhs = ngs.BlockVector( [self.f.vec, \
                                             self.g.vec] )
            else:
                raise "not implemented"
                
        def SetUpDirect(self, stokes, inv_type = None):
            # Direct inverse
            if self.block_la:
                raise "Cannot invert block matrices!"
            else:
                raise "TODO!"
            
        def SetUpAux(self, stokes):
            # Embedded H1 auxiliary space + Block-GS
            raise "TODO!"


        def SetUpFacetAux(self, stokes):
            # Facet wise auxiliary space AMG + Block-GS
            raise "TODO!"
                

        def Assemble(self):
            self.A.Assemble()
            self.B.Assemble()
            self.f.Assemble()
            self.g.Assemble()


    def __init__(self, flow_settings = None, flow_opts = None,
                 disc = None, disc_opts = None):

        # mesh, geometry, physical parameters 
        if flow_settings is not None:
            self.settings = settings
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

        self.InitLinAlg()
        self.AssembleLinAlg()
            
    def InitLinAlg(self):
        self.linalg_stuff = LinAlg(self)
        
    def AssembleLinAlg(self):
        self.linalg_stuff.Assemble()
