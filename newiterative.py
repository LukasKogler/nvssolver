from ngsolve import *
#from ngsolve.krylovspace import MinRes
from ngsolve.la import EigenValues_Preconditioner, ChebyshevIteration

#from mt_global import *
from bramblepasciak import BramblePasciakCG
from minres import MyMinRes

from ngsolve.ngstd import Timer

from ngsolve.utils import TimeFunction

#import ngs_amg

__all__ = ["NavierStokes"]

realcompile = False

class NavierStokes:

    def __init__(self, mesh, nu, inflow, outflow, wall, uin, timestep, order=2, volumeforce=None, hodivfree = False, sym = False):

        if sym:
            self.nu = 2*nu #because of epseps formulation
        else:
            self.nu = nu
            
        self.sym = sym
        self.timestep = timestep
        self.uin = uin
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall
        self.hodivfree = hodivfree
        self.order = order
        V = HDiv(mesh, order=order, dirichlet=inflow + "|" + wall, RT=False, hodivfree = self.hodivfree)
        self.V = V
        Vhat = TangentialFacetFESpace(mesh, order=order - 1, dirichlet=inflow + "|" + wall + "|" + outflow)
        self.Vhat = Vhat
        Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
        

        Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
        Sigma = Compress(Sigma)
        if sym:
            if mesh.dim == 2:
                S = L2(mesh, order=order - 1)
            else:
                S = VectorL2(mesh, order=order - 1)
                
            S.SetCouplingType(IntRange(0, S.ndof), COUPLING_TYPE.HIDDEN_DOF)
            S = Compress(S)

            self.X = FESpace([V, Vhat, Sigma, S])
            self.X2 = FESpace([V, Vhat, Sigma, S])
        else:
            self.X = FESpace([V, Vhat, Sigma])
            self.X2 = FESpace([V, Vhat, Sigma])
            
        for i in range(self.X.ndof):
            if self.X.CouplingType(i) == COUPLING_TYPE.WIREBASKET_DOF:
                self.X.SetCouplingType(i, COUPLING_TYPE.INTERFACE_DOF)
        self.v1dofs = self.X.Range(0)

        #for iterative method
        if True:
            for f in mesh.facets:
                self.X2.SetCouplingType(V.GetDofNrs(f)[1], COUPLING_TYPE.WIREBASKET_DOF)
                self.X2.SetCouplingType(V.ndof + Vhat.GetDofNrs(f)[1], COUPLING_TYPE.WIREBASKET_DOF)

        if sym:
            u, uhat, sigma, W = self.X.TrialFunction()
            v, vhat, tau, R = self.X.TestFunction()
        else:
            u, uhat, sigma = self.X.TrialFunction()
            v, vhat, tau = self.X.TestFunction()

        if mesh.dim == 2:
            def Skew2Vec(m):
                return m[1, 0] - m[0, 1]
        else:
            def Skew2Vec(m):
                return CoefficientFunction((m[0, 1] - m[1, 0], m[2, 0] - m[0, 2], m[1, 2] - m[2, 1]))
            
        dS = dx(element_boundary=True)
        n = specialcf.normal(mesh.dim)

        def tang(u):
            return u - (u * n) * n
        
        self.stokesA = -1/ nu * InnerProduct(sigma, tau) * dx + \
                       (div(sigma) * v + div(tau) * u) * dx + \
                       -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
                       (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS
        
        self.V_trace = nu * div(u) * div(v) * dx
        
        if sym:
            self.stokesA += (InnerProduct(W, Skew2Vec(tau)) + InnerProduct(R, Skew2Vec(sigma))) * dx           

        self.astokes = BilinearForm(self.X, eliminate_hidden=True)
        self.astokes += self.stokesA
        self.astokes += 1e12 * nu * div(u) * div(v) * dx
        self.pre_astokes = Preconditioner(self.astokes, "bddc")

        self.a = BilinearForm(self.X, eliminate_hidden=True)
        self.a += self.stokesA

        self.gfu = GridFunction(self.X)
        self.f = LinearForm(self.X)

        self.mstar = BilinearForm(self.X, eliminate_hidden=True, condense=True)
        self.mstar += u * v * dx + timestep * self.stokesA

        self.premstar = Preconditioner(self.mstar, "bddc")
        #self.mstar.Assemble()
        
        # self.invmstar = self.mstar.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")
        # self.invmstar1 = self.mstar.mat.Inverse(self.X.FreeDofs(self.mstar.condense), inverse="sparsecholesky")

        #self.invmstar1 = CGSolver(self.mstar.mat, pre=self.premstar, precision=1e-4, printrates=False)
        #ext = IdentityMatrix(self.X.ndof) + self.mstar.harmonic_extension
        #extT = IdentityMatrix(self.X.ndof) + self.mstar.harmonic_extension_trans
        #self.invmstar = ext @ self.invmstar1 @ extT + self.mstar.inner_solve

        if False:
            u, v = V.TnT()
            self.conv = BilinearForm(V, nonassemble=True)
            self.conv += SymbolicBFI(InnerProduct(grad(v) * u, u).Compile(True, wait=True))
            self.conv += SymbolicBFI((-IfPos(u * n, u * n * u * v, u * n * u.Other(bnd=self.uin) * v)).Compile(True, wait=True), element_boundary=True)
            emb = Embedding(self.X.ndof, self.v1dofs)
            self.conv_operator = emb @ self.conv.mat @ emb.T
        else:
            VL2 = VectorL2(mesh, order=order, piola=True)
            ul2, vl2 = VL2.TnT()
            self.conv_l2 = BilinearForm(VL2, nonassemble=True)
            self.conv_l2 += InnerProduct(grad(vl2) * ul2, ul2).Compile(realcompile=realcompile, wait=True) * dx
            self.conv_l2 += (-IfPos(ul2 * n, ul2 * n * ul2 * vl2, ul2 * n * ul2.Other(bnd=self.uin) * vl2)).Compile(realcompile=realcompile, wait=True) * dS

            self.convertl2 = V.ConvertL2Operator(VL2) @ Embedding(self.X.ndof, self.v1dofs).T
            self.conv_operator = self.convertl2.T @ self.conv_l2.mat @ self.convertl2

        self.V2 = HDiv(mesh, order=order, RT=False, discontinuous=True)
        if self.hodivfree:
            self.Q = L2(mesh, order=0)            
        else:
            self.Q = L2(mesh, order=order - 1)
            
        self.Qhat = FacetFESpace(mesh, order=order, dirichlet=outflow)
        self.Xproj = FESpace([self.V2, self.Q, self.Qhat])
        (u, p, phat), (v, q, qhat) = self.Xproj.TnT()
        self.aproj = BilinearForm(self.Xproj, condense=True)
        self.aproj += (-u * v + div(u) * q + div(v) * p) * dx + (u * n * qhat + v * n * phat) * dS
        self.cproj = Preconditioner(self.aproj, "bddc", coarsetype="h1amg")
        #self.aproj.Assemble()

        self.gfup = GridFunction(self.Q)
        #self.tmp_projection = self.aproj.mat.CreateColVector()

        # self.invproj1 = aproj.mat.Inverse(self.Xproj.FreeDofs(aproj.condense), inverse="sparsecholesky")

        #self.invproj1 = CGSolver(aproj.mat, pre=cproj, printrates=False)
        #ext = IdentityMatrix() + aproj.harmonic_extension
        #extT = IdentityMatrix() + aproj.harmonic_extension_trans
        #self.invproj = ext @ self.invproj1 @ extT + aproj.inner_solve

        self.bproj = BilinearForm(trialspace=self.V, testspace=self.Xproj)
        self.bproj += SymbolicBFI(div(self.V.TrialFunction()) * q)
        #self.bproj.Assemble()

        # mapping of discontinuous to continuous H(div)
        ind = self.V.ndof * [0]
        for el in mesh.Elements(VOL):
            dofs1 = self.V.GetDofNrs(el)
            dofs2 = self.V2.GetDofNrs(el)
            for d1, d2 in zip(dofs1, dofs2):
                ind[d1] = d2
        self.mapV = PermutationMatrix(self.Xproj.ndof, ind)

        self.h1order = 1
        if mesh.dim == 2:
            self.fesh1_1 = H1(mesh, order=self.h1order, innerorder = 0, dirichlet=inflow + "|" + wall)
            self.fesh1_2 = H1(mesh, order=self.h1order, innerorder = 0, dirichlet=inflow + "|" + wall + "|" + outflow)
            self.fesh1 = FESpace([self.fesh1_1,self.fesh1_2])
        else:
            self.fesh1_1 = H1(mesh, order=self.h1order, innerorder = 0, dirichlet=inflow + "|" + wall)
            self.fesh1_2 = H1(mesh, order=self.h1order, innerorder = 0, dirichlet=inflow + "|" + wall + "|" + outflow)
            self.fesh1_3 = H1(mesh, order=self.h1order,  innerorder = 0, dirichlet=inflow + "|" + wall + "|" + outflow)            
            self.fesh1 = FESpace([self.fesh1_1,self.fesh1_2,self.fesh1_3])

    @property
    def velocity(self):
        return self.gfu.components[0]

    @property
    def pressure(self):
        return -self.gfup
        #return -1e6 / self.nu * div(self.gfu.components[0])

    def InitializeMatrices(self):
        self.mstar.Assemble()
        self.invmstar1 = CGSolver(self.mstar.mat, pre=self.premstar, precision=1e-4, printrates=False)
        ext = IdentityMatrix(self.X.ndof) + self.mstar.harmonic_extension
        extT = IdentityMatrix(self.X.ndof) + self.mstar.harmonic_extension_trans
        self.invmstar = ext @ self.invmstar1 @ extT + self.mstar.inner_solve
        
        self.aproj.Assemble()
        self.tmp_projection = self.aproj.mat.CreateColVector()
        
        self.invproj1 = CGSolver(self.aproj.mat, pre=self.cproj, printrates=False)
        ext = IdentityMatrix() + self.aproj.harmonic_extension
        extT = IdentityMatrix() + self.aproj.harmonic_extension_trans
        self.invproj = ext @ self.invproj1 @ extT + self.aproj.inner_solve

        self.bproj.Assemble()
        
    def SolveInitial(self, timesteps=None, iterative=True, GS = True, use_bddc = False, solver = "BPCG", blocktype = 1, lo_inv = "auxh1", divdivpen = 1):
        self.a.Assemble()
        self.f.Assemble()

        self.gfu.components[0].Set(self.uin, definedon=self.X.mesh.Boundaries(self.inflow))
        self.gfu.components[1].Set(self.uin, definedon=self.X.mesh.Boundaries(self.inflow))

        if not timesteps:

            elinternal = True

            if not elinternal:
                raise Exception("please use elinternal!!!")
            
            use_bddc = use_bddc

            if use_bddc:
                self.X = self.X2
            
            if iterative:                
                p, q = self.Q.TnT()

                if self.sym:
                    u, uhat, sigma, W = self.X.TrialFunction()
                    v, vhat, tau, R = self.X.TestFunction()
                else:
                    u, uhat, sigma = self.X.TrialFunction()
                    v, vhat, tau = self.X.TestFunction()

                if elinternal:
                    if self.hodivfree:
                        store_inner=False
                    else:
                        store_inner=True
                
                blfA = BilinearForm(self.X, eliminate_hidden=True, condense=elinternal, store_inner = store_inner)
                blfA += self.stokesA
                blfA += divdivpen*self.V_trace


                if use_bddc:                    
                    hdiv_bddc = Preconditioner(blfA, type = "bddc")

                    
                blfA.Assemble()
                
                g = LinearForm(self.Q)
                g.Assemble()

                mp = BilinearForm(self.Q)
                mp +=  1/(self.nu*divdivpen) * p * q * dx
                preM = Preconditioner(mp, 'local')
                mp.Assemble()

                blfB = BilinearForm(trialspace=self.X, testspace=self.Q)
                blfB += div(u) * q * dx
                blfB.Assemble()

                mesh = self.gfu.space.mesh
                
                Vlo = HDiv(mesh, order=1) #, hodivfree = self.hodivfree)
                Vhatlo = TangentialFacetFESpace(mesh, order=1)

                if (self.h1order==1):
                    Xlo = FESpace([Vlo,Vhatlo])
                    (ulo,uhatlo), (vlo,vhatlo) = Xlo.TnT()
                                                           
                    ind = Xlo.ndof * [0]
                    
                    for f in mesh.facets:
                        dofs1_div = self.V.GetDofNrs(f)
                        dofs1_facet = self.Vhat.GetDofNrs(f)
                        
                        dofs2_div = Vlo.GetDofNrs(f)
                        dofs2_facet = Vhatlo.GetDofNrs(f)

                        off1 = self.V.ndof
                        off2 = Vlo.ndof
                        if mesh.dim == 2:
                            for i in range(len(dofs2_div)):
                                ind[dofs2_div[i]] = dofs1_div[i]
                            for i in range(len(dofs2_facet)):
                                ind[dofs2_facet[i]+off2] = dofs1_facet[i] + off1
                        else:
                            hdiv_offset = (self.order+1)*(self.order)//2
                            #hdiv-dofs
                            ind[dofs2_div[0]] = dofs1_div[0] 
                            ind[dofs2_div[1]] = dofs1_div[1]
                            ind[dofs2_div[2]] = dofs1_div[1+hdiv_offset]
                            #facet-dofs
                            for i in range(4):
                                ind[dofs2_facet[i]+off2] = dofs1_facet[i] + off1
                            facet_offset = self.order #actually order+1 but facet order is set to order-1
                            ind[dofs2_facet[4]+off2] = dofs1_facet[2*facet_offset] + off1
                            ind[dofs2_facet[5]+off2] = dofs1_facet[2*facet_offset+1] + off1
                    
                    lo_to_high = PermutationMatrix(self.X.ndof, ind)                    
                else:
                    Xlo = self.X
                    ulo = u
                    vlo = v
                    uhatlo = uhat
                    vhatlo = vhat
                    lo_to_high = IdentityMatrix()
                
                ###### Trafo from H1 space ######
                #amixed = BilinearForm(trialspace=self.fesh1, testspace=self.X)
                #acomp = BilinearForm(self.X)
                amixed = BilinearForm(trialspace=self.fesh1, testspace=Xlo)
                acomp = BilinearForm(Xlo)

                #vdual = v.Operator("dual")
                vdual = vlo.Operator("dual")
                                                
                if mesh.dim == 2:
                    (uh1_1,uh1_2) = self.fesh1.TrialFunction()
                    uh1 = CoefficientFunction((uh1_1,uh1_2))
                else:
                    (uh1_1,uh1_2, uh1_3) = self.fesh1.TrialFunction()
                    uh1 = CoefficientFunction((uh1_1, uh1_2, uh1_3))
                
                dS = dx(element_boundary=True)

                mesh = self.gfu.space.mesh
                n = specialcf.normal(mesh.dim)

                def tang(u):
                    return u - (u * n) * n
                
                if not elinternal:
                    acomp += ulo*vdual * dx
                acomp += ulo*vdual * dS
                acomp += tang(uhatlo)*tang(vhatlo) * dS                
                acomp.Assemble()

                if not elinternal:
                    amixed += uh1*vdual * dx
                
                amixed += uh1*vdual * dS
                amixed += uh1*tang(vhatlo) * dS
                amixed.Assemble()
                
                amixed_matT = amixed.mat.CreateTranspose()

                eblocks = []            
                for f in mesh.facets: #edges in 2d, faces in 3d
                    eblocks.append ( Xlo.GetDofNrs(f)  )

                #einv = acomp.mat.CreateBlockSmoother(eblocks)

                '''
                if not elinternal:
                    acomp += u*vdual * dx
                acomp += u*vdual * dS
                acomp += tang(uhat)*tang(vhat) * dS                
                acomp.Assemble()

                if not elinternal:
                    amixed += uh1*vdual * dx
                
                amixed += uh1*vdual * dS
                amixed += uh1*tang(vhat) * dS
                amixed.Assemble()

                eblocks = []            
                for f in mesh.facets: #edges in 2d, faces in 3d
                    eblocks.append ( self.X.GetDofNrs(f)  )

                einv = acomp.mat.CreateBlockSmoother(eblocks)
                '''
                
                fblocks = []
                if not elinternal:
                    if mesh.dim == 3:
                        # (el dofs - V-dofs - ed.dofs - facet.dofs)
                        raise Exception("please use elinternal or change fblocks to inner dofs")                   
                    for f in mesh.faces:
                        # remove hidden dofs (=-2)
                        fblocks.append ( [d for d in self.X.GetDofNrs(f) if d != -2] )
                                
                    #finv = acomp.mat.CreateBlockSmoother(fblocks)
                                
                class MyBasisTrafo(BaseMatrix):
                    def __init__ (self, mat, eblocks, fblocks):
                        super(MyBasisTrafo, self).__init__()
                        self.mat = mat
                        self.einv = mat.CreateBlockSmoother(eblocks)
                        if not elinternal:
                            self.finv = mat.CreateBlockSmoother(fblocks)

                        self.etimer = Timer("myembedding")
                        self.etimer_t = Timer("myembedding_trans")
                    def Mult(self, x, y):
                        self.etimer.Start()
                        if not elinternal:
                            res = self.mat.CreateColVector()
                            y.data = self.einv * x
                            res.data = x - self.mat * y
                            y.data += self.finv * res
                        else:
                            res = self.mat.CreateColVector()
                            res.data = amixed.mat * x
                            y.data = self.einv * res
                        self.etimer.Stop()

                    def MultTrans(self, x, y):
                        self.etimer_t.Start()
                        if not elinternal:
                            res = self.mat.CreateColVector()
                            y.data = self.finv.T * x
                            res.data = x - self.mat.T * y
                            y.data += self.einv.T * res
                        else:
                            res = self.mat.CreateColVector()
                            
                            res.data = self.einv.T * x
                            y.data = amixed_matT * res
                        self.etimer_t.Stop()
                        
                    def CreateColVector(self):
                        return acomp.mat.CreateColVector()

                    def CreateRowVector(self):
                        return amixed.mat.CreateRowVector()
            
                trafo = MyBasisTrafo(acomp.mat, eblocks, fblocks)
                
                #transform = (lo_to_high.T @ trafo @amixed.mat)
                #transform = (trafo @amixed.mat)
                transform = (lo_to_high.T @ trafo)
                #transform = trafo

                #direct epseps+preconditioner
                epseps = False
                
                if mesh.dim ==2 :                                        
                    if epseps:
                        (uh1_1,uh1_2),(vh1_1,vh1_2) = self.fesh1.TnT()
                        
                        grad_u = CoefficientFunction((grad(uh1_1),grad(uh1_2)), dims=(2,2))
                        grad_v = CoefficientFunction((grad(vh1_1),grad(vh1_2)), dims=(2,2))
                        div_u = grad_u[0,0] + grad_u[1,1]
                        div_v = grad_v[0,0] + grad_v[1,1]
                        eps_u = 0.5 * (grad_u + grad_u.trans)
                        eps_v = 0.5 * (grad_v + grad_v.trans)
                    
                        aH1 = BilinearForm(self.fesh1)
                        aH1 += self.nu * InnerProduct(eps_u,eps_v) * dx
                        #aH1 += (self.nu * InnerProduct(grad_u,grad_v) + self.nu*1e6 * div_u*div_v) * dx
                        preAh1 = Preconditioner(aH1, 'direct', inverse = "sparsecholesky")
                        aH1.Assemble()
                    else:
                        uh1_1,vh1_1 = self.fesh1_1.TnT()
                        uh1_2,vh1_2 = self.fesh1_2.TnT()
                        aH1_1 = BilinearForm(self.fesh1_1)                    
                        aH1_1 += self.nu * InnerProduct(grad(uh1_1),grad(vh1_1)) * dx

                        
                        aH1_2 = BilinearForm(self.fesh1_2)
                        aH1_2 += self.nu * InnerProduct(grad(uh1_2),grad(vh1_2)) * dx

                        if self.h1order > 1:
                            preAh1_1 = Preconditioner(aH1_1, 'bddc', coarsetype="h1amg")
                            preAh1_2 = Preconditioner(aH1_2, 'bddc', coarsetype="h1amg")
                        else:
                            #if mpi:
                            #preAh1_1 = Preconditioner(aH1_1, "ngs_amg.h1_scal", ngs_amg_log_level = 2, ngs_amg_log_file = "")
                            #preAh1_2 = Preconditioner(aH1_2, "ngs_amg.h1_scal", ngs_amg_log_level = 2, ngs_amg_log_file = "")
                            #else:
                            preAh1_1 = Preconditioner(aH1_1, 'h1amg')
                            preAh1_2 = Preconditioner(aH1_2, 'h1amg')
                            #preAh1_1 = Preconditioner(aH1_1, 'direct', inverse = "sparsecholesky")
                            #preAh1_2 = Preconditioner(aH1_2, 'direct', inverse = "sparsecholesky")

                        aH1_1.Assemble()
                        aH1_2.Assemble()

                        emb_comp1 = Embedding(self.fesh1.ndof,self.fesh1.Range(0))
                        emb_comp2 = Embedding(self.fesh1.ndof,self.fesh1.Range(1))

                        preAh1 = emb_comp1 @ preAh1_1 @ emb_comp1.T + emb_comp2 @ preAh1_2 @ emb_comp2.T
                        
                else:
                    if epseps:
                        (uh1_1,uh1_2, uh1_3),(vh1_1,vh1_2, vh1_3) = self.fesh1.TnT()
                        grad_u = CoefficientFunction((grad(uh1_1),grad(uh1_2),grad(uh1_3)), dims=(3,3))
                        grad_v = CoefficientFunction((grad(vh1_1),grad(vh1_2),grad(vh1_3)), dims=(3,3))

                        eps_u = 0.5 * (grad_u + grad_u.trans)
                        eps_v = 0.5 * (grad_v + grad_v.trans)
                    
                        aH1 = BilinearForm(self.fesh1)
                        aH1 += self.nu * InnerProduct(eps_u,eps_v) * dx
                        preAh1 = Preconditioner(aH1, 'direct', inverse = "sparsecholesky")
                        aH1.Assemble()
                    else:
                        uh1_1,vh1_1 = self.fesh1_1.TnT()
                        uh1_2,vh1_2 = self.fesh1_2.TnT()
                        uh1_3,vh1_3 = self.fesh1_3.TnT()
                    
                        aH1_1 = BilinearForm(self.fesh1_1)
                        aH1_1 += self.nu * InnerProduct(grad(uh1_1),grad(vh1_1)) * dx

                        aH1_2 = BilinearForm(self.fesh1_2)
                        aH1_2 += self.nu * InnerProduct(grad(uh1_2),grad(vh1_2)) * dx

                        aH1_3 = BilinearForm(self.fesh1_3)
                        aH1_3 += self.nu * InnerProduct(grad(uh1_3),grad(vh1_3)) * dx

                        if self.h1order > 1:
                            preAh1_1 = Preconditioner(aH1_1, 'bddc', coarsetype="h1amg" )
                            preAh1_2 = Preconditioner(aH1_2, 'bddc', coarsetype="h1amg" )
                            preAh1_3 = Preconditioner(aH1_3, 'bddc', coarsetype="h1amg" )
                        else:
                            preAh1_1 = Preconditioner(aH1_1, 'h1amg')
                            preAh1_2 = Preconditioner(aH1_2, 'h1amg')
                            preAh1_3 = Preconditioner(aH1_3, 'h1amg')
                        aH1_1.Assemble()
                        aH1_2.Assemble()
                        aH1_3.Assemble()

                        emb_comp1 = Embedding(self.fesh1.ndof,self.fesh1.Range(0))
                        emb_comp2 = Embedding(self.fesh1.ndof,self.fesh1.Range(1))
                        emb_comp3 = Embedding(self.fesh1.ndof,self.fesh1.Range(2))

                        preAh1 = emb_comp1 @ preAh1_1 @ emb_comp1.T + emb_comp2 @ preAh1_2 @ emb_comp2.T + emb_comp3 @ preAh1_3 @ emb_comp3.T

                # BlockJacobi for H(div)-velocity space
                blocks = []
                #blocktype = 1
                if blocktype == 1:
                    for e in mesh.facets:
                        blocks.append ( [d for d in self.X.GetDofNrs(e) if self.X.FreeDofs(elinternal)[d]])                    
                            
                elif blocktype == 2:
                    for e in mesh.Elements():
                        el_block = []                        
                        for f in e.facets:
                            el_block += [d for d in self.X.GetDofNrs(f) if self.X.FreeDofs(elinternal)[d]]
                        blocks.append(el_block)                        
                elif blocktype == 3:                    
                    for v in mesh.vertices:
                        el_block = []
                        if mesh.dim == 2:
                            for e in v.edges:
                                el_block += [d for d in self.X.GetDofNrs(e) if self.X.FreeDofs(elinternal)[d]]
                        else:
                            for e in v.faces:
                                el_block += [d for d in self.X.GetDofNrs(e) if self.X.FreeDofs(elinternal)[d]]
                        blocks.append ( el_block)
                elif blocktype == 4:
                    for e in mesh.edges:
                        el_block = []
                        for f in e.faces:
                            el_block += [d for d in self.X.GetDofNrs(f) if self.X.FreeDofs(elinternal)[d]]
                        blocks.append ( el_block)
                
                blocks = [x for x in blocks if len(x)]


                lin_dofs = BitArray(self.X.ndof)
                lin_dofs.Clear()
                #lin_dofs2 = BitArray(self.X.ndof)
                #lin_dofs2.Clear()
                
                #for i in ind:
                #    if self.X.FreeDofs(elinternal)[i]:
                #        lin_dofs2.Set(i)
                
                #print(lin_dofs2)

                hdiv_offset = (self.order+1)*(self.order)//2
                
                for f in mesh.facets:
                    dofs = self.V.GetDofNrs(f)
                    fac_dofs = self.Vhat.GetDofNrs(f)

                    if mesh.dim == 2:
                        if self.X.FreeDofs(elinternal)[dofs[0]]:
                            lin_dofs.Set(dofs[0])
                            lin_dofs.Set(dofs[1])
                        if self.X.FreeDofs(elinternal)[fac_dofs[0]+self.V.ndof]:
                            lin_dofs.Set(fac_dofs[0]+self.V.ndof)
                            lin_dofs.Set(fac_dofs[1]+self.V.ndof)
                    else:
                        if self.X.FreeDofs(elinternal)[dofs[0]]:
                            lin_dofs.Set(dofs[0])
                            lin_dofs.Set(dofs[1])
                            lin_dofs.Set(dofs[1+hdiv_offset])
                            
                        if self.X.FreeDofs(elinternal)[fac_dofs[0]+self.V.ndof]:
                            for i in range(4):
                                lin_dofs.Set(fac_dofs[i]+self.V.ndof)
                            facet_offset = self.order #actually order+1 but facet order is set to order-1
                            lin_dofs.Set(fac_dofs[2*facet_offset]+self.V.ndof)
                            lin_dofs.Set(fac_dofs[2*facet_offset+1]+self.V.ndof)

                ####
                
                
                class MypreA(BaseMatrix):
                    def __init__ (self, space, a, jacblocks, GS, loinv):                        
                        super(MypreA, self).__init__()
                        self.space = space
                        self.mat = a.mat
                        self.temp = a.mat.CreateColVector()
                        self.GS = GS

                        
                        self.loinv = loinv
                        #have to switch to full diagonal if no el internal
                        if not elinternal:
                            self.jacobi = a.mat.CreateSmoother(a.space.FreeDofs())
                        else:
                            if True:
                                # Block Jacobi
                                self.jacobi = a.mat.CreateBlockSmoother(jacblocks)
                            else:
                                # Jacobi, diagonal...                                
                                self.jacobi = a.mat.CreateSmoother(a.space.FreeDofs(True))

                        print("WARNING: eigvals for cheby hardcoded!!!")
                        self.cheby = ChebyshevIteration(pre = self.jacobi,mat = self.mat, steps = 5, lam_min = 1-3.8, lam_max = 1-0.007)
                        
                    def Mult(self, x, y):
                        if self.GS:
                            '''
                            # first step has not an updated res
                            
                            self.temp.data = ((transform @ preAh1 @ transform.T)) * x
                            y[:] = 0
                            self.jacobi.Smooth(y,x)
                            self.jacobi.SmoothBack(y,x)
                            
                            y.data +=  self.temp
                            '''
                            # this is multiplicative, and symmetric.
                            # moving Aux-Pre to the end would not be sym anymore
                            # GS with simple Jacobi (and not block jac) is not shared parallel
                            y[:] = 0
                            self.jacobi.Smooth(y,x)

                            self.temp.data = x - self.mat * y
                            y.data += self.loinv * self.temp
                            #y.data += ((transform @ preAh1 @ transform.T)) * self.temp


                            self.jacobi.SmoothBack(y,x)
                            
                        else:
                            
                            #y.data = (self.loinv + self.jacobi) * x
                            
                            y.data = (self.loinv + self.cheby) * x
                            #self.temp.data = x - self.mat * y
                            #y.data += (self.loinv) * self.temp
                            #self.temp.data = x - self.mat * y
                            #y.data += (self.cheby) * self.temp
                            #y.data = (1/2*self.loinv + 1/12*self.jacobi) * x
                            #y.data = self.jacobi * x
                            
                    def Height(self):
                        return self.space.ndof

                    def Width(self):
                        return self.space.ndof

                    def CreateColVector(self):
                        return self.mat.CreateColVector()

                if use_bddc:
                    preA = hdiv_bddc
                else:
                    if lo_inv == "auxh1":
                        auxh1 = ((transform @ preAh1 @ transform.T))
                        preA = MypreA(self.X, blfA, blocks, GS = GS, loinv = auxh1)
                    elif lo_inv == "Aloinv":                        
                        A_loinv = blfA.mat.Inverse(lin_dofs, inverse = "sparsecholesky")
                        preA = MypreA(self.X, blfA, blocks, GS = GS, loinv = A_loinv)
                        
                sol = BlockVector([self.gfu.vec, self.gfup.vec])

                if (self.hodivfree and elinternal):
                    self.f.vec.data += blfA.harmonic_extension_trans * self.f.vec              

                if not self.hodivfree:
                    #a_inv = blfA.mat.Inverse(self.X.FreeDofs(True), inverse = "umfpack")
                    #mat2 = blfB.mat @ (((IdentityMatrix(blfA.mat.height) + blfA.harmonic_extension) @ (a_inv) @ (IdentityMatrix(blfA.mat.height) + blfA.harmonic_extension_trans)) + blfA.inner_solve ) @ blfB.mat.T
                    # B \hat A ^ {-1} B^T
                    mat2 = blfB.mat @ (((IdentityMatrix(blfA.mat.height) + blfA.harmonic_extension) @ (preA) @ (IdentityMatrix(blfA.mat.height) + blfA.harmonic_extension_trans)) + blfA.inner_solve ) @ blfB.mat.T
                else:                    
                    mat2 = blfB.mat @ preA @ blfB.mat.T
                    
                    

                #Qinv = mp.mat.Inverse(self.Q.FreeDofs(), inverse="sparsecholesky")
                #lams = EigenValues_Preconditioner(mat=mat2, pre=mQinv, tol=1e-10)
                #print(lams)
                if True:                                        
                    lams = EigenValues_Preconditioner(mat=mat2, pre=preM, tol=1e-10)
                    import pickle
                    pickle.dump([l for l in lams], open("lams.out", "wb"))  
                    print(lams)
                    print("###############################")
                    print("condition Shat", max(lams) / min(lams))
                    print("max(lams) = ", max(lams))
                    print("min(lams) = ", min(lams))
                    print("###############################")
                    
                    lams = EigenValues_Preconditioner(mat=blfA.mat, pre=preA, tol=1e-10)                    
                    print("###############################")
                    print("condition Ahat", max(lams) / min(lams))
                    print("max(lams) = ", max(lams))
                    print("min(lams) = ", min(lams))
                    print("###############################")
                    #exit()
                    
                if solver == "BPCG":
                    it, t_prep, t_it = BramblePasciakCG(blfA, blfB, None, self.f.vec, g.vec, preA, preM, sol, initialize=False, tol=1e-10, maxsteps=1000, rel_err=True, staticcond = (self.hodivfree and elinternal))
                else:
                    if not self.hodivfree:
                        full_amat =  (IdentityMatrix(blfA.mat.height) - blfA.harmonic_extension_trans) @ (blfA.mat + blfA.inner_matrix) @ (IdentityMatrix(blfA.mat.height) - blfA.harmonic_extension)
                        full_preA = ((IdentityMatrix(blfA.mat.height) + blfA.harmonic_extension) @ (preA) @ (IdentityMatrix(blfA.mat.height) + blfA.harmonic_extension_trans)) + blfA.inner_solve
                    else:
                        full_amat = blfA.mat
                        full_preA = preA
                        
                    big_blf = BlockMatrix([[full_amat,blfB.mat.T],
                                               [blfB.mat,None]])
                    
                    big_rhs = BlockVector([self.f.vec, g.vec])

                    SetNumThreads(4)
                    if solver == "MinRes":
                        big_pre = BlockMatrix([[full_preA,None],
                                               [None,preM]])
                    
                        it, t_prep, t_it = MyMinRes(mat=big_blf, rhs=big_rhs, sol=sol, pre=big_pre, initialize=False, maxsteps=1000, tol = 1e-10, printrates = True)
                    elif solver == "GMRes":
                        pre_offdiag = None # -preM @ blfB.mat @ full_preA
                        big_pre = BlockMatrix([[full_preA,None],
                                               [ pre_offdiag ,preM]])
                        
                        it, t_prep, t_it = (1,1,1)
                        timer_gmres = Timer("Timer-GMRes")
                        timer_gmres.Start()
                        solvers.GMRes(A=big_blf, b=big_rhs, x=sol, pre=big_pre, maxsteps=1000, tol = 1e-10, printrates = True)
                        timer_gmres.Stop()
                        t_it = timer_gmres.time
                    else:
                        raise Exception("solver not available")
                    
                if (self.hodivfree and elinternal):
                    self.gfu.vec.data +=blfA.inner_solve * self.f.vec
                    self.gfu.vec.data +=blfA.harmonic_extension * self.gfu.vec
                
                return it, t_prep, t_it, self.X.ndof + self.Q.ndof
            else:
                self.astokes.Assemble()
                temp = self.astokes.mat.CreateColVector()
                temp.data = -self.astokes.mat * self.gfu.vec + self.f.vec
                inv = self.astokes.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")

                self.gfu.vec.data += inv * temp
                self.gfup.Set(1e6 / self.nu * div(self.gfu.components[0]))
                #self.gfup.vec.data = 1e6 / self.nu * div(self.gfu.components[0])
                
        else:
            self.Project(self.gfu.vec[0:self.V.ndof])
            for it in range(timesteps):
                print("it =", it)
                self.temp = self.a.mat.CreateColVector()
                self.temp2 = self.a.mat.CreateColVector()
                # self.f.Assemble()
                # self.temp.data = self.conv_operator * self.gfu.vec
                # self.temp.data += self.f.vec
                self.temp.data = -self.a.mat * self.gfu.vec

                self.temp2.data = self.invmstar * self.temp
                self.Project(self.temp2[0:self.V.ndof])
                self.gfu.vec.data += self.timestep * self.temp2.data
                self.Project(self.gfu.vec[0:self.V.ndof])

    def AddForce(self, force):
        force = CoefficientFunction(force)
        if self.sym:
            v, vhat, tau, R = self.X.TestFunction()
        else:
            v, vhat, tau = self.X.TestFunction()
        self.f += SymbolicLFI(force * v)

    def DoTimeStep(self):

        self.temp = self.a.mat.CreateColVector()
        self.temp2 = self.a.mat.CreateColVector()
        self.f.Assemble()
        self.temp.data = self.conv_operator * self.gfu.vec
        self.temp.data += self.f.vec
        self.temp.data += -self.a.mat * self.gfu.vec

        self.temp2.data = self.invmstar * self.temp
        self.Project(self.temp2[0:self.V.ndof])
        self.gfu.vec.data += self.timestep * self.temp2.data

    def Project(self,vel):
        self.tmp_projection.data = (self.invproj @ self.bproj.mat) * vel
        self.gfup.vec.data = self.tmp_projection[self.V2.ndof:self.V2.ndof+ self.Q.ndof]
        vel.data -= self.mapV * self.tmp_projection
        #vel.data -= (self.mapV @ self.invproj @ self.bproj.mat) * vel
