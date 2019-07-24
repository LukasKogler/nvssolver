from ngsolve import *
#from ngsolve.krylovspace import MinRes
from ngsolve.la import EigenValues_Preconditioner

#from mt_global import *
from bramblepasciak import BramblePasciakCG

__all__ = ["NavierStokes"]

realcompile = False

class NavierStokes:

    def __init__(self, mesh, nu, inflow, outflow, wall, uin, timestep, order=2, volumeforce=None):

        self.nu = nu
        self.timestep = timestep
        self.uin = uin
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall

        V = HDiv(mesh, order=order, dirichlet=inflow + "|" + wall, RT=False)
        self.V = V
        Vhat = VectorFacet(mesh, order=order - 1, dirichlet=inflow + "|" + wall + "|" + outflow)
        Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)
        if mesh.dim == 2:
            S = L2(mesh, order=order - 1)
        else:
            S = VectorL2(mesh, order=order - 1)

        Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
        Sigma = Compress(Sigma)
        S.SetCouplingType(IntRange(0, S.ndof), COUPLING_TYPE.HIDDEN_DOF)
        S = Compress(S)

        self.X = FESpace([V, Vhat, Sigma, S])
        for i in range(self.X.ndof):
            if self.X.CouplingType(i) == COUPLING_TYPE.WIREBASKET_DOF:
                self.X.SetCouplingType(i, COUPLING_TYPE.INTERFACE_DOF)
        self.v1dofs = self.X.Range(0)

        #for iterative method
        self.X2 = FESpace([V, Vhat, Sigma, S])
        for f in mesh.facets:
            self.X2.SetCouplingType(V.GetDofNrs(f)[1], COUPLING_TYPE.WIREBASKET_DOF)
            self.X2.SetCouplingType(V.ndof + Vhat.GetDofNrs(f)[1], COUPLING_TYPE.WIREBASKET_DOF)
        
        u, uhat, sigma, W = self.X.TrialFunction()
        v, vhat, tau, R = self.X.TestFunction()

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

        self.stokesA = -0.5 / nu * InnerProduct(sigma, tau) * dx + \
                       (div(sigma) * v + div(tau) * u) * dx + \
                       (InnerProduct(W, Skew2Vec(tau)) + InnerProduct(R, Skew2Vec(sigma))) * dx + \
                       -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
                       (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS

        self.V_trace = 2 * nu * div(u) * div(v) * dx

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
        
        if mesh.dim == 2:
            self.fesh1_1 = H1(mesh, order=order, dirichlet=inflow + "|" + wall)
            self.fesh1_2 = H1(mesh, order=order, dirichlet=inflow + "|" + wall + "|" + outflow)            
            self.fesh1 = FESpace([self.fesh1_1,self.fesh1_2])
        else:
            self.fesh1_1 = H1(mesh, order=1, dirichlet=inflow + "|" + wall)
            self.fesh1_2 = H1(mesh, order=1, dirichlet=inflow + "|" + wall + "|" + outflow)
            self.fesh1_3 = H1(mesh, order=1, dirichlet=inflow + "|" + wall + "|" + outflow)            
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
        
    def SolveInitial(self, timesteps=None, iterative=True):
        self.a.Assemble()
        self.f.Assemble()

        self.gfu.components[0].Set(self.uin, definedon=self.X.mesh.Boundaries(self.inflow))
        self.gfu.components[1].Set(self.uin, definedon=self.X.mesh.Boundaries(self.inflow))

        if not timesteps:
           
            #temp2 = self.a.mat.CreateColVector()

            # False does not work anymore!
            elinternal = True
            
            if iterative:                
                p, q = self.Q.TnT()
                u, uhat, sigma, W = self.X2.TrialFunction()
                v, vhat, tau, R = self.X2.TestFunction()
                #u, v = self.X.TnT()

                blfA = BilinearForm(self.X2, eliminate_hidden=True, condense=elinternal, store_inner = elinternal)
                blfA += self.stokesA
                blfA += self.V_trace

                blfA.Assemble()
                
                g = LinearForm(self.Q)
                g.Assemble()

                mp = BilinearForm(self.Q)
                mp +=  1/(2 * self.nu) * p * q * dx
                preM = Preconditioner(mp, 'local')
                mp.Assemble()

                blfB = BilinearForm(trialspace=self.X2, testspace=self.Q)
                blfB += div(u) * q * dx
                blfB.Assemble()

                ###### Trafo to H1 space ######
                amixed = BilinearForm(trialspace=self.fesh1, testspace=self.X2)
                acomp = BilinearForm(self.X2)

                vdual = v.Operator("dual")
                mesh = self.gfu.space.mesh

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
                    eblocks.append ( self.X2.GetDofNrs(f)  )

                einv = acomp.mat.CreateBlockSmoother(eblocks)
                
                fblocks = []
                if not elinternal:
                    if mesh.dim == 3:
                        # (el dofs - V-dofs - ed.dofs - facet.dofs)
                        raise Exception("please use elinternal or change fblocks to inner dofs")                   
                    for f in mesh.faces:
                        # remove hidden dofs (=-2)
                        fblocks.append ( [d for d in self.X2.GetDofNrs(f) if d != -2] )
                                
                    finv = acomp.mat.CreateBlockSmoother(fblocks)
                                
                class MyBasisTrafo(BaseMatrix):
                    def __init__ (self, mat, eblocks, fblocks):
                        super(MyBasisTrafo, self).__init__()
                        self.mat = mat
                        self.einv = mat.CreateBlockSmoother(eblocks)
                        if not elinternal:
                            self.finv = mat.CreateBlockSmoother(fblocks)

                    def Mult(self, x, y):
                        if not elinternal:
                            res = self.mat.CreateColVector()
                            y.data = self.einv * x
                            res.data = x - self.mat * y
                            y.data += finv * res
                        else:
                            y.data = self.einv * x

                    def MultTrans(self, x, y):
                        if not elinternal:
                            res = self.mat.CreateColVector()
                            y.data = self.finv.T * x
                            res.data = x - self.mat.T * y
                            y.data += einv.T * res
                        else:
                            y.data = einv.T * x

                trafo = MyBasisTrafo(acomp.mat, eblocks, fblocks)
                transform = (trafo@amixed.mat)

                if mesh.dim ==2 :
                    uh1_1,vh1_1 = self.fesh1_1.TnT()
                    uh1_2,vh1_2 = self.fesh1_2.TnT()
    
                    aH1_1 = BilinearForm(self.fesh1_1)
                    aH1_1 += 2*self.nu * InnerProduct(grad(uh1_1),grad(vh1_1)) * dx

                    aH1_2 = BilinearForm(self.fesh1_2)
                    aH1_2 += 2*self.nu * InnerProduct(grad(uh1_2),grad(vh1_2)) * dx

                    #preAh1_1 = Preconditioner(aH1_1, 'bddc')
                    preAh1_1 = Preconditioner(aH1_1, 'h1amg')
                    aH1_1.Assemble()

                    #preAh1_2 = Preconditioner(aH1_2, 'bddc')
                    preAh1_2 = Preconditioner(aH1_2, 'h1amg')
                    aH1_2.Assemble()

                    emb_comp1 = Embedding(self.fesh1.ndof,self.fesh1.Range(0))
                    emb_comp2 = Embedding(self.fesh1.ndof,self.fesh1.Range(1))
    
                    preAh1 = emb_comp1 @ preAh1_1 @ emb_comp1.T + emb_comp2 @ preAh1_2 @ emb_comp2.T
                else:
                    uh1_1,vh1_1 = self.fesh1_1.TnT()
                    uh1_2,vh1_2 = self.fesh1_2.TnT()
                    uh1_3,vh1_3 = self.fesh1_3.TnT()
    
                    aH1_1 = BilinearForm(self.fesh1_1)
                    aH1_1 += 2*self.nu * InnerProduct(grad(uh1_1),grad(vh1_1)) * dx

                    aH1_2 = BilinearForm(self.fesh1_2)
                    aH1_2 += 2*self.nu * InnerProduct(grad(uh1_2),grad(vh1_2)) * dx

                    aH1_3 = BilinearForm(self.fesh1_3)
                    aH1_3 += 2*self.nu * InnerProduct(grad(uh1_3),grad(vh1_3)) * dx
    
                    preAh1_1 = Preconditioner(aH1_1, 'h1amg')
                    aH1_1.Assemble()

                    preAh1_2 = Preconditioner(aH1_2, 'h1amg')
                    aH1_2.Assemble()

                    preAh1_3 = Preconditioner(aH1_3, 'h1amg')
                    aH1_3.Assemble()
                    
                    emb_comp1 = Embedding(self.fesh1.ndof,self.fesh1.Range(0))
                    emb_comp2 = Embedding(self.fesh1.ndof,self.fesh1.Range(1))
                    emb_comp3 = Embedding(self.fesh1.ndof,self.fesh1.Range(2))
    
                    preAh1 = emb_comp1 @ preAh1_1 @ emb_comp1.T + emb_comp2 @ preAh1_2 @ emb_comp2.T + emb_comp3 @ preAh1_3 @ emb_comp3.T

                # BlockJacobi for H(div)-velocity space
                blocks = []
                for e in mesh.facets:
                    blocks.append ( [d for d in self.X2.GetDofNrs(e) if self.X2.FreeDofs(elinternal)[d]])
                
                class MypreA(BaseMatrix):
                    def __init__ (self, space, a, jacblocks, GS):                        
                        super(MypreA, self).__init__()
                        self.space = space
                        self.mat = a.mat
                        self.temp = a.mat.CreateColVector()
                        self.GS = GS

                        #have to switch to full diagonal if no el internal
                        if not elinternal:
                            self.jacobi = a.mat.CreateSmoother(a.space.FreeDofs())
                        else:
                            self.jacobi = a.mat.CreateBlockSmoother(jacblocks)

                    def Mult(self, x, y):
                        if self.GS:
                            y[:] = 0
                            self.jacobi.Smooth(y,x)
                            self.temp.data = x - self.mat * y
                            y.data += ((transform @ preAh1 @ transform.T)) * self.temp            
                            self.jacobi.SmoothBack(y,x)
                        else:                                                
                            y.data = ((transform @ preAh1 @ transform.T) + self.jacobi) * x
                            #y.data = self.jacobi * x

                    def Height(self):
                        return self.space.ndof

                    def Width(self):
                        return self.space.ndof

                    def CreateColVector(self):
                        return self.mat.CreateColVector()
                    
                preA = MypreA(self.X2, blfA, blocks, GS = True)

                # for testing
                #preA= blfA.mat.Inverse(self.X2.FreeDofs(elinternal), inverse = "pardiso")

                #######################                          
                #mat1 = blfB.mat @ preA @ blfB.mat.T                               
                #lams = EigenValues_Preconditioner(mat=mat1, pre=preM, tol=1e-3)
                #print(lams)
                #######################                          
                sol = BlockVector([self.gfu.vec, self.gfup.vec])                
                BramblePasciakCG(blfA, blfB, None, self.f.vec, g.vec, preA, preM, sol, initialize=False, tol=1e-10, maxsteps=2000, rel_err=True)                                
                
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
        v, vhat, tau, R = self.X.TestFunction()
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
