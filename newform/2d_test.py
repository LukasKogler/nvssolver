from ngsolve import *
from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions
from netgen.geom2d import unit_square
from ngsolve.la import EigenValues_Preconditioner

mesh = Mesh(unit_square.GenerateMesh(maxh = 0.1))
Draw(mesh)

nu = 1
order = 2
        
VT = HDiv(mesh, order=order, discontinuous = True)
VT2 = HDiv(mesh, order=order, discontinuous = False, dirichlet="bottom|right|top|left")
Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet="bottom|right|top|left")
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)

#Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
#Sigma = Compress(Sigma)

QT = L2(mesh, order = order-1, lowest_order_wb = False)
Qhat = FacetFESpace(mesh, order = order) ###???????????

V = FESpace([VT,Vhat, Sigma])
V2 = FESpace([VT2,Vhat, Sigma])
Q = FESpace([QT, Qhat])
#V = FESpace([VT,Vhat, Sigma, QT, Qhat])
   
(u,uhat, sigma),(v,vhat, tau) = V.TnT()

(p,phat),(q,qhat) = Q.TnT()

dS = dx(element_boundary=True)
        
n = specialcf.normal(2)
h = specialcf.mesh_size

def tang(v):
    return v - (v*n)*n

a= BilinearForm(V, condense = False, eliminate_hidden=False, store_inner = False)
a += -1/ nu * InnerProduct(sigma, tau) * dx + \
 (div(sigma) * v + div(tau) * u) * dx + \
 -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
 (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS \
 + nu* div(u) * div(v) * dx

f = LinearForm(V)
f += CoefficientFunction( (0,-x) ) * v * dx

ap = BilinearForm(Q)
ap += 1/nu * grad(p) * grad(q)*dx \
    + 1/nu * (grad(p) * n * (qhat - q) + grad(q) * n * (phat - p)) * dS \
    + 1/nu * 4*order**2/h * (phat - p) * (qhat - q) * dS \
    + 1e-8 * p * q * dx
                        
b = BilinearForm(trialspace=V, testspace=Q)
b += div(u) * q * dx + u*n * qhat * dS

(u,uhat, sigma),(v,vhat, tau) = V2.TnT()

a2= BilinearForm(V2, condense = False, eliminate_hidden=False, store_inner = False)
a2 += -1/ nu * InnerProduct(sigma, tau) * dx + \
 (div(sigma) * v + div(tau) * u) * dx + \
 -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
 (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS \
 + nu* div(u) * div(v) * dx


f2 = LinearForm(V2)
f2 += CoefficientFunction( (1,1) ) * v * dx

# mapping of discontinuous to continuous H(div)
ind = V2.ndof * [0]
for el in mesh.Elements(VOL):
    dofs1 = V2.GetDofNrs(el)
    dofs2 = V.GetDofNrs(el)
    for d1,d2 in zip(dofs1,dofs2):
        ind[d1] = d2
    #print(dofs1)
    #print(dofs2)
    #print(ind)
    #input()
mapV = PermutationMatrix(V.ndof, ind)
print(mapV.height)
print(mapV.width)
#input()


class C2DM(BaseMatrix):
    def __init__(self, mesh, VC, VD):
        super(C2DM, self).__init__()
        self.M = mesh
        self.VC = VC
        self.VD = VD
    def Mult(self, x, y):
        for f in self.M.facets:

g = LinearForm(Q)




gfu = GridFunction(V)
gfuc = GridFunction(V2)
gfp = GridFunction(Q)

with TaskManager():
    bddc_a2 = Preconditioner(a2,type = 'bddc')
    a.Assemble()
    a2.Assemble()                           

    bddc_a = mapV.T @ bddc_a2 @ mapV
    
    bddc_ap = Preconditioner(ap,type = 'bddc')
    ap.Assemble()

    b.Assemble()
    f.Assemble()
    f2.Assemble()
    g.Assemble()
    stokes_mat = BlockMatrix([[a.mat,b.mat.T],
                              [b.mat,None]])

    pre = BlockMatrix([[bddc_a,None],
                       [None,bddc_ap]])

    gf = BlockVector([gfu.vec, gfp.vec])

    rhs= BlockVector([f.vec, g.vec])

    gf.data= pre * rhs
    gfuc.vec.data = bddc_a2 * f2.vec
    gf[0].data = mapV.T * gfuc.vec
    
    print(Norm(gf[0]))
    print(Norm(gf[1]))
    #astokes_inv = astokes.mat.Inverse(V.FreeDofs(True), inverse = "umfpack")
    
    #Astokes_inv = ((IdentityMatrix(astokes.mat.height) + astokes.harmonic_extension) @ (astokes_inv) @ (IdentityMatrix(astokes.mat.height) + astokes.harmonic_extension_trans)) + astokes.inner_solve

    #Astokes_mat = (IdentityMatrix(astokes.mat.height) - astokes.harmonic_extension_trans) @ (astokes.mat + astokes.inner_matrix) @ (IdentityMatrix(astokes.mat.height) - astokes.harmonic_extension)
    
    #solvers.MinRes(mat = stokes_mat, pre = pre, sol = gf, rhs = rhs)
    #gf.vec.data = Astokes_inv * f.vec

Draw (gfu.components[0], mesh, "velocity")
Draw (gfuc.components[0], mesh, "velocity2")
Draw (gfp.components[0], mesh, "p")
visoptions.scalfunction='velocity:0'
     
if True:
    lams = EigenValues_Preconditioner(mat=a.mat, pre= bddc_a , tol=1e-10)
    print("###############################")
    print("condition Ahat MCS", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")
