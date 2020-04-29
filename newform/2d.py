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
Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet="bottom|right|top|left")
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)

#Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
#Sigma = Compress(Sigma)

QT = L2(mesh, order = order-1, lowest_order_wb = False)
Qhat = FacetFESpace(mesh, order = order) ###???????????

V = FESpace([VT,Vhat, Sigma])
Q = FESpace([QT, Qhat])
#V = FESpace([VT,Vhat, Sigma, QT, Qhat])
   
(u,uhat, sigma),(v,vhat, tau) = V.TnT()

(p,phat),(q,qhat) = Q.TnT()

dS = dx(element_boundary=True)
        
n = specialcf.normal(2)
h = specialcf.mesh_size

def tang(v):
    return v - (v*n)*n

eps = 1e-6

a= BilinearForm(V, condense = False, eliminate_hidden=False, store_inner = False)
a += -1/ nu * InnerProduct(sigma, tau) * dx + \
 (div(sigma) * v + div(tau) * u) * dx + \
 -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
 (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS \
 + nu* div(u) * div(v) * dx \
 + nu * eps * u * v * dx

f = LinearForm(V)
f += CoefficientFunction( (0,-x) ) * v * dx

ap = BilinearForm(Q)
ap += 1/(nu * eps)  * grad(p) * grad(q)*dx \
    + 1/(nu * eps) * (grad(p) * n * (qhat - q) + grad(q) * n * (phat - p)) * dS \
    + 1/(nu * eps) * 4*order**2/h * (phat - p) * (qhat - q) * dS \
    + 1e-8/(nu * eps) * p * q * dx
                        
b = BilinearForm(trialspace=V, testspace=Q)
b += div(u) * q * dx + u*n * qhat * dS
            
 
g = LinearForm(Q)

gfu = GridFunction(V)

gfp = GridFunction(Q)

with TaskManager():
    bddc_a = Preconditioner(a,type = 'bddc')
    a.Assemble()
    bddc_a = a.mat.Inverse(V.FreeDofs(), inverse = "umfpack")
    
    bddc_ap = Preconditioner(ap,type = 'bddc')
    ap.Assemble()

    b.Assemble()

    f.Assemble()
    g.Assemble()
    
    stokes_mat = BlockMatrix([[a.mat,b.mat.T],
                              [b.mat,None]])

    pre = BlockMatrix([[bddc_a,None],
                       [None,bddc_ap]])

    gf = BlockVector([gfu.vec, gfp.vec])

    rhs= BlockVector([f.vec, g.vec])

    #astokes_inv = astokes.mat.Inverse(V.FreeDofs(True), inverse = "umfpack")
    
    #Astokes_inv = ((IdentityMatrix(astokes.mat.height) + astokes.harmonic_extension) @ (astokes_inv) @ (IdentityMatrix(astokes.mat.height) + astokes.harmonic_extension_trans)) + astokes.inner_solve

    #Astokes_mat = (IdentityMatrix(astokes.mat.height) - astokes.harmonic_extension_trans) @ (astokes.mat + astokes.inner_matrix) @ (IdentityMatrix(astokes.mat.height) - astokes.harmonic_extension)
    
    solvers.MinRes(mat = stokes_mat, pre = pre, sol = gf, rhs = rhs, maxsteps = 1000)
    #gf.vec.data = Astokes_inv * f.vec

Draw (gfu.components[0], mesh, "velocity")
Draw (gfp.components[0], mesh, "p")
visoptions.scalfunction='velocity:0'
     
if True:
    lams = EigenValues_Preconditioner(mat=a.mat, pre= bddc_a , tol=1e-10)
    print("###############################")
    print("condition Ahat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")
