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

Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
Sigma = Compress(Sigma)

QT = L2(mesh, order = order, lowest_order_wb = False)
Qhat = FacetFESpace(mesh, order = order) ###???????????

V = FESpace([VT,Vhat, Sigma])
#V.SetCouplingType(V.Range(2), COUPLING_TYPE.HIDDEN_DOF)
#V = Compress(V)

Qc = FESpace([QT,Qhat])
Q = Qhat

#V = FESpace([VT,Vhat, Sigma, QT, Qhat])
   
(u,uhat, sigma),(v,vhat, tau) = V.TnT()

#(p,phat),(q,qhat) = Q.TnT()
phat,qhat = Q.TnT()

dS = dx(element_boundary=True)
        
n = specialcf.normal(2)
h = specialcf.mesh_size

def tang(v):
    return v - (v*n)*n

eps = 10

a= BilinearForm(V, condense = False, eliminate_hidden=True)
a += -1/ nu * InnerProduct(sigma, tau) * dx + \
 (div(sigma) * v + div(tau) * u) * dx + \
 -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
 (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS \
 + nu*1e6* div(u) * div(v) * dx 
#+ nu * eps * u * v * dx

ahat= BilinearForm(V, condense = True, eliminate_hidden=True)
ahat += -1/ nu * InnerProduct(sigma, tau) * dx + \
 (div(sigma) * v + div(tau) * u) * dx + \
 -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
 (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS \
 + nu*1e6* div(u) * div(v) * dx \
 + nu * eps * u * v * dx

f = LinearForm(V)
f += CoefficientFunction( (0,-x) ) * v * dx

b = BilinearForm(trialspace=V, testspace=Q)
b += u*n * qhat * dS


(p,phat),(q,qhat) = Qc.TnT()

ap = BilinearForm(Qc, condense = True)
ap += 1/(nu*eps)  * grad(p) * grad(q)*dx \
    + 1/(nu*eps) * (grad(p) * n * (qhat - q) + grad(q) * n * (phat - p)) * dS \
    + 1/(nu*eps) * 4*order**2/h * (phat - p) * (qhat - q) * dS \
    + 1e-2/(nu*eps) * p * q * dx



embQhat = Embedding(Qc.ndof,Qc.Range(1))

#ap += 1/(nu * eps) * phat * qhat *dS 
 
g = LinearForm(Q)

gfu = GridFunction(V)

gfp = GridFunction(Q)

with TaskManager():
    bddc_a = Preconditioner(ahat,type = 'bddc')
    a.Assemble()
    ahat.Assemble()
    #bddc_a = ahat.mat.Inverse(V.FreeDofs(), inverse = "umfpack")
    
    #bddc_ap = Preconditioner(ap,type = 'bddc')
    #bddc_ap =Preconditioner(ap,type = 'direct')

    ap_inv = Preconditioner(ap, type = "bddc")
    ap.Assemble()
    #ap_inv = ap.mat.Inverse(Qc.FreeDofs(), inverse= "sparsecholesky")
    #ap_inv_big = ((IdentityMatrix(ap.mat.height) + ap.harmonic_extension) @ (ap_inv) @ (IdentityMatrix(ap.mat.height) + ap.harmonic_extension_trans)) + ap.inner_solve
    
    bddc_ap = embQhat.T @ ap_inv @ embQhat
    
    b.Assemble()

    f.Assemble()
    g.Assemble()
    
    stokes_mat = BlockMatrix([[a.mat,b.mat.T],
                              [b.mat,None]])

    bddc_a_big = ((IdentityMatrix(ahat.mat.height) + ahat.harmonic_extension) @ (bddc_a) @ (IdentityMatrix(ahat.mat.height) + ahat.harmonic_extension_trans)) + ahat.inner_solve
    
    pre = BlockMatrix([[bddc_a_big,None],
                       [None,bddc_ap]])

    gf = BlockVector([gfu.vec, gfp.vec])

    rhs= BlockVector([f.vec, g.vec])
        
    solvers.MinRes(mat = stokes_mat, pre = pre, sol = gf, rhs = rhs, maxsteps = 1000) #,tol=1e-10)


Draw (gfu.components[0], mesh, "velocity")
visoptions.scalfunction='velocity:0'
     
if True:
    
    lams = EigenValues_Preconditioner(mat=ahat.mat, pre= bddc_a , tol=1e-10)
    print("###############################")
    print("condition Ahat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")
    
    ahat_inv = ahat.mat.Inverse(V.FreeDofs(True), inverse = "sparsecholesky")
    
    ahat_inv_big = ((IdentityMatrix(ahat.mat.height) + ahat.harmonic_extension) @ (ahat_inv) @ (IdentityMatrix(ahat.mat.height) + ahat.harmonic_extension_trans)) + ahat.inner_solve
    
    S =  b.mat @ ahat_inv_big @ b.mat.T

    lams = EigenValues_Preconditioner(mat=S, pre= bddc_ap , tol=1e-10)
    print("###############################")
    print("condition Shat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    #print("lams", lams)
    print("###############################")

    lams = EigenValues_Preconditioner(mat=ap.mat, pre= ap_inv , tol=1e-10)
    print("###############################")
    print("condition Shat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")
