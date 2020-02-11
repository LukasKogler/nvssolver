from ngsolve import *
from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions
from netgen.geom2d import unit_square
from ngsolve.la import EigenValues_Preconditioner
import sys 
mesh = Mesh(unit_square.GenerateMesh(maxh = 0.1))
Draw(mesh)

nu = 1
order = 2
        
VT = HDiv(mesh, order=order, discontinuous = True)
Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet="bottom|right|top|left")
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, discontinuous=True)

S = L2(mesh, order=order - 1)

S.SetCouplingType(IntRange(0, S.ndof), COUPLING_TYPE.HIDDEN_DOF)
S = Compress(S)

Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
Sigma = Compress(Sigma)

QT = L2(mesh, order = order-1) 
Qhat = FacetFESpace(mesh, order = order)#, dirichlet="bottom|right|top|left") ###???????????

V = FESpace([VT,Vhat, Sigma,S])
#V.SetCouplingType(V.Range(2), COUPLING_TYPE.HIDDEN_DOF)
#V = Compress(V)

Qsigma = HDiv(mesh, order = order, discontinuous = True)
N = NumberSpace(mesh)
Qc = FESpace([Qsigma,QT,Qhat])
Q = Qhat

#V = FESpace([VT,Vhat, Sigma, QT, Qhat])
   
(u,uhat, sigma, W),(v,vhat, tau, R) = V.TnT()

def Skew2Vec(m):
    return m[1, 0] - m[0, 1]

#(p,phat),(q,qhat) = Q.TnT()
phat,qhat = Q.TnT()

dS = dx(element_boundary=True)
        
n = specialcf.normal(2)
h = specialcf.mesh_size

def tang(v):
    return v - (v*n)*n

eps = 0.01
divdiv = 1e6

a= BilinearForm(V, condense = False, eliminate_hidden=True)
a += -1/ nu * InnerProduct(sigma, tau) * dx + \
 (div(sigma) * v + div(tau) * u) * dx + \
 (InnerProduct(W, Skew2Vec(tau)) + InnerProduct(R, Skew2Vec(sigma))) * dx \
 -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
 (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS \
 + nu*divdiv* div(u) * div(v) * dx 
#+ nu * eps * u * v * dx

ahat= BilinearForm(V, condense = True, eliminate_hidden=True)
ahat += -1/ nu * InnerProduct(sigma, tau) * dx + \
 (div(sigma) * v + div(tau) * u) * dx + \
 (InnerProduct(W, Skew2Vec(tau)) + InnerProduct(R, Skew2Vec(sigma))) * dx \
 -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
 (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS \
 + nu*divdiv* div(u) * div(v) * dx \
 + nu * eps * u * v * dx

f = LinearForm(V)
f += CoefficientFunction( (0,-x) ) * v * dx

h = specialcf.mesh_size

b = BilinearForm(trialspace=V, testspace=Q)
b += u*n * qhat * dS

#(sigmap,p,phat, lam),(tauq,q,qhat, mu) = Qc.TnT()
(sigmap,p,phat),(tauq,q,qhat) = Qc.TnT()

ap = BilinearForm(Qc, condense = True)
ap +=-(nu*eps)  * sigmap * tauq * dx \
    +(div(sigmap) * q + div(tauq) * p) * dx \
    + (sigmap * n * qhat + tauq * n * phat) * dS \
    + 1e-2/(nu*eps) * p * q * dx

embQhat = Embedding(Qc.ndof,Qc.Range(2))

h = specialcf.mesh_size

Qc2 = FESpace([Qhat, N])

(phat,lam), (qhat, mu) = Qc2.TnT()

ap2 = BilinearForm(Qc2, condense = False) 
ap2 += h * phat * qhat * dS \
    + (mu * phat + lam * qhat) * dx(element_boundary = True) \
    + (mu * phat.Trace() + lam * qhat.Trace()) * ds



bt_phat = BitArray(Qc2.ndof)
bt_phat.Clear()

for e in mesh.edges:
    dofs = Qhat.GetDofNrs(e)
    for i in range(1,len(dofs)):
        bt_phat.Set(dofs[i])
    #print(dofs)
    #input()

blocks = []

for v in mesh.vertices:
    el_block = []    
    for e in v.edges:        
        el_block += Qc2.GetDofNrs(e)
        #print(Qc2.GetDofNrs(e))
        
    el_block.append(Qhat.ndof)    # this is the meanvalue dof number
    blocks.append ( el_block)

'''
for v in mesh.Elements():
    el_block = []    
    for e in v.edges:        
        el_block += Qc2.GetDofNrs(e)
        #print(Qc2.GetDofNrs(e))
        
    el_block.append(Qhat.ndof)    # this is the meanvalue dof number
    blocks.append ( el_block)
'''

embQhat2 = Embedding(Qc2.ndof,Qc2.Range(0))
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
    
    ap2.Assemble()
    #ap2_inv = ap2.mat.Inverse(bt_phat, inverse = "umfpack")
    ap2_inv = ap2.mat.CreateBlockSmoother(blocks)
    #ap_inv = ap.mat.Inverse(Qc.FreeDofs(True), inverse= "umfpack")
    
    #ap_inv_big = ((IdentityMatrix(ap.mat.height) + ap.harmonic_extension) @ (ap_inv) @ (IdentityMatrix(ap.mat.height) + ap.harmonic_extension_trans)) + ap.inner_solve
    
    
    #bddc_ap = embQhat.T @ ap_inv @ embQhat
    #bddc_ap = embQhat.T @ (ap_inv) @ embQhat + embQhat2.T @ (ap2_inv) @ embQhat2
    bddc_ap = embQhat.T @ (ap_inv) @ embQhat + embQhat2.T @ (ap2_inv) @ embQhat2
    
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
Draw (1e6 * div(gfu.components[0]), mesh, "p")
visoptions.scalfunction='velocity:0'
     
if True:
    
    lams = EigenValues_Preconditioner(mat=ahat.mat, pre= bddc_a , tol=1e-10)
    print("###############################")
    print("condition Ahat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")
    
    #ahat_inv = ahat.mat.Inverse(V.FreeDofs(True), inverse = "sparsecholesky")
    ahat_inv_big = ((IdentityMatrix(ahat.mat.height) + ahat.harmonic_extension) @ (bddc_a) @ (IdentityMatrix(ahat.mat.height) + ahat.harmonic_extension_trans)) + ahat.inner_solve

    sys.stdout.flush()
    sys.stderr.flush()
    
    S =  b.mat @ ahat_inv_big @ b.mat.T
    
    #ap_inv_big = embQhat.T @ (((IdentityMatrix(ap.mat.height) + ap.harmonic_extension) @ (ap_inv) @ (IdentityMatrix(ap.mat.height) + ap.harmonic_extension_trans)) + ap.inner_solve) @ embQhat
    
    lams = EigenValues_Preconditioner(mat=S, pre= bddc_ap , tol=1e-10)
    #lams = EigenValues_Preconditioner(mat=S, pre= ap_inv_big , tol=1e-10)
    print("###############################")
    print("condition Shat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("lams", lams)
    print("###############################")

    sys.stdout.flush()
    sys.stderr.flush()

    
    lams = EigenValues_Preconditioner(mat=ap.mat, pre= ap_inv , tol=1e-10)
    print("###############################")
    print("condition ap", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")
