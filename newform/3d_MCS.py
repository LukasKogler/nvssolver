from ngsolve import *
from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions
from netgen.csg import unit_cube, Pnt, Vec, Plane, CSGeometry
from ngsolve.la import EigenValues_Preconditioner

'''
s = 1

unit_cube = CSGeometry()
p1 = Plane(Pnt(0,0,0),Vec(-1,0,0)).bc("back")
p3 = Plane(Pnt(0,0,0),Vec(0,-1,0)).bc("left")
p5 = Plane(Pnt(0,0,0),Vec(0,0,-1)).bc("bottom")

p2 = Plane(Pnt(s,s,s),Vec(1,0,0)).bc("front")
p4 = Plane(Pnt(s,s,s),Vec(0,1,0)).bc("right")
p6 = Plane(Pnt(s,s,s),Vec(0,0,1)).bc("top")

p7 = Plane(Pnt(s,0,0), Vec(1,1,1)).bc("diag")
p8 = Plane(Pnt(-s,0,0), Vec(-1,1,1)).bc("diag")

# unit_cube.Add (p1*p2*p3*p4*p5*p6*p7, col=(0,0,1))
unit_cube.Add (p3*p5*p7*p8, col=(0,0,1))
'''


mesh = Mesh(unit_cube.GenerateMesh(maxh = 0.4))
#mesh.Refine()
Draw(mesh)

#from netgen.meshing import MeshPoint, Element2D, Element3D, FaceDescriptor, Pnt
# ngmesh = netgen.meshing.Mesh()
# pids = [ ngmesh.Add(MeshPoint(Pnt(*t))) for t in [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]]
# ngmesh.Add(Element3D(1, pids))
# ngmesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
# ngmesh.Add(Element2D(vertices=[1,2,4], index=1))
# ngmesh.Add(Element2D(vertices=[1,2,3], index=1))
# ngmesh.Add(Element2D(vertices=[1,3,4], index=1))
# ngmesh.Add(Element2D(vertices=[2,3,4], index=1))
# ngmesh.SetMaterial(1, "inner")
# mesh = Mesh(ngmesh)

nu = 1
order = 2
        
V = HDiv(mesh, order=order, dirichlet="top|bottom|left|right|back")
Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet="top|bottom|left|right|back")
Sigma = HCurlDiv(mesh, order=order - 1, orderinner=order, ordertrace = order-1, discontinuous=True)

Sigma.SetCouplingType(IntRange(0, Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
Sigma = Compress(Sigma)
        
V_vel = FESpace([V,Vhat, Sigma])
   
(u,uhat, sigma),(v,vhat, tau) = V_vel.TnT()


dS = dx(element_boundary=True)
        
alpha_penalty = 20

n = specialcf.normal(3)
h_HDG = specialcf.mesh_size

def tang(v):
    return v - (v*n)*n

a = BilinearForm(V_vel, condense = True, eliminate_hidden=True) # , printelmat=True, elmatev=True)
a += -1/ nu * InnerProduct(sigma, tau) * dx + \
 (div(sigma) * v + div(tau) * u) * dx + \
 -(((sigma * n) * n) * (v * n) + ((tau * n) * n) * (u * n)) * dS + \
 (-(sigma * n) * tang(vhat) - (tau * n) * tang(uhat)) * dS

f = LinearForm(V_vel)
f += CoefficientFunction( (1,1,1) ) * v * dx

gf = GridFunction(V_vel)

with TaskManager():
    bddc_a = Preconditioner(a,type = 'bddc')

    a.Assemble()
    f.Assemble()    
    print ("solve bvp")
    # BVP(a, f, gf, pre=bddc_a)
    solvers.BVP(bf=a, lf=f, gf=gf, pre=bddc_a, print = False)
    print ("done")

Draw (gf.components[0], mesh, "gfu_mcs")
     
if True:
    lams = EigenValues_Preconditioner(mat=a.mat, pre= bddc_a , tol=1e-10)
    print("###############################")
    print("condition Ahat MCS", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")

V_hdg = HDiv(mesh, order=order, dirichlet="top|bottom|left|right|back")
Vhat_hdg = TangentialFacetFESpace(mesh, order=order , highest_order_dc=True, dirichlet="top|bottom|left|right|back")

V_vel_hdg = FESpace([V_hdg,Vhat_hdg])


(u,uhat),(v,vhat) = V_vel_hdg.TnT()

alpha_penalty = 11

a = BilinearForm(V_vel_hdg, condense = True)
a += nu*InnerProduct(Grad(u),Grad(v))*dx \
   - nu*(Grad(u)*n)*(tang(v-vhat))*dx(element_boundary=True) \
  - nu*(Grad(v)*n)*(tang(u-uhat))*dx(element_boundary=True) \
  + nu*alpha_penalty*(order)**2/h_HDG*tang(u-uhat)*tang(v-vhat)*dx(element_boundary=True)
  

f = LinearForm(V_vel_hdg)
f += CoefficientFunction( (1,1,1) ) * v * dx

gf = GridFunction(V_vel_hdg)

with TaskManager():
    bddc_a = Preconditioner(a,type = 'bddc')

    a.Assemble()
    f.Assemble()
    solvers.BVP(bf=a, lf=f, gf=gf, pre=bddc_a, print = False)

Draw (gf.components[0], mesh, "gfu_hdg")

if True:
    lams = EigenValues_Preconditioner(mat=a.mat, pre= bddc_a , tol=1e-10)
    print("###############################")
    print("condition Ahat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")

VscalT = L2(mesh, order=order)
Vscalhat = FacetFESpace(mesh, order=order, dirichlet="top|bottom|left|right|back")

Vscal = FESpace([VscalT,Vscalhat])

(u,uhat),(v,vhat) = Vscal.TnT()

a = BilinearForm(Vscal, condense = True) # , printelmat=True, elmatev=True)
a += nu*InnerProduct(grad(u),grad(v))*dx \
  - nu*(grad(u)*n)*(v-vhat)*dx(element_boundary=True) \
  - nu*(grad(v)*n)*(u-uhat)*dx(element_boundary=True) \
  + nu*alpha_penalty*(order)**2/h_HDG*(u-uhat)*(v-vhat)*dx(element_boundary=True)


f = LinearForm(Vscal)
f += 1 * v * dx

gf = GridFunction(Vscal)

with TaskManager():
    bddc_a = Preconditioner(a,type = 'bddc')
    a.Assemble()
    f.Assemble()

    print ("solve bvp")
    # BVP(a, f, gf, pre=bddc_a)
    solvers.BVP(bf=a, lf=f, gf=gf, pre=bddc_a, print = False)
    print ("done")

Draw (gf.components[0], mesh, "scal")

# for i in range (V_vel.ndof):
#    print ("ct[",i,"] = ", V_vel.couplingtype[i])
# print ("ct = ", V_vel.couplingtype)
    
if True:
    lams = EigenValues_Preconditioner(mat=a.mat, pre= bddc_a , tol=1e-10)

    print("###############################")
    print("condition Ahat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")


##################################################################################
VscalT_mixed = L2(mesh, order=order-1)
Vscalhat_mixed = FacetFESpace(mesh, order=order, dirichlet="top|bottom|left|right|back")
Sigmascal_mixed = HDiv(mesh, order, discontinuous = True)

Vscal_mixed = FESpace([VscalT_mixed,Vscalhat_mixed,Sigmascal_mixed])

(u,uhat, sigma),(v,vhat, tau) = Vscal_mixed.TnT()

a_mixed = BilinearForm(Vscal_mixed, condense = False) # , printelmat=True, elmatev=True)
a_mixed += -1/nu*sigma * tau*dx \
 + div(tau) * u *dx + div(sigma)*v * dx \
 +( sigma * n * vhat + tau * n * uhat) * dS


f_mixed = LinearForm(Vscal_mixed)
f_mixed += 1 * v * dx

gf_mixed = GridFunction(Vscal_mixed)

with TaskManager():
    #bddc_a_mixed = Preconditioner(a_mixed,type = 'bddc')
    a_mixed.Assemble()
    a_inv = a_mixed.mat.Inverse(Vscal_mixed.FreeDofs(), inverse = "umfpack")
    f_mixed.Assemble()
    gf_mixed.vec.data= a_inv * f_mixed.vec
    
    print ("solve bvp")
    # BVP(a, f, gf, pre=bddc_a)
    #solvers.BVP(bf=a_mixed, lf=f_mixed, gf=gf_mixed, pre=bddc_a_mixed, print = False)
    print ("done")

Draw (gf_mixed.components[0], mesh, "scal_mixed")

# for i in range (V_vel.ndof):
#    print ("ct[",i,"] = ", V_vel.couplingtype[i])
# print ("ct = ", V_vel.couplingtype)
    
if True:
    lams = EigenValues_Preconditioner(mat=a.mat, pre= bddc_a , tol=1e-10)

    print("###############################")
    print("condition Ahat", max(lams) / min(lams))
    print("max(lams) = ", max(lams))
    print("min(lams) = ", min(lams))
    print("###############################")





    



