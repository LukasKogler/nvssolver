from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions

import sys
sys.path.append('../')

#from NavierStokesSIMPLE_iterative_vertex import *
#from NavierStokesSIMPLE_iterative import *
from newiterative import *

geo = SplineGeometry()
H = 1
geo.AddRectangle((0, 0), (1,1), bcs=("wall", "outlet", "wall", "inlet"))
#geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

ngsglobals.msg_level = 0
mesh = Mesh(geo.GenerateMesh(maxh=0.1))

VD = VectorH1(mesh, order = 1)
gfd = GridFunction(VD)
alpha = 4
gfd.Set(CoefficientFunction((alpha * x,0))) 
#mesh.Curve(3)
mesh.SetDeformation(gfd)

ngsglobals.msg_level = 2

SetHeapSize(100 * 1000 * 100)
timestep = 1e-3

order = 5

it = []

cores = 4
SetNumThreads(cores)
for i in range(1):
 with TaskManager(): #pajetrace = 1000*1000*1000):
    hodivfree = False 
    GS = False
    bddc= False
    
    navstokes = NavierStokes(mesh, nu=0.001, order=order, timestep=timestep,
                             inflow="inlet", outflow="outlet", wall="cyl|wall",
                             uin=CoefficientFunction((1.5 * 4 * y * (H - y)/(H*H), 0)),
                             hodivfree = hodivfree,
                             sym = False
                             )

    #solver = "BPCG"
    solver = "MinRes"
    #solver = "GMRes"
    
    lit, lt_prep, lt_its, ndof = navstokes.SolveInitial(iterative=True, GS = GS,use_bddc = bddc, solver = solver)
    print("###############################")
    print("lt_prep = ", lt_prep)
    print("lt_its = ", lt_its)
    print("sum_time = ", lt_its + lt_prep)
    print("ndof = ", ndof)
    
    dpcs_prep = ndof/(lt_prep * cores)
    dpcs_its = ndof/(lt_its * cores)
    dpcs = ndof/((lt_its+lt_prep) * cores)
    print("k * dpcs_prep = ", dpcs_prep/1000)
    print("k * dpcs_its = ", dpcs_its/1000)
    print("k * dpcs = ", dpcs/1000)
    print("Norm(vel)", Integrate(navstokes.velocity*navstokes.velocity, mesh, VOL))
    print("###############################")
    it.append(lit)
    
    
    #navstokes.InitializeMatrices()

Draw(navstokes.velocity,mesh, "velocity")
Draw(navstokes.pressure,mesh, "pressure")
visoptions.scalfunction='velocity:0'
print("order = ", order)
print("iterations ", it)
#tend = 10
#t = 0
#input("A")
#with TaskManager():
#    while t < tend:
#        print (t)
#        navstokes.DoTimeStep()
#        t = t+timestep
#        Redraw()
#

