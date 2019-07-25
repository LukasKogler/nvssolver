from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions

import sys
sys.path.append('../')

SetNumThreads(24)
from NavierStokesSIMPLE_iterative import *

geo = SplineGeometry()
geo.AddRectangle((0, 0), (2, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

mesh = Mesh(geo.GenerateMesh(maxh=0.01))
mesh.Curve(3)
ngsglobals.msg_level = 0

SetHeapSize(100 * 1000 * 1000)
timestep = 1e-3

order = 8

it = []


for i in range(1):
 with TaskManager(pajetrace = 500*1000*1000):
    #with TaskManager():
    navstokes = NavierStokes(mesh, nu=0.001, order=order, timestep=timestep,
                             inflow="inlet", outflow="outlet", wall="cyl|wall",
                             uin=CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
                             )

    lit, lt_prep, lt_its, ndof = navstokes.SolveInitial(iterative=True)
    print("###############################")
    print("lt_prep = ", lt_prep)
    print("lt_its = ", lt_its)
    print("sum_time = ", lt_its + lt_prep)
    print("ndof = ", ndof)
    
    dpcs_prep = ndof/(lt_prep * 24)
    dpcs_its = ndof/(lt_its * 24)
    dpcs = ndof/((lt_its+lt_prep) * 24)
    print("dpcs_prep = ", dpcs_prep/100)
    print("dpcs_its = ", dpcs_its/100)
    print("dpcs = ", dpcs/100)
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

