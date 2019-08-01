from ngsolve import *

#import netgen.gui
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions

import sys
sys.path.append('../')
from NavierStokesSIMPLE_iterative import *

ngsglobals.msg_level = 0
from netgen.csg import *
geo = CSGeometry()
channel = OrthoBrick( Pnt(-1, 0, 0), Pnt(3, 0.41, 0.41) ).bc("wall")
inlet = Plane (Pnt(0,0,0), Vec(-1,0,0)).bc("inlet")
outlet = Plane (Pnt(2.5, 0,0), Vec(1,0,0)).bc("outlet")
cyl = Cylinder(Pnt(0.5, 0.2,0), Pnt(0.5,0.2,0.41), 0.05).bc("wall")
fluiddom = channel*inlet*outlet-cyl
geo.Add(fluiddom)
mesh = Mesh( geo.GenerateMesh(maxh=0.1))
mesh.Curve(3)
Draw(mesh)

ngsglobals.msg_level = 0

SetHeapSize(100*1000*1000)
timestep = 0.002
cores = 24
SetNumThreads(cores)

with TaskManager():
  navstokes = NavierStokes (mesh, nu=0.001, order=3, timestep = timestep,
                              inflow="inlet", outflow="outlet", wall="wall|cyl",
                              uin=CoefficientFunction( (16*y*(0.41-y)*z*(0.41-z)/(0.41*0.41*0.41*0.41), 0, 0) ))
                              
  lit, lt_prep, lt_its, ndof = navstokes.SolveInitial(iterative=True)
  print("###############################")
  print("lt_prep = ", lt_prep)
  print("lt_its = ", lt_its)
  print("sum_time = ", lt_its + lt_prep)
  print("ndof = ", ndof)
  
  dpcs_prep = ndof/(lt_prep * cores)
  dpcs_its = ndof/(lt_its * cores)
  dpcs = ndof/((lt_its+lt_prep) * cores)
  print("dpcs_prep = ", dpcs_prep/100)
  print("dpcs_its = ", dpcs_its/100)
  print("dpcs = ", dpcs/100)
  print("Norm(vel)", Integrate(navstokes.velocity*navstokes.velocity, mesh, VOL))
  print("###############################")
  #it.append(lit)
  

Draw (navstokes.pressure, mesh, "pressure")
Draw (navstokes.velocity, mesh, "velocity")
visoptions.scalfunction='velocity:0'

