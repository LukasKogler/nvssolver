from ngsolve import *
from NavierStokesSIMPLE_iterative import *
import netgen.gui
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions

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

SetHeapSize(100*1000*1000)
timestep = 0.002

with TaskManager():
  navstokes = NavierStokes (mesh, nu=0.001, order=3, timestep = timestep,
                              inflow="inlet", outflow="outlet", wall="wall|cyl",
                              uin=CoefficientFunction( (16*y*(0.41-y)*z*(0.41-z)/(0.41*0.41*0.41*0.41), 0, 0) ))
                              

navstokes.SolveInitial(iterative = True)

Draw (navstokes.pressure, mesh, "pressure")
Draw (navstokes.velocity, mesh, "velocity")
visoptions.scalfunction='velocity:0'

