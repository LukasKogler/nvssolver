from ngsolve import *
from NavierStokesSIMPLE_iterative import *
import netgen.gui
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions

geo = SplineGeometry()
geo.AddRectangle((0, 0), (20, 0.41), bcs=("wall", "outlet", "wall", "inlet"))
#geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

mesh = Mesh(geo.GenerateMesh(maxh=0.05))
mesh.Curve(3)
ngsglobals.msg_level = 10

SetHeapSize(100 * 1000 * 1000)
timestep = 1e-3
with TaskManager():
    navstokes = NavierStokes(mesh, nu=0.001, order=2, timestep=timestep,
                             inflow="inlet", outflow="outlet", wall="cyl|wall",
                             uin=CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41 * 0.41), 0))
                             )

    navstokes.SolveInitial(iterative=True)

Draw(navstokes.velocity,mesh, "velocity")
Draw(navstokes.pressure,mesh, "pressure")
visoptions.scalfunction='velocity:0'

input("end")
