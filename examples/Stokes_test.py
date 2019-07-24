from ngsolve import *
import netgen.gui
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions

import sys
sys.path.append('../')

from NavierStokesSIMPLE_iterative import *

geo = SplineGeometry()
geo.AddRectangle((0, 0), (1, 1), bcs=("wall", "outlet", "wall", "inlet"))
#geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")

mesh = Mesh(geo.GenerateMesh(maxh=0.05))
mesh.Curve(3)
ngsglobals.msg_level = 0

SetHeapSize(100 * 1000 * 1000)
timestep = 1e-3

nu = 0.001

zeta = x**2*(1-x)**2*y**2*(1-y)**2
u_ex = CoefficientFunction((zeta.Diff(y),-zeta.Diff(x)))
Draw(u_ex, mesh, "u_ex")
p_ex = x**5+y**5-1/3

f_1 = -nu * (u_ex[0].Diff(x).Diff(x) + u_ex[0].Diff(y).Diff(y)) + p_ex.Diff(x)
f_2 = -nu * (u_ex[1].Diff(x).Diff(x) + u_ex[1].Diff(y).Diff(y)) + p_ex.Diff(y)
force = CoefficientFunction((f_1,f_2))

with TaskManager():
    navstokes = NavierStokes(mesh, nu=0.001, order=2, timestep=timestep,
                             inflow="", outflow="", wall="cyl|wall|inlet|outlet",
                             uin=CoefficientFunction((0, 0))
                             )
    navstokes.AddForce(force)
    navstokes.SolveInitial(iterative=True)

Draw(navstokes.velocity,mesh, "velocity")
Draw(navstokes.pressure,mesh, "pressure")
visoptions.scalfunction='velocity:0'

input("end")
