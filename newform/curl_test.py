from ngsolve import *
from ngsolve.internal import visoptions
from ngsolve.internal import viewoptions
from netgen.geom2d import *
from ngsolve.la import EigenValues_Preconditioner


mesh = Mesh(unit_square.GenerateMesh(maxh = 0.1))
Draw(mesh)


Vcurl = HCurl(mesh, order = 2, dirichlet = ".*")

u,v = Vcurl.TnT()

acurl = BilinearForm(Vcurl)
acurl += (curl(u) * curl(v) + 1e-8 * u * v ) * dx

acurl.Assemble()


f = LinearForm(Vcurl)
f += CoefficientFunction( (y,-x) ) * v * dx


gf = GridFunction(Vcurl)


a_inv = acurl.mat.Inverse(Vcurl.FreeDofs(), inverse = "sparsecholesky")
gf.vec.data = a_inv * f.vec

Draw (gf, mesh, "velocity")
#Draw (gf.components[3], mesh, "p")
#visoptions.scalfunction='velocity:0'
