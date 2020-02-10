from ngsolve import *
import netgen.geom2d as g2d
import netgen.csg as csg

from FlowTemplates import *


geo = g2d.unit_square
mesh = Mesh(geo.GenerateMesh(maxh=0.3))

inflow = "left"
outflow = "right"
wall_slip = ""
wall_noslip = "top|bottom"
uin = CoefficientFunction((y * (1-y), 0))
nu = 1

disc_opts = { "order" : 2,
              "hodivfree" : False,
              "truecompile" : True }

sol_opts = { "pc_ver" : "direct",
             "elint" : False,
             "block_la" : False,
             "pc_opts" : { } }


with TaskManager():
    if True:
        stokes = StokesTemplate(disc_opts = disc_opts,
                                flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inflow = inflow, outflow = outflow, wall_slip = wall_slip,
                                                            wall_noslip = wall_noslip, uin = uin, symmetric = False),
                                sol_opts = sol_opts )
    else:
        stokes = StokesTemplate(disc_opts = disc_opts,
                                flow_opts = { "geom" : geo, "mesh" : mesh, "nu" : nu, "inflow" : inflow, "outflow" : outflow,
                                              "wall_slip" : wall_slip, "wall_noslip" : wall_noslip, "uin" : uin },
                                sol_opts = sol_opts )


