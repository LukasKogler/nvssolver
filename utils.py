import ngsolve as ngs
import netgen as ng
import netgen.meshing as ngm
import netgen.geom2d as g2d
import netgen.csg as csg

from FlowTemplates import FlowOptions

def gen_ref_mesh(geo, comm, maxh, nref = 0, mesh_file = "", load = False, save = False):
    if load:
        mesh = ngs.comp.Mesh(mesh_file, comm)
    else:
        if comm.rank == 0:
            ngmesh = geo.GenerateMesh(maxh=maxh)
            if save:
                ngmesh.Save(mesh_file)
            if comm.size > 1:
                ngmesh.Distribute(comm)
        else:
            ngmesh = ngm.Mesh.Receive(comm)
        for l in range(nref):
            ngmesh.Refine()
        mesh = ngs.comp.Mesh(ngmesh)
    return mesh

__case_setups = dict()

def register(func, name):
#    __case_setups [name] = lambda **stuff : func(**stuff)
    __case_setups [name] = func

def geo_2dchannel(H, L, obstacle=True):
    geo = g2d.SplineGeometry()
    geo.AddRectangle((0, 0), (L,H), bcs=("bottom", "right", "top", "left"))
    if obstacle == True:
        pos = 0.2/0.41*H
        if L < 1.3 * pos:
            raise "2dchannel: Increase L!"
        geo.AddCircle((pos, pos), r=pos/4, leftdomain=0, rightdomain=1, bc="obstacle")
    return geo

def geo_3dchannel(H, L, W, obstacle=True):
    geo = csg.CSGeometry()
    channel = csg.OrthoBrick( csg.Pnt(-1, 0, 0), csg.Pnt(L+1, H, W) ).bc("wall")
    inlet = csg.Plane (csg.Pnt(0,0,0), csg.Vec(-1,0,0)).bc("left")
    outlet = csg.Plane (csg.Pnt(L, 0,0), csg.Vec(1,0,0)).bc("right")
    if obstacle == True:
        pos = (L/5, 0.21 * (H/0.41))
        r = 0.05 * (H/0.41)
        if pos[1]+r > L:
            raise "2dchannel: Increase L!"
        cyl = csg.Cylinder(csg.Pnt(*pos, 0), csg.Pnt(*pos, 1), r).bc("obstacle")
    fluiddom = channel*inlet*outlet-cyl
    geo.Add(fluiddom)
    return geo


##
## 2D Schaefer Turek benchmark
##
def ST_2d(maxh, nref=0, save=False, load=False, nu=1, symmetric=False):
    H, L = 0.41, 2
    geo = geo_2dchannel(H=H, L=L, obstacle=True)
    mesh = gen_ref_mesh(geo, ngs.mpi_world, maxh=maxh, nref=nref, save=save, load=load)
    uin = ngs.CoefficientFunction( (4 * (2/H)**2 * ngs.y * (H - ngs.y), 0))
    flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "left", outlet = "right", wall_slip = "",
                                wall_noslip = "bottom|top|obstacle", uin = uin, symmetric = symmetric, vol_force = None)
    return flow_settings

register(ST_2d, "ST_2d")

##
## 2D Schaefer Turek benchmark
##
def ST_3d(maxh, nref=0, save=False, load=False, nu=1, symmetric=False):
    H, W, L = 0.41, 0.41, 2
    geo = geo_3dchannel(H=H, W=W, L=L, obstacle=True)
    mesh = gen_ref_mesh(geo, ngs.mpi_world, maxh=maxh, nref=nref, save=save, load=load)
    uin = ngs.CoefficientFunction( (4 * (2/H)**2 * ngs.y * (H - ngs.y), 0, 0))
    flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "left", outlet = "right", wall_slip = "",
                                wall_noslip = "wall|obstacle", uin = uin, symmetric = symmetric, vol_force = None)
    return flow_settings

register(ST_3d, "ST_3d")

##
## 2d channel:
## [0, L] \times [0, 1]
## inflow left, outflow right, wall top/bottom
##
def channel2d(maxh, L=1, nref=0, save=False, load=False, nu=1, symmetric=False):
    geo = geo_2dchannel(H=1, L=L, obstacle=False)
    mesh = gen_ref_mesh(geo, ngs.mpi_world, maxh=maxh, nref=nref, save=save, load=load)
    uin = ngs.CoefficientFunction( (4 * ngs.y * (1-ngs.y), 0) )
    flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "left", outlet = "right", wall_slip = "",
                                wall_noslip = "top|bottom|obstacle", uin = uin, symmetric = symmetric, vol_force = None)
    return flow_settings

##
## 2d vortex
##
##
##
def vortex2d(maxh, nref=0, save=False, load=False, nu=1):
    geo = geo_2dchannel(H=1, L=1, obstacle=False)
    mesh = gen_ref_mesh(geo, ngs.mpi_world, maxh=maxh, nref=nref, save=save, load=load)
    x, y, z = ngs.x, ngs.y, ngs.z
    zeta = x**2*(1-x)**2*y**2*(1-y)**2
    u_ex = ngs.CoefficientFunction((zeta.Diff(y),-zeta.Diff(x)))
    p_ex = x**5+y**5-1/3
    f_1 = -nu * (u_ex[0].Diff(x).Diff(x) + u_ex[0].Diff(y).Diff(y)) + p_ex.Diff(x)
    f_2 = -nu * (u_ex[1].Diff(x).Diff(x) + u_ex[1].Diff(y).Diff(y)) + p_ex.Diff(y)
    vol_force = ngs.CoefficientFunction((f_1,f_2))
    uin = ngs.CoefficientFunction( (0, 0) )
    flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "", outlet = "", wall_slip = "",
                                wall_noslip = ".*", uin = uin, symmetric = False, vol_force = vol_force)
    return flow_settings

def MakeSettings(case, **kwargs):
    if case in __case_setups:
        return __case_setups[case](**kwargs)
    else:
        raise "Unknown case!!"