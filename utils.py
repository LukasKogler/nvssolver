import sys, os
import ngsolve as ngs
import netgen as ng
import netgen.meshing as ngm
import netgen.geom2d as g2d
import netgen.csg as csg

from FlowTemplates import FlowOptions


def MatIter (amat, n_vecs = 5, lam_max = 1, lam_min = 0, reverse = True, M = 1e5, startvec = None, tol=1e-8, freedofs = None, proj=None,
             innerproduct = None):
    I = ngs.IdentityMatrix(amat.height) if proj is None else proj
    A = I - 1/(lam_max-lam_min) * amat if reverse else amat
    if freedofs is not None and proj is None:
        proj = ngs.Projector(freedofs, True)
        A = proj @ A @ proj
    if innerproduct == None:
        innerproduct = lambda x, y : ngs.InnerProduct(x, y)
    evecs = list()
    evals = list()
    if startvec is None:
        startvec = amat.CreateColVector()
        import random
        for k in range(len(startvec)):
            startvec.local_vec[k] = random.randint(1,1000) / 1000
        startvec.Distribute()
        startvec.Cumulate()
    # print("svec", startvec)
    sys.stdout.flush()
    ngs.mpi_world.Barrier()
    startvec *= innerproduct(startvec, startvec)**(-0.5)
    tempvec = startvec.CreateVector()
    errvec = startvec.CreateVector()
    def ortho(vec, base):
        for k,bv in enumerate(base):
            ip = innerproduct(vec, bv)
            vec -= ip * bv
    for K in range(n_vecs):
        for l in range(int(M)):
            tempvec.data = A * startvec
            ortho(tempvec, evecs)
            ip = innerproduct(tempvec, startvec)
            errvec.data = tempvec - ip * startvec
            # print('\rit = ', l, ', norms = ', ngs.Norm(startvec), ngs.Norm(tempvec), ngs.Norm(errvec))
            startvec.data = 1/innerproduct(tempvec, tempvec)**0.5 * tempvec
            err = ngs.Norm(errvec)
            if err < tol * ip:
                break
            if reverse:
                if ngs.mpi_world.rank == 0:
                    print('\revec', K, 'after', l, 'its with err', err, 'and eval', ip, 'orig eval', (lam_max - lam_min) * ( 1 - ip), end="")
                sys.stdout.flush()
            else:
                if ngs.mpi_world.rank == 0:
                    print('\revec', K, 'after', l, 'its with err', err, 'and eval', ip, end="")
                sys.stdout.flush()
        vk = startvec.CreateVector()
        vk.data = startvec
        avk = amat.CreateColVector()
        lamk = ( lam_max - lam_min ) * ( 1 - ip ) if reverse else ip
        avk.data = amat * vk - lamk * vk
        navk = innerproduct(avk, avk)
        nvk  = innerproduct(vk, vk)
        print("\nevec", K, ", actual eval = ", lamk, ", final error =", navk, ", rel final error = ", navk/nvk)
        evecs.append(vk)
        evals.append(ip)
    return evecs, evals


def gen_ref_mesh(geo, comm, maxh, nref = 0, mesh_file = "", load = False, save = False):
    msl = ngs.ngsglobals.msg_level
    ngs.ngsglobals.msg_level = 0
    if comm.rank == 0:
        if load:
            ngmesh = ngm.Mesh()
            ngmesh.Load(mesh_file)
        else:
            ngmesh = geo.GenerateMesh(maxh=maxh)
            if save:
                ngmesh.Save(mesh_file)
        # ngmesh.EnableTable("edges", False)
        # ngmesh.EnableTable("faces", False)
        if comm.size > 1:
            ngmesh.Distribute(comm)
    else:
        ngmesh = ngm.Mesh.Receive(comm)
        # ngmesh.EnableTable("edges", False)
        # ngmesh.EnableTable("faces", False)
        ngmesh.SetGeometry(geo)
    for l in range(nref):
        ngmesh.Refine()
    mesh = ngs.comp.Mesh(ngmesh)
    ngs.ngsglobals.msg_level = msl
    return mesh

__case_setups = dict()

def register(func, name):
#    __case_setups [name] = lambda **stuff : func(**stuff)
    __case_setups [name] = func

def geo_2dchannel(H, L, obstacle=True):
    geo = g2d.SplineGeometry()
    geo.AddRectangle((0, 0), (L,H), bcs=("bottom", "right", "top", "left"))
    if obstacle == True:
        pos = 0.2
        r = 0.05
        geo.AddCircle((pos, pos), r=r, leftdomain=0, rightdomain=1, bc="obstacle")
    return geo

def geo_3dchannel(H, L, W, obstacle=True):
    geo = csg.CSGeometry()
    channel = csg.OrthoBrick( csg.Pnt(-1, 0, 0), csg.Pnt(L+1, H, W) ).bc("wall")
    inlet = csg.Plane (csg.Pnt(0,0,0), csg.Vec(-1,0,0)).bc("left")
    outlet = csg.Plane (csg.Pnt(L, 0,0), csg.Vec(1,0,0)).bc("right")
    if obstacle == True:
        pos = (0.5, 0.2)
        r = 0.05
        cyl = csg.Cylinder(csg.Pnt(pos[0], pos[1], 0), csg.Pnt(pos[0], pos[1], 1), r).bc("obstacle")
        fluiddom = channel*inlet*outlet-cyl
    else:
        fluiddom = channel*inlet*outlet
    geo.Add(fluiddom)
    return geo


##
## 2D Schaefer Turek benchmark
##
def ST_2d(maxh, nref=0, save=False, load=False, nu=1, symmetric=False, obstacle=True, mesh_file=None):
    H, L = 0.41, 2.2
    geo = geo_2dchannel(H=H, L=L, obstacle=obstacle)
    mesh = gen_ref_mesh(geo, ngs.mpi_world, maxh=maxh, nref=nref, save=save, load=load, mesh_file=mesh_file)
    uin = ngs.CoefficientFunction( (1.5 * (2/H)**2 * ngs.y * (H - ngs.y), 0))
    flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "left", outlet = "right", wall_slip = "",
                                wall_noslip = "bottom|top|obstacle", uin = uin, symmetric = symmetric, vol_force = None)
    return flow_settings

register(ST_2d, "ST_2d")

##
## 2D Schaefer Turek benchmark
##
def ST_3d(maxh, nref=0, save=False, load=False, nu=1, symmetric=False, obstacle=True, mesh_file=None, L=2.5):
    H, W, L = 0.41, 0.41, L
    geo = geo_3dchannel(H=H, W=W, L=L, obstacle=obstacle)
    mesh = gen_ref_mesh(geo, ngs.mpi_world, maxh=maxh, nref=nref, save=save, load=load, mesh_file=mesh_file)
    uin = ngs.CoefficientFunction( (2.25 * (2/H)**2 * (2/W)**2 * ngs.y * (H - ngs.y) * ngs.z * (W - ngs.z), 0, 0))
    flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "left", outlet = "right", wall_slip = "",
                                wall_noslip = "wall|obstacle", uin = uin, symmetric = symmetric, vol_force = None)
    return flow_settings

register(ST_3d, "ST_3d")

##
## 2d channel:
## [0, L] \times [0, 1]
## inflow left, outflow right, wall top/bottom
##
def channel2d(maxh, H=1, L=1, nref=0, save=False, load=False, nu=1, symmetric=False):
    geo = geo_2dchannel(H=H, L=L, obstacle=False)
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
    u_ex = ngs.CF((zeta.Diff(y),-zeta.Diff(x)))
    p_ex = x**5+y**5-1/3
    f_1 = -nu * (u_ex[0].Diff(x).Diff(x) + u_ex[0].Diff(y).Diff(y)) + p_ex.Diff(x)
    f_2 = -nu * (u_ex[1].Diff(x).Diff(x) + u_ex[1].Diff(y).Diff(y)) + p_ex.Diff(y)
    vol_force = ngs.CF((f_1,f_2))
    uin = ngs.CF( (0, 0) )
    flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "", outlet = "", wall_slip = "",
                                wall_noslip = ".*", uin = uin, symmetric = False, vol_force = vol_force)
    return flow_settings

def vortex3d(maxh, nref=0, save=False, load=False, nu=1):
    x, y, z = ngs.x, ngs.y, ngs.z
    geo = ngs.netgen.csg.unit_cube
    mesh = gen_ref_mesh(geo, ngs.mpi_world, maxh=maxh, nref=nref, save=save, load=load)
    nu = 1e-3
    zeta = x**2*(1-x)**2*y**2*(1-y)**2*z*z*(1-z)*(1-z)
    u_ex = ngs.CF((zeta.Diff(y)-zeta.Diff(z),-zeta.Diff(x)+zeta.Diff(z), zeta.Diff(x) - zeta.Diff(y)))
    p_ex = x**5+y**5+z**5-1/2
    u_ex_deriv = ngs.CF((u_ex[0].Diff(x),u_ex[0].Diff(y),u_ex[0].Diff(z)))
    v_ex_deriv = ngs.CF((u_ex[1].Diff(x),u_ex[1].Diff(y),u_ex[1].Diff(z)))
    w_ex_deriv = ngs.CF((u_ex[2].Diff(x),u_ex[2].Diff(y),u_ex[2].Diff(z)))
    velexderiv = ngs.CF(((u_ex_deriv, v_ex_deriv, w_ex_deriv ), ) , dims=(1,9) )
    vel_ex_deriv = ngs.CF( (velexderiv), dims = (3,3) )
    f_1 = -nu * (vel_ex_deriv[0,0].Diff(x) + vel_ex_deriv[0,1].Diff(y) + vel_ex_deriv[0,2].Diff(z)) + p_ex.Diff(x)
    f_2 = -nu * (vel_ex_deriv[1,0].Diff(x) + vel_ex_deriv[1,1].Diff(y) + vel_ex_deriv[1,2].Diff(z)) + p_ex.Diff(y)
    f_3 = -nu * (vel_ex_deriv[2,0].Diff(x) + vel_ex_deriv[2,1].Diff(y) + vel_ex_deriv[2,2].Diff(z)) + p_ex.Diff(z)
    vol_force = ngs.CF((f_1,f_2,f_3))
    uin = ngs.CF( (0, 0, 0) )
    flow_settings = FlowOptions(geom = geo, mesh = mesh, nu = nu, inlet = "", outlet = "", wall_slip = "",
                                wall_noslip = ".*", uin = uin, symmetric = False, vol_force = vol_force)
    return flow_settings

def MakeSettings(case, **kwargs):
    if case in __case_setups:
        return __case_setups[case](**kwargs)
    else:
        raise "Unknown case!!"
