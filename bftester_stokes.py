from ngsolve import *


class Parameters():
    def __init__(self, comm, pc):
        self.pc = pc
        self.comm = comm
        self.max_l = pc.GetNLevels(1 if comm.size>1 else 0)-1
        self.max_d = [pc.GetNDof(k,0) for k in range(self.max_l+1)]
        self.lev,self.dof,self.add = 0,0,0
        self.scal = 1.0 if comm.rank==0 else 0
        self.os = 1
        self.doquit = False

    def Update(self):
        if self.comm.rank == 0:
            self.ParseInput()
        if self.comm.size > 1:
            raise("Parameters Update scatter todo..")

    def ParseInput(self):
        ret = input('what to do next? (Q to quit)\n')

        print("parse: ", ret)

        if ret in ['q','Q']:
            self.doquit = True
            return

        if len(ret) == 0:
            self.dof = self.dof + self.os
            return

        if ret[0]=='a':
            ret = ret[1:]
            self.add = 1
        else:
            self.add = 0;

        if len(ret) == 0:
            return

        if ret[0] in ['l', 'L']:
            self.lev = int(ret[1:])
            self.dof = 0
        elif ret[0] == 'o':
            self.os = int(ret[1:])
            print('set OS to', os)
        else:
            intlist = [int(x) for x in ret.split(',')]
            lil = len(intlist)
            if lil >= 1:
                self.dof = intlist[0]
            if lil >= 2:
                self.dof = intlist[1]
            if lil >= 3:
                self.scal = intlist[2]

        
        
class Console():
    def __init__(self, comm, pc):
        self.params = Parameters(comm, pc)

    def Update(self):
        self.params.Update()
        
        
def shape_test(stokes, aux_pc):
    mesh = stokes.settings.mesh
    dim = mesh.dim
    comm = mesh.comm
    X = stokes.disc.X
    Q = stokes.disc.Q
    Q0 = L2(mesh, order=0)

    (un, ut, sigma), (vn, vt, tau) = X.TnT()
    L = VectorL2(mesh, order=2)
    ul, vl = L.TnT()
    n = specialcf.normal(mesh.dim)
    normal = lambda cf : (cf*n)*n
    tang = lambda cf : cf - (cf*n)*n
    m1 = BilinearForm(trialspace=X, testspace=L)
    m1 += normal(un) * normal(vl) * dx(element_vb=BND)
    m1 += tang(ut) * tang(vl) * dx(element_vb=BND)
    m1.Assemble()
    m2 = BilinearForm(L)
    m2 += ul * vl * dx(element_vb=BND)
    m2.Assemble()
    m2i = m2.mat.Inverse()
    E = m2i@m1.mat

    print("X", X.ndof)
    print("X0", X.components[0].ndof)
    print("X1", X.components[1].ndof)
    print("L", L.ndof)


    gfu = GridFunction(X)
    # sc_vn = Draw(gfu.components[0], mesh, name = "U_n")
    # sc_vt = Draw(gfu.components[1], mesh, name = "U_t")
    # sc_v  = Draw(CoefficientFunction(gfu.components[0] + gfu.components[1]), mesh, name = "U")

    gfl = GridFunction(L)
    sc_v = Draw(gfl, mesh, name = "U")
    
    B = stokes.la.B

    gfp = GridFunction(Q)
    gfp.Set(x)
    gfp0 = GridFunction(Q0)

    sc_p = Draw(gfp, mesh, name = "p")
    # sc_p0 = Draw(gfp, mesh, name = "p0")

    console = Console(comm, aux_pc)
    params = console.params

    bf_vec = gfu.vec.CreateVector()

    while not params.doquit:
        console.Update()

        print("next: ", params.lev, params.dof, params.scal)

        aux_pc.GetBF(bf_vec, params.lev, 0, params.dof)

        if params.add:
            gfu.vec.data += params.scal * bf_vec
        else:
            gfu.vec.data = params.scal * bf_vec
            
        gfl.vec.data = E * gfu.vec    
        gfp.vec.data = B * gfu.vec
        gfp0.Set(gfp)

        print("gfu vec", [ x for x in enumerate(gfu.vec) if x[1]!=0.0])
        print("gfp0 vec", [ x for x in enumerate(gfp0.vec) if x[1]!=0.0])

        Redraw()
