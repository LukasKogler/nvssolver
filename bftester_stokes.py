from ngsolve import *


class Parameters():
    def __init__(self, comm, pc):
        self.pc = pc
        self.comm = comm
        self.max_l = pc.GetNLevels(1 if comm.size>1 else 0)-1
        self.max_d = [pc.GetNDof(k,0) for k in range(self.max_l+1)]
        self.lev,self.dof,self.add = 0,0,0
        self.scal = 1.0 if comm.rank==0 else 0
        self.os = 0
        self.doquit = False

    def Update(self):
        if self.comm.rank == 0:
            self.ParseInput()
        if self.comm.size > 1:
            raise("Parameters Update scatter todo..")

    def ParseInput(self):
        ret = input('what to do next? (Q to quit)\n')
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

    gfu = GridFunction(X)
    gfu.components[0].Set(CoefficientFunction( (1, 0) ))

    sc_vn = Draw(gfu.components[0], mesh, name = "U_n")
    # sc_vt = Draw(gfu.components[1], mesh, name = "U_t")
    # sc_v  = Draw(CoefficientFunction(gfu.components[0] + gfu.components[1]), mesh, name = "U")

    B = stokes.la.B

    gfp = GridFunction(Q)
    gfp.Set(x)

    sc_p = Draw(gfp, mesh, name = "p")

    console = Console(comm, aux_pc)
    params = console.params

    bf_vec = gfu.vec.CreateVector()

    while not params.doquit:
        console.Update()

        aux_pc.GetBF(bf_vec, params.lev, 0, params.dof)

        if params.add:
            gfu.vec.data += params.scal * bf_vec
        else:
            gfu.vec.data = params.scal * bf_vec
            
        print("gfu vec", gfu.vec)
            
        gfp.vec.data = B * gfu.vec
        Redraw()
