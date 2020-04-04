import ngsolve as ngs
import ngsolve.la as la
from math import sqrt
from typing import Optional, Callable



class BPCGSolver(ngs.BaseMatrix):
    def __init__(self, A : ngs.BaseMatrix, B : ngs.BaseMatrix, BT : ngs.BaseMatrix = None, C : ngs.BaseMatrix = None,
                 Ahat : ngs.BaseMatrix = None, Shat : ngs.BaseMatrix = None,
                 tol : float = 1e-6, maxsteps : int = 3e2, printrates : bool = True, rel_err : bool = True,
                 callback : Optional[Callable[[int, float], None]] = None):
        super().__init__()
        self.__setup(A, B, BT, C, Ahat, Shat, tol, maxsteps, printrates, rel_err, callback)
        
    def __init__(self, M : ngs.BlockMatrix, Mhat : ngs.BlockMatrix, tol : float, maxsteps : int = 3e2, printrates : bool = True, rel_err : bool = True,
                 callback : Optional[Callable[[int, float], None]] = None):
        self.__setup(A = M[0, 0], B = M[1, 0], BT = M[0, 1], C = M[1, 1], Ahat = Mhat[0, 0], Shat = Mhat[1, 1], tol = tol,
                     maxsteps = maxsteps, printrates = printrates, rel_err = rel_err, callback = callback)

    def __setup(self, A : ngs.BaseMatrix, B : ngs.BaseMatrix, BT : ngs.BaseMatrix = None, C : ngs.BaseMatrix = None,
                Ahat : ngs.BaseMatrix = None, Shat : ngs.BaseMatrix = None,
                tol : float = 1e-6, maxsteps : int = 3e2, printrates : bool = True, rel_err : bool = True,
                callback : Optional[Callable[[int, float], None]] = None):
        super().__init__()
        self.A = A
        self.B = B
        self.BT = BT if BT is not None else self.B.T
        self.C = C
        self.Ahat = Ahat
        self.Shat = Shat
        self.tol = tol
        self.maxsteps = maxsteps
        self.printrates = printrates
        self.callback = callback
        self.rel_err = rel_err

        self._tmp_blk_vecs = [ ngs.BlockVector([self.A.CreateColVector(), self.B.CreateColVector()]) for k in range(7) ]
        self._tmp_vecs0 = [ self.A.CreateColVector() for k in range(7) ]
        self._tmp_vecs1 = [ self.B.CreateColVector() for k in range(3) ]
        self._tmp_vecs = self._tmp_blk_vecs + self._tmp_vecs0 + self._tmp_vecs1

        self.errors = []
        self.iterations = 0
        self.timer_prep = ngs.Timer("BPCG-Preparation")
        self.timer_its = ngs.Timer("BPCG-Iterations")
        self.timer_prepev = ngs.Timer("BPCG-Preparation-EV")

        
    def Height(self) -> int:
        return self.A.width
    def Width(self) -> int:
        return self.A.width
    def IsComplex(self) -> bool:
        return self.A.IsComplex()
    def Mult(self, x : ngs.BaseVector, y : ngs.BaseVector) -> None:
        self.Solve(rhs=x, sol=y, initialize=True)

    def CalcScaleFactor(self, A, Ahat, tol=1e-10):
        self.timer_prepev.Start()
        lams = list(ngs.la.EigenValues_Preconditioner(mat=A, pre=Ahat, tol=tol))
        # scal = 1.0 / (lams[0] * 0.95)
        scal = 1.0 / (lams[-1] * 1.05)
        if ngs.mpi_world.rank == 0:
            print("###############################")
            print("scale = ", scal)
            print("condition = ", lams[-1] / lams[0])
            print("max(lams) = ", lams[-1])
            print("min(lams) = ", lams[0])
            print("###############################")
        self.timer_prepev.Stop()
        return scal
        
    def ScaleAhat(self, tol=1e-10):
        scal = self.CalcScaleFactor(self.A, self.Ahat, tol)
        self.Ahat = scal * self.Ahat

    def Solve(self, rhs : ngs.BaseVector, sol : ngs.BaseVector, initialize : bool = True):

        # lams = list(ngs.la.EigenValues_Preconditioner(mat=self.A, pre=self.Ahat, tol=1e-10))
        # print("###############################")
        # print("BEFORE SOL")
        # print("condition = ", lams[-1] / lams[0])
        # print("max(lams) = ", lams[-1])
        # print("min(lams) = ", lams[0])
        # print("###############################")

        self.timer_prep.Start()        
    
        u = sol if sol is not None else rhs.CreateVector()
        if initialize:
            u[:] = 0.0

        orig_rhs = rhs
        f, g = orig_rhs[0], orig_rhs[1]
        
        d, w, v, z, z_old, s, pr = self._tmp_blk_vecs
        tmp0, f_new, tmp1, tmp2, tmp4, matA_s0, matB_s1 = self._tmp_vecs0
        g_new, tmp3, tmp5 = self._tmp_vecs1

        for somevec in self._tmp_vecs:
            somevec[:] = 0

        tmp0.data = self.Ahat * f
        f_new.data = self.A * tmp0 - f
        g_new.data = self.B * tmp0 - g

        rhs = ngs.BlockVector([f_new, g_new])

        tmp0.data = self.A * u[0] + self.BT * u[1]
        tmp1.data = self.Ahat * tmp0
        tmp2.data = self.A * tmp1
        tmp4.data = tmp1 - u[0]
        tmp3.data = self.B * tmp4
        d[0].data = rhs[0] - (tmp2 - tmp0)
        d[1].data = rhs[1] - tmp3
        pr[0].data = self.Ahat * f
        tmp5.data = self.B * pr[0] - g
        pr[1].data = self.Shat * tmp5
        w[0].data = pr[0] - tmp1
        w[1].data = pr[1] - self.Shat * tmp3
        wdn = ngs.InnerProduct(w, d)
        s.data = w
        err0 = sqrt(abs(wdn))
        
        if self.printrates:
            print("it = ", 0, " err = ", err0, " " * 20)
        if self.callback is not None:
            self.callback(0, err0)


        self.timer_prep.Stop()
        self.timer_its.Start()   

        for it in range(1, 1+self.maxsteps):
            if it == 1:
                matA_s0.data = self.A * s[0]
                z[0].data = matA_s0  # A*w_0_0
            else:
                matA_s0.data = beta * matA_s0 + z_old[0] - alpha * tmp2
            matB_s1.data = self.BT * s[1]
            tmp0.data = matA_s0 + matB_s1
            tmp1.data = self.Ahat * tmp0
            tmp2.data = self.A * tmp1
            tmp4.data = tmp1 - s[0]
            tmp3.data = self.B * tmp4
            z_old[0].data = z[0]
            v[0].data = tmp2 - tmp0
            v[1].data = tmp3
            wd = wdn
            as_s = ngs.InnerProduct(s, v)
            alpha = wd / as_s
            u.data += alpha * s
            d.data += (-alpha) * v
            w[0].data = w[0] + (-alpha) * tmp1
            w[1].data = w[1] + (-alpha) * self.Shat * tmp3
            wdn = ngs.InnerProduct(w, d)
            beta = wdn / wd
            z[0].data -= alpha * tmp2
            s *= beta
            s.data += w
            err = sqrt(abs(wd))
            self.errors.append(err)
            if self.printrates:
                print("it = ", it, " err = ", err, " " * 20)
            if self.callback is not None:
                self.callback(it, err)
            if err < self.tol * (err0 if self.rel_err else 1):
                self.timer_its.Stop()   
                self.iterations = it
                break
        else:
            self.iterations = -1
            if ngs.mpi_world.rank==0:
                print("Warning: BPCG did not converge to TOL")

