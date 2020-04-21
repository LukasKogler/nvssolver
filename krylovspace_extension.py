import ngsolve as ngs
import ngsolve.la as la
from math import sqrt
from typing import Optional, Callable


class IterativeSolver(ngs.BaseMatrix):
    def __init__(self, sysmat, tol : float = 1e-6, maxsteps : int = 3e2, printrates : bool = True, callback : Optional[Callable[[int, float], None]] = None,
                 rel_err : bool = True):
        super().__init__()
        self.sysmat = sysmat
        self.tol = tol
        self.maxsteps = maxsteps
        self.printrates = printrates
        self.callback = callback
        self.rel_err = rel_err
        self.iterations = 0
        self.err0 = -1
        self.errors = []
    def CheckError(self, it, err):
        self.iterations = it + 1
        if it == -1:
            self.err0 = err
            if self.printrates:
                print("Initial err = ", err)
        else:
            if self.printrates:
                print("it = ", it, " err = ", err)
            if self.callback is not None:
                self.callback(it, err)
            self.errors.append(err)
        return err < self.tol * (self.err0 if self.rel_err else 1)
    def Height(self) -> int:
        return self.A.width
    def Width(self) -> int:
        return self.A.width
    def IsComplex(self) -> bool:
        return self.A.IsComplex()
    def CreateColVector(self):
        return self.sysmat.CreateColVector()
    def CreateRowVector(self):
        return self.sysmat.CreateRowVector()
    def Mult(self, x : ngs.BaseVector, y : ngs.BaseVector) -> None:
        self.Solve(rhs=x, sol=y, initialize=True)
    def Solve(self, rhs : ngs.BaseVector, sol : ngs.BaseVector, initialize : bool = True):
        raise Exception("Solve not overloaded!!")


class BPCGSolver(IterativeSolver):
    def __init__(self, A : ngs.BaseMatrix, B : ngs.BaseMatrix, BT : ngs.BaseMatrix = None, C : ngs.BaseMatrix = None,
                 Ahat : ngs.BaseMatrix = None, Shat : ngs.BaseMatrix = None,
                 tol : float = 1e-6, maxsteps : int = 3e2, printrates : bool = True, rel_err : bool = True,
                 callback : Optional[Callable[[int, float], None]] = None):
        super().__init__(sysmat=ngs.BlockMatrix([[A, BT], [B, None]]), tol=tol, maxsteps=maxsteps, printrates=printrates, rel_err=rel_err, callback=callback)
        self.__setup(A, B, BT, C, Ahat, Shat)
        
    def __init__(self, M : ngs.BlockMatrix, Mhat : ngs.BlockMatrix, tol : float, maxsteps : int = 3e2, printrates : bool = True, rel_err : bool = True,
                 callback : Optional[Callable[[int, float], None]] = None):
        super().__init__(sysmat=M, tol=tol, maxsteps=maxsteps, printrates=printrates, rel_err=rel_err, callback=callback)
        self.__setup(A = M[0, 0], B = M[1, 0], BT = M[0, 1], C = M[1, 1], Ahat = Mhat[0, 0], Shat = Mhat[1, 1])

    def __setup(self, A : ngs.BaseMatrix, B : ngs.BaseMatrix, BT : ngs.BaseMatrix = None, C : ngs.BaseMatrix = None,
                Ahat : ngs.BaseMatrix = None, Shat : ngs.BaseMatrix = None):
        self.A = A
        self.B = B
        self.BT = BT if BT is not None else self.B.T
        self.C = C
        self.Ahat = Ahat
        self.Shat = Shat

        self._tmp_blk_vecs = [ ngs.BlockVector([self.A.CreateColVector(), self.B.CreateColVector()]) for k in range(7) ]
        self._tmp_vecs0 = [ self.A.CreateColVector() for k in range(7) ]
        self._tmp_vecs1 = [ self.B.CreateColVector() for k in range(3) ]
        self._tmp_vecs = self._tmp_blk_vecs + self._tmp_vecs0 + self._tmp_vecs1

        self.timer_prep = ngs.Timer("BPCG-Preparation")
        self.timer_its = ngs.Timer("BPCG-Iterations")
        self.timer_prepev = ngs.Timer("BPCG-Preparation-EV")


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
        
    def ScaleAhat(self, tol=1e-10, scal=None):
        if scal is None:
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

        self.CheckError(-1, err0)

        # self.errors.append(err0)
        # if self.printrates:
        #     print("it = ", 0, " err = ", err0, " " * 20)
        # if self.callback is not None:
        #     self.callback(0, err0)

        self.timer_prep.Stop()
        self.timer_its.Start()   

        for it in range(self.maxsteps):
            if it == 0:
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
            if self.CheckError(it, err):
                self.timer_its.Stop()   
                return

        self.timer_its.Stop()
        self.iterations = -int(abs(self.iterations))
        if ngs.mpi_world.rank==0:
            print("Warning: BPCG did not converge to TOL")


class GMResSolver(IterativeSolver):
    def __init__(self, M : ngs.BaseMatrix, Mhat : ngs.BaseMatrix, restart : Optional[int] = None,
                 innerproduct : Optional[Callable[[ngs.BaseVector, ngs.BaseVector], float]] = None,
                 tol : float = 1e-6, maxsteps : int = 3e2, printrates : bool = True, rel_err : bool = True,
                 callback : Optional[Callable[[int, float], None]] = None):
        super().__init__(sysmat=M, tol=tol, maxsteps=maxsteps, printrates=printrates, rel_err=rel_err, callback=callback)
        self.M = M
        self.Mhat = Mhat
        self.restart = restart
        self.tmp = self.M.CreateColVector()
        self.r = self.M.CreateColVector()
        # self.is_complex = self.M.is_complex # BlockMatrix has no is_complex
        self.is_complex = self.tmp.is_complex
        if innerproduct is None:
            self.innerproduct = lambda x, y : y.InnerProduct(x, conjugate=self.is_complex)
            self.norm = ngs.Norm
        else:
            self.innerproduct = innerproduct
            self.norm = lambda x : sqrt(innerproduct(x,x).real)
        self.restarts = 0

    def arnoldi(self, A, Ahat, Q, m, k):
        q = A.CreateColVector()
        self.tmp.data = A * Q[k]
        q.data = Ahat * self.tmp
        h = ngs.Vector(m+1, self.is_complex)
        h[:] = 0
        for i in range(k+1):
            h[i] = self.innerproduct(Q[i],q)
            q.data += (-1)* h[i] * Q[i]
        h[k+1] = self.norm(q)
        if abs(h[k+1]) < 1e-12:
            return h, None
        q *= 1./h[k+1].real
        return h, q

    def givens_rotation(self, v1, v2):
        if v2 == 0:
            return 1, 0
        elif v1 == 0:
            return 0, v2/abs(v2)
        else:
            t = sqrt((v1.conjugate()*v1+v2.conjugate()*v2).real)
            cs = abs(v1)/t
            sn = v1/abs(v1) * v2.conjugate()/t
            return cs,sn

    def apply_givens_rotation(self, h, cs, sn, k):
        for i in range(k):
            temp = cs[i] * h[i] + sn[i] * h[i+1]
            h[i+1] = -sn[i].conjugate() * h[i] + cs[i].conjugate() * h[i+1]
            h[i] = temp
        cs[k], sn[k] = self.givens_rotation(h[k], h[k+1])
        h[k] = cs[k] * h[k] + sn[k] * h[k+1]
        h[k+1] = 0

    def calcSolution(self, k, H, Q, beta, x):
        mat = ngs.Matrix(k+1,k+1, self.is_complex)
        for i in range(k+1):
            mat[:,i] = H[i][:k+1]
        rs = ngs.Vector(k+1, self.is_complex)
        rs[:] = beta[:k+1]
        y = mat.I * rs
        for i in range(k+1):
            x.data += y[i] * Q[i]

    def Solve_impl(self, rhs, sol, curr_it, initialize):
        if sol is None:
            sol = self.M.CreateColVector()
            sol[:] = 0
        if initialize:
            sol[:] = 0
        if curr_it == 0:
            self.restarts = 0
        else:
            self.restarts += 1

        m = self.maxsteps - curr_it
        sn, cs = ngs.Vector(m, self.is_complex), ngs.Vector(m, self.is_complex)
        sn[:] = 0
        cs[:] = 0

        self.tmp.data = rhs - self.M * sol
        self.r.data = self.Mhat * self.tmp

        Q, H = [], []
        
        Q.append(self.M.CreateColVector())
        r_norm = self.norm(self.r)

        if curr_it == 0:
            self.CheckError(-1, r_norm)

        Q[0].data = 1./r_norm * self.r

        beta = ngs.Vector(m+1, self.is_complex)
        beta[:] = 0
        beta[0] = r_norm
       
        for k in range(m):
            h,q = self.arnoldi(self.M, self.Mhat, Q, m, k)
            H.append(h)
            if q is None:
                break
            Q.append(q)
            self.apply_givens_rotation(h, cs, sn, k)
            beta[k+1] = -sn[k].conjugate() * beta[k]
            beta[k] = cs[k] * beta[k]
            error = abs(beta[k+1])
            if self.CheckError(curr_it, error):
                self.calcSolution(k, H, Q, beta, sol)
                return
            curr_it += 1
            if self.restart and k+1 == self.restart and not (self.restart == self.maxsteps):
                self.calcSolution(k, H, Q, beta, sol)
                del Q
                self.Solve_impl(sol=sol, rhs=rhs, curr_it=curr_it, initialize=False)
                return

        self.iterations = -int(abs(self.iterations))
        if ngs.mpi_world.rank==0:
            print("Warning: GMRes did not converge to TOL")

    def Solve(self, rhs : ngs.BaseVector, sol : ngs.BaseVector, initialize : bool = True):
        self.Solve_impl(rhs=rhs, sol=sol, curr_it=0, initialize=initialize)


class MinResSolver(IterativeSolver):
    def __init__(self, M : ngs.BaseMatrix, Mhat : ngs.BaseMatrix,
                 tol : float = 1e-6, maxsteps : int = 3e2, printrates : bool = True, rel_err : bool = True,
                 callback : Optional[Callable[[int, float], None]] = None):
        super().__init__(sysmat=M, tol=tol, maxsteps=maxsteps, printrates=printrates, rel_err=rel_err, callback=callback)

        self.M = M
        self.Mhat = Mhat

        self._tmp_vecs = [ self.M.CreateColVector() for k in range(9) ]

    def Solve(self, rhs : ngs.BaseVector, sol : ngs.BaseVector, initialize : bool = True):

        u = sol if sol is not None else self.M.CreateColVector()

        v, v_new, v_old, w, w_new, w_old, z, z_new, mz = self._tmp_vecs

        if initialize:
            u[:] = 0
            v.data = rhs
        else:
            v.data = rhs - self.M * u

        z.data = self.Mhat * v if self.Mhat is not None else v

        gamma = sqrt(ngs.InnerProduct(z, v))
        gamma_new = 0
        z.data = 1/gamma * z
        v.data = 1/gamma * v   

        ResNorm = gamma      
        ResNorm_old = gamma  

        self.CheckError(-1, ResNorm)

        eta_old = gamma
        c_old, c = 1, 1
        s, s_old, s_new = 0, 0, 0

        v_old[:] = 0.0
        w_old[:] = 0.0
        w[:] = 0.0

        for k in range(self.maxsteps):
            mz.data = self.M * z
            delta = ngs.InnerProduct(mz,z)
            v_new.data = mz - delta * v - gamma * v_old

            z_new.data = self.Mhat * v_new if self.Mhat is not None else v_new

            # this can be -eps when we plug in a direct solver as preconditioner
            gamma_new_sq = ngs.InnerProduct(z_new, v_new)

            alpha0 = c * delta - c_old * s * gamma    
            alpha1 = sqrt(alpha0 * alpha0 + gamma_new_sq) #**
            alpha2 = s * delta + c_old * c * gamma
            alpha3 = s_old * gamma

            c_new = alpha0/alpha1
            # we are cheating if gamma_new is -eps, in which case we return enyways before it matters
            s_new = sqrt(abs(gamma_new_sq))/alpha1

            w_new.data = z - alpha3 * w_old - alpha2 * w
            w_new.data = 1/alpha1 * w_new   

            u.data += c_new * eta_old * w_new
            eta = -s_new * eta_old

            ResNorm = abs(s_new) * ResNorm_old

            if self.CheckError(k, ResNorm):
                return

            # ok, now sqrt should be safe
            gamma_new = sqrt(gamma_new_sq)
            z_new *= 1/gamma_new
            v_new *= 1/gamma_new

            v_old, v, v_new = v, v_new, v_old
            w_old, w, w_new = w, w_new, w_old
            z, z_new = z_new, z
            eta_old = eta
            s_old, s = s, s_new
            c_old, c = c, c_new
            gamma = gamma_new
            ResNorm_old = ResNorm

        self.iterations = -int(abs(self.iterations))
        if ngs.mpi_world.rank==0:
            print("Warning: MinRes did not converge to TOL")
