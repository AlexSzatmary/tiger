#!/usr/bin/env python

import numpy as np
import scipy
import scipy.integrate

# Dirichlet:
# y0[0] = a
# dydt[0] = 0
# von Neumann BC:
# y0 = ...
# dydt[0] = D * d2[m]/dx2 = D * (2 * m[1] +
#                                (-2 + 2 * dx * a / b) * m[0]
#                                -2 * dx * c / b)
# lbc(m)

# lbc(m)

class Dirichlet(object):
    def __init__(self, u_0):
        self.u_0 = u_0

class Robin(object):
    """Boundary condition for a u + b u' = c"""
    def __init__(self, a=None, b=None, c=None, side=None):
        self.a = a
        self.b = b
        self.c = c
        if side in ['left', 'right']:
            self.side = side
        else:
            raise ValueError("side must be 'left' or 'right'")

    def d2udx2(self, u, dx):
        if self.side == 'left':
            return (2 * u[1] +
                    (-2 + 2 * dx * self.a / self.b) * u[0]
                    - 2 * dx * self.c / self.b)
        elif self.side == 'right':
            return (2 * u[-2] +
                    (-2 - 2 * dx * self.a / self.b) * u[-1]
                    + 2 * dx * self.c / self.b)

class VonNeumann(Robin):
    def __init__(self, dudx=None, side=None):
        super(VonNeumann, self).__init__(a=0, b=1, c=dudx, side=side)

class Model1s(object):
    def __init__(self, N=16, dx=None, m0=None, lhs=None, rhs=None, D=None,
                 dt=None, Nt=None):
        self.N = N
        self.dx = dx
        self.m0 = m0
        self.lhs = lhs
        self.rhs = rhs
        self.D = D
        self.dt = dt
        self.Nt = Nt

        self.t = np.arange(0, self.Nt) * self.dt
        if type(self.lhs) is Dirichlet:
            self.m0[0] = self.lhs.u_0
        if type(self.rhs) is Dirichlet:
            self.m0[-1] = self.rhs.u_0

    def odeint_wrapper(self):
        self.m = scipy.integrate.odeint(
            self.dmdt, self.m0, self.t,
            args=(self.dx, self.D, self.lhs, self.rhs))
        
    @staticmethod
    def dmdt(m, t, dx, D, lhs, rhs):
        r = np.zeros(np.size(m))

        if type(lhs) is Dirichlet:
            r[0] = 0.
        else:
            r[0] = D * lhs.d2udx2(m, dx)

        if type(rhs) is Dirichlet:
            r[-1] = 0.
        else:
            r[-1] = D * rhs.d2udx2(m, dx)

        r[1:-1] = (D * (m[0:-2] - 2. * m[1:-1] + m[2:]) / dx ** 2)
        return r

m1s = Model1s(N=16, dx=1./(16-1), m0=np.zeros(16),
              lhs=Dirichlet(1.), rhs=VonNeumann(0.), D=1.,
              dt=0.01, Nt=100)

def main(argv=None):
    if argv is None:
        argv = sys.argv

if __name__ == "__main__":
    sys.exit(main())
