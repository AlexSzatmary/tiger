#!/usr/bin/env python

import numpy as np
import scipy
import scipy.integrate
import numbers

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


# Boundary conditions
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

    def opd2udx2(self, u, dx):
        if self.side == 'left':
            return (2 * u[1] +
                    (-2 + 2 * dx * self.a / self.b) * u[0]
                    - 2 * dx * self.c / self.b) / dx ** 2
        elif self.side == 'right':
            return (2 * u[-2] +
                    (-2 - 2 * dx * self.a / self.b) * u[-1]
                    + 2 * dx * self.c / self.b) / dx ** 2


class VonNeumann(Robin):
    def __init__(self, dudx=None, side=None):
        super(VonNeumann, self).__init__(a=0, b=1, c=dudx, side=side)


def opd2udx2(u, dx):
    """Applies second-derivative finite difference operator on u

    Note that this takes some array, u, of length N, but returns second
    derivatives for only nodes 2 through N-1"""
    return (u[0:-2] - 2. * u[1:-1] + u[2:]) / dx ** 2


class PDESolver(object):
    """Time-varying reaction-diffusion type of solver
    Solves du/dt = f(u, du/dx, d2u/dx2)"""
    def __init__(self, N=16, dudt=None, L=None, u0=None, lhs=None, rhs=None,
                 dt=None, Nt=None, run=True):
        self.N = N
        self.dudt = dudt
        self.L = L
        self.dx = L / (N - 1)
        self.x = np.arange(0, N) * self.dx
        if issubclass(type(u0), numbers.Number):
            self.u0 = np.ones(self.N) * u0
        else:
            self.u0 = u0(self.x)
        self.lhs = lhs
        self.rhs = rhs
        self.dt = dt
        self.Nt = Nt

        self.t = np.arange(0, self.Nt) * self.dt
        if type(self.lhs) is Dirichlet:
            self.u0[0] = self.lhs.u_0
        if type(self.rhs) is Dirichlet:
            self.u0[-1] = self.rhs.u_0

        if run:
            self.run()

    def run(self):
        self.u = scipy.integrate.odeint(
            self.d_dudt, self.u0, self.t,
            args=(self,))

    @staticmethod
    def d_dudt(u, t, self):
        """"
        Discretized application of dudt

        Watch out! Because this is a staticmethod, as required by odeint, self
        is the third argument
        """
        r = np.zeros(np.size(u))

        if type(self.lhs) is Dirichlet:
            r[0] = 0.
        else:
            r[0] = self.dudt(u[0], self.lhs.opd2udx2(u, self.dx))

        if type(self.rhs) is Dirichlet:
            r[-1] = 0.
        else:
            r[-1] = self.dudt(u[-1], self.rhs.opd2udx2(u, self.dx))
#            r[-1] = self.dudt(u[-1], opd2udx2(u[-3:], self.dx))

        r[1:-1] = self.dudt(u[1:-1], opd2udx2(u, self.dx))
        return r


class LinearPDESolver(PDESolver):
    def __init__(self, D=None, a=None, c=None, **kwargs):
        self.D = D
        self.a = a
        self.c = c
        def dudt(u, upp):
            "du/dt = D d2u/dx2 + a u + c"
            return D * upp + a * u + c
        super(LinearPDESolver, self).__init__(dudt=dudt, **kwargs)

m1s = LinearPDESolver(N=16, L=1., u0=0.,
                      lhs=Dirichlet(1.),
                      rhs=VonNeumann(dudx=0., side='right'), D=1., a=0,
                      dt=0.01, Nt=1000, c=0.)


def main(argv=None):
    if argv is None:
        argv = sys.argv


if __name__ == "__main__":
    sys.exit(main())
