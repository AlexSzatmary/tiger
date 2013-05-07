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

    def opdudx(self, u):
        return (self.c - self.a * u) / self.b

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


def opdudx(u, dx):
    """Applies second-order first-derivate finite difference operator on u

    Note that this takes some array, u, of length N, but returns second
    derivatives for only nodes 2 through N-1"""
    return (u[2:] - u[0:-2]) / dx ** 2


def opd2udx2(u, dx):
    """Applies second-derivative finite difference operator on u

    Note that this takes some array, u, of length N, but returns second
    derivatives for only nodes 2 through N-1"""
    return (u[0:-2] - 2. * u[1:-1] + u[2:]) / dx ** 2


# Attributes:
# Given:
#   Physics:
#     f(u1, u2, ..., uN, ux, uxx)
#     u_0
#     uL
#     uR
#   Numerical parameters:
#     N, dx, x
# Solution:
#   u

class PDE(object):
    """
    Really just data representing one PDE in a set of PDE's,
    du_i/dt = f(u_1, u_2, ..., u_m, du_i/dx, d2u_i/dx2)
    where u_1, u_2, ..., u_m are functions of x and t.
    u_0 are the initial conditions for u_i, and u_L and u_r are the right
    and left boundary conditions. L is the length of the domain.
    N is the number of finite difference nodes.
    """
    def __init__(self, f=None, u_0=None, u_L=None, u_r=None, x_L=0, x_r=None,
                 n=None):
        self.f = f
        if issubclass(type(u_0), numbers.Number):
            self.u_0 = np.ones(n) * u_0
        else:
            self.u_0 = u_0(self.x)
        self.u_L = u_L
        self.u_r = u_r
        if type(self.u_L) is Dirichlet:
            self.u_0[0] = self.u_L.u_0
        if type(self.u_r) is Dirichlet:
            self.u_0[-1] = self.u_r.u_0
        self.x_L = x_L
        self.x_r = x_r
        self.n = n
        self.dx = (self.x_r - self.x_L) / (n - 1)
        self.x = np.arange(0, n) * self.dx + self.x_L


class SingleLinearPDE(PDE):
    def __init__(self, D=None, a=None, c=None, **kwargs):
        self.D = D
        self.a = a
        self.c = c

        def f(L_u, up, upp):
            "du/dt = D d2u/dx2 + a u + c"
            return D * upp + a * L_u[0] + c
        super(SingleLinearPDE, self).__init__(f=f, **kwargs)


class CoupledPDESolver(object):
    """
    Time-varying reaction-diffusion solver for coupled species reactions

    Solves
    du/dt = f(u, du/dx, d2u/dx2, v)
    dv/dt = f(v, dv/dx, d2v/dx2, u)
    etc.
    """
    def __init__(self, L_pde=None, dt=None, Nt=None, run=True):
        self.L_pde = L_pde
        self.dt = dt
        self.Nt = Nt

        self.t = np.arange(0, self.Nt) * self.dt
        L_n = [pde.n for pde in L_pde]
        self.L_b_r = np.cumsum(L_n)
        self.L_b_L = self.L_b_r - L_n
        # self.make_x_to_x()
        if run:
            self.run()

    # @staticmethod
    # def x_to_x(x_1, x_2):
    #     return maximum(minimum((db + da) / 2 -
    #                            abs(np.tile(a, (Nb, 1)) -
    #                                np.tile(b, (Na, 1)).T),
    #                            da), 0.)

    # def make_x_to_x(self):
    #     L_x = [pde.x for pde in L_pde]
    #     self.L_L_x_to_x = [[self.x_to_x(x_1, x_2) for x_1 in L_x]
    #                        for x_2 in L_x]

    def run(self):
        self.u = scipy.integrate.odeint(
            self.d_dudt,
            np.hstack([pde.u_0 for pde in self.L_pde]), self.t,
            args=(self,))

        for (b_L, b_r, pde) in zip(self.L_b_L, self.L_b_r, self.L_pde):
            pde.u = self.u[:, b_L:b_r]

    @staticmethod
    def d_dudt(u, t, self):
        """"
        Discretized application of dudt

        Watch out! Because this is a staticmethod, as required by odeint, self
        is the third argument
        """
        # Split the u vector into vectors representing each function being
        # solved for. Note that this creates views of u, so each item in L_u
        # should not be modified.
        L_u = [u[b_L:b_r] for (b_L, b_r) in zip(self.L_b_L, self.L_b_r)]
        r = np.zeros(np.shape(u))
        for (k, b_L, b_r, pde) in zip(range(len(self.L_pde)),
                                      self.L_b_L, self.L_b_r, self.L_pde):
            # L_u_on_x_k = [np.dot(xtox, u) for (xtox, u) in zip(
            #         self.L_L_xtox[k], L_u)]
            r[b_L:b_r] = self.d_dukdt(L_u, k, pde)
        return r

    def d_dukdt(self, L_u, k, pde):
        u = L_u[k]
        r = np.zeros(np.size(u))

        if type(pde.u_L) is Dirichlet:
            r[0] = 0.
        else:
            r[0] = pde.f([utmp[0] for utmp in L_u],
                         pde.u_L.opdudx(u),
                         pde.u_L.opd2udx2(u, pde.dx))

        if type(pde.u_r) is Dirichlet:
            r[-1] = 0.
        else:
            r[-1] = pde.f(
                [utmp[-1] for utmp in L_u],
                pde.u_r.opdudx(u),
                pde.u_r.opd2udx2(u, pde.dx))

        r[1:-1] = pde.f(
            [utmp[1:-1] for utmp in L_u],
            opdudx(u, pde.dx),
            opd2udx2(u, pde.dx))
        return r


class PDESolver(CoupledPDESolver):
    """Time-varying reaction-diffusion type of solver
    Solves du/dt = f(u, du/dx, d2u/dx2)"""
    def __init__(self, pde=None, **kwargs):
        super(PDESolver, self).__init__(L_pde=[pde], **kwargs)
        self.pde = self.L_pde[0]


m1s = PDESolver(
    pde=SingleLinearPDE(
        n=16, x_r=1., u_0=0.,
        u_L=Dirichlet(1.),
        u_r=VonNeumann(dudx=0., side='right'), D=1., a=0,
        c=0.),
    dt=0.01, Nt=1000)

# m2 = CoupledPDESolver(
#     L_pde=[
#         PDE(


def main(argv=None):
    if argv is None:
        argv = sys.argv


if __name__ == "__main__":
    sys.exit(main())
