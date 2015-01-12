#!/usr/bin/env python
import numpy as np
import scipy
import scipy.integrate
import numbers


# Boundary conditions
class Dirichlet(object):
    '''
    Represents Dirichlet boundary conditions that are constant over time.
    '''
    def __init__(self, u_0):
        self.u_0 = u_0

    def set_pde(self, pde, side):
        pass

    def opdudt(self, t, x, L_u, u_k):
        return 0.


class Robin(object):
    """Boundary condition for a u + b u' = c"""
    def __init__(self, a=None, b=None, c=None):
        self.a = a
        self.b = b
        self.c = c

    def set_pde(self, pde, side):
        self.dx = pde.dx
        self.f = pde.f
        self.side = side

    def opdudx(self, u):
        if self.side == 'left':
            return (self.c - self.a * u[0]) / self.b
        elif self.side == 'right':
            return (self.c - self.a * u[-1]) / self.b

    def opd2udx2(self, u):
        if self.side == 'left':
            return (2 * u[1] +
                    (-2 + 2 * self.dx * self.a / self.b) * u[0]
                    - 2 * self.dx * self.c / self.b) / self.dx ** 2
        elif self.side == 'right':
            return (2 * u[-2] +
                    (-2 - 2 * self.dx * self.a / self.b) * u[-1]
                    + 2 * self.dx * self.c / self.b) / self.dx ** 2

    def opdudt(self, t, x, L_u, u_k):
        if self.side == 'left':
            u_b = [u[0] for u in L_u]
        elif self.side == 'right':
            u_b = [u[-1] for u in L_u]
        return self.f(t, x, u_b, self.opdudx(u_k), self.opd2udx2(u_k))


class VonNeumann(Robin):
    def __init__(self, dudx=None):
        super(VonNeumann, self).__init__(a=0, b=1, c=dudx)


class SphericalVonNeumannZero(object):
    '''
    This is the boundary condition to use for r=0 in a PDE in spherical
    coordinates

    It is always assumed that r in [0, r_max], so this bc is the bc at the
    left end of the domain.
    '''
    def __init__(self):
        pass

    def set_pde(self, pde, side):
        self.pde = pde
        self.dx = pde.dx

    def opdudt(self, t, x, L_u, u_k):
        # This operator is derived as,
        # du/dt = (D/r^2) d/dr (r^2 du/dr)
        #       = D * (d^2 u/dr^2 + (2 / r) * du / dr)
        # Taking the limit as r -> 0:
        # (2 / r) * du / dr -> 2 * d^2 u/dr^2
        # therefore, du/dt = 3 * d^2 u / dr^2 in the limit of r -> 0
        # d^2u/dr^2 @r=0 ~ (u[-1] - 2 * u[0] + u[1]) / dr^2
        # u[-1] = u[1] (symmetry)
        # therefore,
        # du/dt = 3 * D * 2 * (u[1] - u[0]) / dr^2
        return 6 * self.pde.D * (u_k[1] - u_k[0]) / self.dx ** 2


class CylindricalVonNeumannZero(object):
    '''
    This is the boundary condition to use for r=0 in a PDE in cylindrical
    coordinates

    It is always assumed that r in [0, r_max], so this bc is the bc at the
    left end of the domain.
    '''
    def __init__(self):
        pass

    def set_pde(self, pde, side):
        self.pde = pde
        self.dx = pde.dx

    def opdudt(self, t, x, L_u, u_k):
        # This operator is derived as,
        # du/dt = (D/r) d/dr (r du/dr)
        #       = D * (d^2 u/dr^2 + (1 / r) * du / dr)
        # Taking the limit as r -> 0:
        # (1 / r) * du / dr -> 1 * d^2 u/dr^2
        # therefore, du/dt = 2 * d^2 u / dr^2 in the limit of r -> 0
        # d^2u/dr^2 @r=0 ~ (u[-1] - 2 * u[0] + u[1]) / dr^2
        # u[-1] = u[1] (symmetry)
        # therefore,
        # du/dt = 2 * D * 2 * (u[1] - u[0]) / dr^2
        return 4 * self.pde.D * (u_k[1] - u_k[0]) / self.dx ** 2


class Continuity(object):
    '''
    A continuity boundary condition for the heat equation in rectangular
    coordinates with no source
    '''
    def __init__(self, left_index=None, right_index=None, L_pde=None):
        '''
        Set up the continuity condition for the diffusion after initializing
        the PDEs it links. left_index and right_index are the indices to L_pde
        that give the subdomains on the left and right sides of the continuity
        point, respectively. L_pde is the list of PDEs.

        This assumes that u at the point of continuity is the same for both
        subdomains.
        '''
        self.left_index = left_index
        self.right_index = right_index

    def set_pde(self, pde, side):
        if side == 'left':
            self.right = pde
        if side == 'right':
            self.left = pde

    def opdudt(self, t, x, L_u, u_k):
        '''
        Operator giving du/dt for point of continuity. This kind of operator
        for the continuity condition is specific to a problem type, but is not
        difficult to re-implement for other problems. It would probably be
        best to expand its usefulness by deriving it explicitly in terms of
        du/dx and d2u/dx2 operators. As it is now, it's just the d2u/dx2
        operator.
        '''
        u_a_m2 = L_u[self.left_index][-2]
        u_ab = L_u[self.right_index][0]
        u_b_1 = L_u[self.right_index][1]
        return ((self.right.D * (u_b_1 - u_ab) / self.right.dx -
                 self.left.D * (u_ab - u_a_m2) / self.left.dx) /
                (self.left.dx / 2. + self.right.dx / 2.))


class SphericalContinuity(Continuity):
    '''
    A continuity boundary condition for the heat equation in spherical
    coordinates with no source
    '''
    def opdudt(self, t, x, L_u, u_k):
        '''
        Operator giving du/dt for point of continuity in spherical coordinates
        '''
        r_a_m2 = self.left.x[-2]
        r_ab = self.right.x[0]
        r_b_1 = self.right.x[1]
        # The same operator from opdudt in the Continuity class can be used
        # here by scaling u by r as in the following 3 lines, and then scaling
        # the result back again.
        u_a_m2 = L_u[self.left_index][-2] * r_a_m2 ** 2
        u_ab = L_u[self.right_index][0] * r_ab ** 2
        u_b_1 = L_u[self.right_index][1] * r_b_1 ** 2
        # This return is the same as Continuity.opdudt, but divided by r_ab
        return ((self.right.D * (u_b_1 - u_ab) / self.right.dx -
                 self.left.D * (u_ab - u_a_m2) / self.left.dx) /
                (self.left.dx / 2. + self.right.dx / 2.)) / r_ab ** 2



class CylindricalContinuity(Continuity):
    '''
    A continuity boundary condition for the heat equation in cylindrical
    coordinates with no source
    '''
    def opdudt(self, t, x, L_u, u_k):
        '''Operator giving du/dt for point of continuity in cylindrical
        coordinates
        '''
        r_a_m2 = self.left.x[-2]
        r_ab = self.right.x[0]
        r_b_1 = self.right.x[1]
        # The same operator from opdudt in the Continuity class can be used
        # here by scaling u by r as in the following 3 lines, and then scaling
        # the result back again.
        u_a_m2 = L_u[self.left_index][-2] * r_a_m2
        u_ab = L_u[self.right_index][0] * r_ab
        u_b_1 = L_u[self.right_index][1] * r_b_1
        # This return is the same as Continuity.opdudt, but divided by r_ab
        return ((self.right.D * (u_b_1 - u_ab) / self.right.dx -
                 self.left.D * (u_ab - u_a_m2) / self.left.dx) /
                (self.left.dx / 2. + self.right.dx / 2.)) / r_ab



def opdudx(u, dx):
    """Applies second-order first-derivate finite difference operator on u

    Note that this takes some array, u, of length N, but returns
    derivatives for only nodes 2 through N-1"""
    return (u[2:] - u[0:-2]) / (2 * dx)


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
        self.x_L = x_L
        self.x_r = x_r
        self.n = n
        self.dx = (self.x_r - self.x_L) / (n - 1)
        self.x = np.arange(0, n) * self.dx + self.x_L
        if issubclass(type(u_0), numbers.Number):
            self.u_0 = np.ones(n) * u_0
        elif issubclass(type(u_0), np.ndarray):
            self.u_0 = u_0
        else:
            self.u_0 = u_0(self.x)
        self.u_L = u_L
        self.u_L.set_pde(self, 'left')
        self.u_r = u_r
        self.u_r.set_pde(self, 'right')
        if type(self.u_L) is Dirichlet:
            self.u_0[0] = self.u_L.u_0
        if type(self.u_r) is Dirichlet:
            self.u_0[-1] = self.u_r.u_0


class SingleLinearPDE(PDE):
    def __init__(self, D=None, a=None, c=None, **kwargs):
        self.D = D
        self.a = a
        self.c = c

        def f(t, x, L_u, up, upp):
            "du/dt = D d2u/dx2 + a u + c"
            return D * upp + a * L_u[0] + c
        super(SingleLinearPDE, self).__init__(f=f, **kwargs)


def x_to_x(pde_1, pde_2):
    a = np.tile(pde_1.x, (pde_2.n, 1))
    b = np.tile(pde_2.x, (pde_1.n, 1)).T
    a_l = a - pde_1.dx / 2
    a_r = a + pde_1.dx / 2
    b_l = b - pde_2.dx / 2
    b_r = b + pde_2.dx / 2
    x_l = np.maximum(a_l, b_l)
    x_r = np.minimum(a_r, b_r)
    return np.maximum(x_r - x_l, 0.) / pde_2.dx


class CoupledPDESolver(object):
    """
    Time-varying reaction-diffusion solver for coupled species reactions

    Solves
    du/dt = f(u, du/dx, d2u/dx2, v)
    dv/dt = f(v, dv/dx, d2v/dx2, u)
    etc.
    """
    def __init__(self, L_pde=None, dt=None, Nt=None, run=False):
        self.L_pde = L_pde
        self.dt = dt
        self.Nt = Nt

        self.t = np.arange(0, self.Nt) * self.dt
        L_n = [pde.n for pde in L_pde]
        self.L_b_r = np.cumsum(L_n)
        self.L_b_L = self.L_b_r - L_n
        self.make_x_to_x()
        if run:
            self.run()

    def make_x_to_x(self):
        self.L_L_x_to_x = [[x_to_x(pde_1, pde_2) for pde_1 in self.L_pde]
                           for pde_2 in self.L_pde]

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
        L_u = self.split_u(u)
        r = np.zeros(np.shape(u))
        for (k, b_L, b_r, pde) in zip(range(len(self.L_pde)),
                                      self.L_b_L, self.L_b_r, self.L_pde):
            # L_u_on_x_k = [np.dot(xtox, u) for (xtox, u) in zip(
            #         self.L_L_xtox[k], L_u)]
            r[b_L:b_r] = self.d_dukdt(t, L_u, k, pde)
        return r

    def split_u(self, u):
        '''
        Splits the u vector handled by odeint into each u_i
        '''
        return [u[b_L:b_r] for (b_L, b_r) in zip(self.L_b_L, self.L_b_r)]

    def d_dukdt(self, t, L_u, k, pde):
        L_u_on_x = [np.dot(x_to_x, utmp) for (x_to_x, utmp) in
                    zip(self.L_L_x_to_x[k], L_u)]
        u_k = L_u[k]
        # r is the value of du/dt, not the new u value.
        r = np.zeros(np.size(u_k))

        r[1:-1] = pde.f(
            t, pde.x[1:-1],
            [utmp[1:-1] for utmp in L_u_on_x],
            opdudx(u_k, pde.dx),
            opd2udx2(u_k, pde.dx))
        r[0] = pde.u_L.opdudt(t, pde.x[0], L_u, u_k)
        r[-1] = pde.u_r.opdudt(t, pde.x[-1], L_u, u_k)
        return r


class PDESolver(CoupledPDESolver):
    """Time-varying reaction-diffusion type of solver
    Solves du/dt = f(u, du/dx, d2u/dx2)"""
    def __init__(self, pde=None, **kwargs):
        super(PDESolver, self).__init__(L_pde=[pde], **kwargs)
        self.pde = self.L_pde[0]


def main(argv=None):
    if argv is None:
        argv = sys.argv


if __name__ == "__main__":
    sys.exit(main())
