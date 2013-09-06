import numpy as np
import tiger
import nose

class testNoSource(object):
    def setUp(self):
        self.model = tiger.PDESolver(
            tiger.SingleLinearPDE(
                n=16, x_r=1., u_0=0., u_L=tiger.Dirichlet(1.),
                u_r=tiger.Dirichlet(0.), D=1., a=0, c=0.),
            dt=0.01, Nt=1000, run=True)
        self.solution = 1. - self.model.pde.x

    def test(self):
        nose.tools.assert_almost_equal(
            np.max(np.abs(self.model.pde.u[-1] - self.solution)), 0.)

class testUniformSource(testNoSource):
    def setUp(self):
        self.model = tiger.PDESolver(
            tiger.SingleLinearPDE(
                n=16, x_r=1., u_0=0., u_L=tiger.Dirichlet(0.),
                u_r=tiger.Dirichlet(0.), D=1., a=0, c=1.),
                dt=0.01, Nt=1000, run=True)
        self.solution = self.model.pde.x * (1 - self.model.pde.x) / 2.

class testVonNeumannBC(testNoSource):
    def setUp(self):
        self.model = tiger.PDESolver(
            tiger.SingleLinearPDE(
                n=16, x_r=1., u_0=0., u_L=tiger.Dirichlet(0.),
                u_r=tiger.VonNeumann(dudx=0.), D=1., a=0,
                c=1.),
            dt=0.01, Nt=1000, run=True)
        self.solution = self.model.pde.x * (2 - self.model.pde.x) / 2.
    

class testCoupledPDESolver(object):
    def setUp(self):
        self.model = tiger.CoupledPDESolver(
            L_pde=[
                tiger.PDE(f=lambda L_u, dudx, d2udx2: d2udx2,
                             u_0=0., u_L=tiger.Dirichlet(1.),
                             u_r=tiger.Dirichlet(0.), x_r=1., n=16),
                tiger.PDE(
                    f=lambda L_u, dudx, d2udx2: -10 * (L_u[1] + L_u[0]),
                    u_0=0., u_L=tiger.Dirichlet(-1.),
                    u_r=tiger.Dirichlet(0.), x_r=1., n=16)],
            dt=0.01, Nt=1000, run=True)
        self.solution = [1. - self.model.L_pde[0].x,
                         -1 + self.model.L_pde[1].x]

    def test(self):
        nose.tools.assert_almost_equal(
            np.max(np.abs(self.model.L_pde[0].u[-1] - self.solution[0])), 0.)
        nose.tools.assert_almost_equal(
            np.max(np.abs(self.model.L_pde[1].u[-1] - self.solution[1])), 0.)


class testGridConvert(testCoupledPDESolver):
    def setUp(self):
        self.model = tiger.CoupledPDESolver(
            L_pde=[
                tiger.PDE(f=lambda L_u, dudx, d2udx2: d2udx2,
                             u_0=0., u_L=tiger.Dirichlet(1.),
                             u_r=tiger.Dirichlet(0.), x_r=1., n=25),
                tiger.PDE(
                    f=lambda L_u, dudx, d2udx2: -10 * (L_u[1] + L_u[0]),
                    u_0=0., u_L=tiger.Dirichlet(-1.),
                    u_r=tiger.Dirichlet(0.), x_r=1., n=16)],
            dt=0.01, Nt=1000, run=True)
        self.solution = [1. - self.model.L_pde[0].x,
                         -1 + self.model.L_pde[1].x]

    #todo Make this work
    def test(self):
        pass

#    def test(self):
# For uniform source
#        

class testContinuityCondition(object):
    def setUp(self):
        c = tiger.Continuity(left_index=0, right_index=1)
        self.model = tiger.CoupledPDESolver(
            L_pde=[
                tiger.PDE(f=lambda L_u, dudx, d2udx2: 2 * d2udx2 + 1.,
                    u_0=0., u_L=tiger.Dirichlet(0.),
                    u_r=c, x_L=-1., x_r=0., n=65),
                tiger.PDE(f=lambda L_u, dudx, d2udx2: d2udx2 + 1.,
                    u_0=0., u_L=c,
                    u_r=tiger.Dirichlet(0.), x_L=0., x_r=1., n=65)],
            dt=1., Nt=1000, run=False)
        self.model.L_pde[0].D = 2.
        self.model.L_pde[1].D = 1.
        self.model.run()

        self.solution = [
            (lambda x: -x ** 2 / 4. + x / 12. + 1./3.)(self.model.L_pde[0].x),
            (lambda x: -x ** 2 / 2. + x / 6. + 1./3.)(self.model.L_pde[1].x)]

    def test(self):
        nose.tools.assert_less_equal(
            np.max(np.abs(self.model.L_pde[0].u[-1] - self.solution[0])),
            0.01)

        nose.tools.assert_less_equal(
            np.max(np.abs(self.model.L_pde[1].u[-1] - self.solution[1])), 
            0.01)

class testContinuityConditionDiscontinuity(object):
    def setUp(self):
        c = tiger.Continuity(left_index=0, right_index=1)
        self.model = tiger.CoupledPDESolver(
            L_pde=[
                tiger.PDE(f=lambda L_u, dudx, d2udx2: 2 * d2udx2 + 1.,
                    u_0=1., u_L=tiger.Dirichlet(0.),
                    u_r=c, x_L=-1., x_r=0., n=65),
                tiger.PDE(f=lambda L_u, dudx, d2udx2: d2udx2 + 1.,
                    u_0=0., u_L=c,
                    u_r=tiger.Dirichlet(0.), x_L=0., x_r=1., n=65)],
            dt=1e-3, Nt=1000, run=False)
        self.model.L_pde[0].u_0[-1] = 0.
        self.model.L_pde[0].D = 2.
        self.model.L_pde[1].D = 1.
        self.model.run()

        self.solution = [
            (lambda x: -x ** 2 / 4. + x / 12. + 1./3.)(self.model.L_pde[0].x),
            (lambda x: -x ** 2 / 2. + x / 6. + 1./3.)(self.model.L_pde[1].x)]

    def test(self):
        nose.tools.assert_less_equal(
            np.max(np.abs(self.model.L_pde[0].u[-1] - self.solution[0])),
            0.01)

        nose.tools.assert_less_equal(
            np.max(np.abs(self.model.L_pde[1].u[-1] - self.solution[1])), 
            0.01)

