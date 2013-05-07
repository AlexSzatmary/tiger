import numpy as np
import my_model
import nose

class testNoSource(object):
    def setUp(self):
        self.model = my_model.PDESolver(
            my_model.SingleLinearPDE(
                n=16, x_r=1., u_0=0., u_L=my_model.Dirichlet(1.),
                u_r=my_model.Dirichlet(0.), D=1., a=0, c=0.),
            dt=0.01, Nt=1000, run=True)
        self.solution = 1. - self.model.pde.x

    def test(self):
        nose.tools.assert_almost_equal(
            np.max(np.abs(self.model.pde.u[-1] - self.solution)), 0.)

class testUniformSource(testNoSource):
    def setUp(self):
        self.model = my_model.PDESolver(
            my_model.SingleLinearPDE(
                n=16, x_r=1., u_0=0., u_L=my_model.Dirichlet(0.),
                u_r=my_model.Dirichlet(0.), D=1., a=0, c=1.),
                dt=0.01, Nt=1000, run=True)
        self.solution = self.model.pde.x * (1 - self.model.pde.x) / 2.

class testVonNeumannBC(testNoSource):
    def setUp(self):
        self.model = my_model.PDESolver(
            my_model.SingleLinearPDE(
                n=16, x_r=1., u_0=0., u_L=my_model.Dirichlet(0.),
                u_r=my_model.VonNeumann(dudx=0., side='right'), D=1., a=0,
                c=1.),
            dt=0.01, Nt=1000, run=True)
        self.solution = self.model.pde.x * (2 - self.model.pde.x) / 2.
    

class testCoupledPDESolver(object):
    def setUp(self):
        self.model = my_model.CoupledPDESolver(
            L_pde=[
                my_model.PDE(f=lambda L_u, dudx, d2udx2: d2udx2,
                             u_0=0., u_L=my_model.Dirichlet(1.),
                             u_r=my_model.Dirichlet(0.), x_r=1., n=16),
                my_model.PDE(
                    f=lambda L_u, dudx, d2udx2: -10 * (L_u[1] + L_u[0]),
                    u_0=0., u_L=my_model.Dirichlet(-1.),
                    u_r=my_model.Dirichlet(0.), x_r=1., n=16)],
            dt=0.01, Nt=1000, run=True)
        self.solution = [1. - self.model.L_pde[0].x,
                         -1 + self.model.L_pde[1].x]

    def test(self):
        nose.tools.assert_almost_equal(
            np.max(np.abs(self.model.L_pde[0].u[-1] - self.solution[0])), 0.)
        nose.tools.assert_almost_equal(
            np.max(np.abs(self.model.L_pde[1].u[-1] - self.solution[1])), 0.)


#    def test(self):
# For uniform source
#        
