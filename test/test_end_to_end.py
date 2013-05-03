import numpy as np
import my_model
import nose

class testNoSource(object):
    def setUp(self):
        self.pde = my_model.LinearPDESolver(
            N=16, L=1., u0=0., lhs=my_model.Dirichlet(1.),
            rhs=my_model.Dirichlet(0.), D=1., a=0,
            dt=0.01, Nt=1000, c=0.)
        self.solution = 1. - self.pde.x

    def test(self):
        nose.tools.assert_almost_equal(
            np.max(np.abs(self.pde.u[-1] - self.solution)), 0.)

class testUniformSource(testNoSource):
    def setUp(self):
        self.pde = my_model.LinearPDESolver(
            N=16, L=1., u0=0., lhs=my_model.Dirichlet(0.),
            rhs=my_model.Dirichlet(0.), D=1., a=0,
            dt=0.01, Nt=1000, c=1.)
        self.solution = self.pde.x * (1 - self.pde.x) / 2.

class testVonNeumannBC(testNoSource):
    def setUp(self):
        self.pde = my_model.LinearPDESolver(
            N=16, L=1., u0=0., lhs=my_model.Dirichlet(0.),
            rhs=my_model.VonNeumann(dudx=0., side='right'), D=1., a=0,
            dt=0.01, Nt=1000, c=1.)
        self.solution = self.pde.x * (2 - self.pde.x) / 2.
    

#    def test(self):
# For uniform source
#        
