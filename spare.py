# The following models are spare parts

# Turing type model
class Model(object):
    def __init__(self, N=16, dx = None, fu=None, fv=None, gu=None, gv=None,
                 Du=None, Dv=None, Nt=None, dt=None):
        self.N = N
        self.dx = dx
        self.U = np.zeros(self.N)
        self.V = np.zeros(self.N)
        self.fu = fu
        self.fv = fv
        self.gu = gu
        self.gv = gv
        self.Du = Du
        self.Dv = Dv
        self.Nt = Nt
        self.dt = dt

    def odeint_wrapper(self):
        y0 = np.hstack((self.U, self.V))
        t = np.arange(self.Nt) * self.dt
        self.y = scipy.integrate.odeint(
            self.dydt, y0, t,
            args=(self.N, self.dx,
                  self.fu, self.fv, self.gu,
                  self.gv, self.Du, self.Dv))
        
    @staticmethod
    def dydt(y, t, N, dx, fu, fv, gu, gv, Du, Dv):
        foo = np.zeros(N * 2)
        U = y[:N]
        V = y[N:]
        foo[0] = (fu * U[0] + fv * V[0] +
                  Du * (U[-1] - 2. * U[0] + U[1]) /
                  dx ** 2)
        foo[1:N - 1] = (fu * U[1:-1] + fv * V[1:-1] +
                             Du * (U[0:-2] - 2. * U[1:-1] +
                                        U[2:]) /
                             dx ** 2)
        foo[N - 1] = (fu * U[-1] + fv * V[-1] +
                           Du * (U[-2] - 2. * U[-1] +
                                      U[0]) /
                           dx ** 2)
                
        foo[N] = (gu * U[0] + gv * V[0] +
                  Dv * (V[-1] - 2. * V[0] + V[1]) /
                  dx ** 2)
        foo[N + 1:-1] = (gu * U[1:-1] + gv * V[1:-1] +
                              Dv * (V[0:-2] - 2. * V[1:-1] +
                                         V[2:]) /
                              dx ** 2)
        foo[-1] = (gu * U[-1] + gv * V[-1] +
                           Dv * (V[-2] - 2. * V[-1] +
                                      V[0]) /
                           dx ** 2)
        return foo

a = 1.
Ta = Model(N=16, dx=1., fu=a, fv=1., gu=1., gv=a,
           Du=0.25, Dv=0.25, Nt=100, dt=0.01)

class RDSSModel(object):
    # Solves y'' = f(x, y) at steady-state using finite differences
    def __init__(self, x_a=0., x_b=1., n_x=11, f=None):
        self.x_a = x_a
        self.x_b = x_b
        self.x = np.linspace(x_a, x_b, n_x)
        self.y = np.zeros(n_x)
        self.h = (x_b - x_a) / (n_x - 1)
        self.T = 1. / self.h ** 2.
        self.f = f

    def iterate(self):
        self.Sinv = 1. / (2. / self.h ** 2. + self.f(self.x, self.y))
        self.y[1:-1] = (self.T * self.Sinv[:-2] * self.y[:-2] +
                        self.T * self.Sinv[2:] * self.y[2:])
        self.boundary_conditions()

    def boundary_conditions(self):
        pass

