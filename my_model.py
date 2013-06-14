import numpy as np
from tiger import *

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


m2 = CoupledPDESolver(
    L_pde=[
        PDE(f=lambda L_u, dudx, d2udx2: d2udx2,
            u_0=0., u_L=Dirichlet(1.),
            u_r=Dirichlet(0.), x_r=1., n=16),
        PDE(f=lambda L_u, dudx, d2udx2: -10 * (L_u[1] + L_u[0]),
            u_0=0., u_L=Dirichlet(-1.),
            u_r=Dirichlet(0.), x_r=1., n=16)],
    dt=0.01, Nt=1000)


def f0(L_u, dudx, d2udx2):
    m = L_u[0]
    c = L_u[1]
    m_s = L_u[2]
    return d2udx2 + np.maximum(0., m_s - m) - m * c

m3 = CoupledPDESolver(
    L_pde=[
        #m
        PDE(f=f0,
            u_0=0., u_L=VonNeumann(dudx=0., side='left'),
            u_r=VonNeumann(dudx=0., side='right'), x_L=0., x_r=11., n=128),
        #c
        PDE(f=lambda L_u, dudx, d2udx2: -L_u[1] * L_u[0],
            u_0=1., u_L=Dirichlet(0.),
            u_r=Dirichlet(0.), x_L=0.5, x_r=11.5, n=12),
        #m_s
        PDE(f=lambda L_u, dudx, d2udx2: -(L_u[2] - L_u[0]),
            u_0=10., u_L=Dirichlet(0.),
            u_r=Dirichlet(0.), x_L=-0.5, x_r=1.5, n=3)],
    dt=0.01, Nt=1000, run=False)


def m40(L_u, dudx, d2udx2):
    m = L_u[0]
    s = L_u[1]
    return d2udx2 + s - m

m4 = CoupledPDESolver(
    L_pde=[
        #m
        PDE(f=m40,
            u_0=0., u_L=VonNeumann(dudx=0., side='left'),
            u_r=VonNeumann(dudx=0., side='right'), x_L=0., x_r=6., n=25),
        #m_s
        PDE(f=lambda L_u, dudx, d2udx2: -(L_u[1] - L_u[0]),
            u_0=10., u_L=Dirichlet(0.),
            u_r=Dirichlet(0.), x_L=-0.5, x_r=1.5, n=3)],
    dt=0.01, Nt=1000, run=False)
