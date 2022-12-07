import numpy as np
import sympy as sp


def D(E, v, type: str = 'plane_strain'):
    # Plane Stress
    if type == 'plane_stress':
        d11 = E / (1 - v**2)
        d12 = v * d11
        d33 = E / (2 * (1 + v))
    # Plane Strain
    if type == 'plane_strain':
        d11 = (E * (1 - v)) / ((1 + v) * (1 - 2 * v))
        d12 = d11 * (v / (1 - v))
        d33 = E / (2 * (1 + v))
    return np.array([[d11, d12, 0], [d12, d11, 0], [0, 0, d33]])


def B(xi: float, eta: float):
    dN1dx, dN1dy = -(1 - eta), -(1 - xi)
    dN2dx, dN2dy = (1 - eta), -(1 + xi)
    dN3dx, dN3dy = (1 + eta), (1 + xi)
    dN4dx, dN4dy = -(1 - eta), (1 - xi)
    return np.array([
        [dN1dx, 0, dN2dx, 0, dN3dx, 0, dN4dx, 0],
        [0, dN1dy, 0, dN2dy, 0, dN3dy, 0, dN4dy],
        [dN1dx, dN1dy, dN2dx, dN2dy, dN3dx, dN3dy, dN4dx, dN4dy]])


if __name__ == "__main__":

    np.set_printoptions(precision=3)

    D_mat = D(1e9, 0.2)
    print(D_mat)

    print((4 / 16) * (D_mat[0, 1] + D_mat[2, 2]))
