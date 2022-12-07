
import numpy as np
from scipy.special import legendre
from scipy.integrate import quad

import matplotlib.pyplot as plt


def fa(x: float):
    return 1 + 2 * x**2 + 3 * 3 * x**3 + 4 * x**4 + 5 * x**5 + 6 * x**6


def fb(x: float):
    return np.sqrt(7) * x**7


def I1(x: float):
    return np.cos((np.pi * x) / 2)


def I2(x: float):
    return x**2 + 1 / np.sqrt(1 + x)


def quadrature(f, a: float, b: float, n: int):
    p = legendre(n)
    x = p.roots
    w = 2 / ((1 - x**2) * (np.polyval(p.deriv(), x))**2)
    v, z = 2 / (b - a), (a + b) / (a - b)
    v, z = (b - a) / 2, (a + b) / 2
    return np.sum(w * f(v * x + z)) * v


if __name__ == "__main__":

    print(quadrature(fa, -3, 4, 3))
    print(quad(fa, -3, 4))
    print(quadrature(fb, -1, 1, 3))
    print(quad(fb, -1, 1))

    I1_exact = 4 / np.pi
    I2_exact = 2 / 3 + 2 * np.sqrt(2)
    print(I1_exact, I2_exact)
    points = [1, 2, 4, 16, 32]
    E1, E2 = [], []
    for n in points:
        y1 = quadrature(I1, -1, 1, n)
        E1 += [np.abs((y1 - I1_exact) / I1_exact)]
        y2 = quadrature(I2, -1, 1, n)
        E2 += [np.abs((y2 - I2_exact) / I2_exact)]

    print(E1)
    print(E2)

    plt.figure()
    plt.plot(points, E1)
    plt.plot(points, E2)
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Errors From Gauss-Legendre Quadrature')
    plt.xlabel('Number of Points in Quadrature')
    plt.ylabel('Error')
    plt.legend(['I1', 'I2'])
    plt.grid()
    plt.show()
