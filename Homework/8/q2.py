
import numpy as np


def abc(n: int, X, Y):
    assert n >= 1 and n <= 3
    idx = np.array([0, 1, 2])
    i = idx[(n - 1) % 3]
    j = idx[(n) % 3]
    k = idx[(n + 1) % 3]
    a = X[j] * Y[k] - X[k] * Y[j]
    b = Y[j] - Y[k]
    c = X[k] - X[j]
    return a, b, c


def B(X, Y):
    _, dN1dx, dN1dy = abc(1, X, Y)
    _, dN2dx, dN2dy = abc(2, X, Y)
    _, dN3dx, dN3dy = abc(3, X, Y)
    A = (1 / 2) * np.linalg.norm(np.cross(X, Y))
    return (1 / (2 * A)) * np.array([
        [dN1dx, 0, dN2dx, 0, dN3dx, 0],
        [0, dN1dy, 0, dN2dy, 0, dN3dy],
        [dN1dy, dN1dx, dN2dy, dN2dx, dN3dy, dN3dx]])


def D(E, v):
    d11 = E / (1 + v**2)
    d12 = v * d11
    d33 = E / (2 * (1 + v))
    return np.array([
        [d11, d12, 0],
        [d12, d11, 0],
        [0, 0, d33]])


if __name__ == "__main__":

    from rich import print
    np.set_printoptions(precision=3)

    def section(s: str):
        print('')
        print(s + ' ' + (80 - 1 - len(s)) * '-')
        print('')

    X = np.array([0, 2, 1])
    Y = np.array([0, 1, 2])
    Ux = np.array([0.6, 0.7, 0.8])
    Uy = np.array([0.7, 0.8, 0.7])
    U = np.zeros(Ux.size + Uy.size)
    U[0::2], U[1::2] = Ux, Uy

    strain = (B(X, Y) @ U).reshape((3, 1))
    stress = (D(1e9, 0.2) @ strain)

    section('(a, b, c) for N1, N2, N3:')
    print(abc(1, X, Y))
    print(abc(2, X, Y))
    print(abc(3, X, Y))

    section('B Matrix')
    print(B(X, Y))

    section('Strain:')
    print(strain)

    section('Stress:')
    print(stress)
