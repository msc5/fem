
import numpy as np
import sympy as sp

from rich import print


def centroid(P: np.ndarray):
    n, d = P.shape
    assert d == 2 or d == 3
    if d == 2:
        P = np.c_[np.array(P), np.zeros(n)]
    A = (1 / 2) * np.array([np.linalg.norm(np.cross(P[i], P[i + 1]))
                            for i in range(n - 1)]).reshape((n - 1, 1))
    C = (1 / 3) * np.array([P[i] + P[i + 1] for i in range(n - 1)])
    return np.sum(A), np.sum(A * C, 0) / np.sum(A)


if __name__ == "__main__":

    points = np.array([[0, 0], [1, 0], [1, 1], [-1, 1 / 2]])
    print(points)

    c = centroid(points)
    print(c)

    u_p = np.array([[0.3, 0.2], [0.5, 0.1], [0.2, 0.4], [0.4, 0.3]])

    u_m = (1 / 4) * np.array([1, 1, 1, 1]) @ u_p
    print(u_m)

    J_inv = sp.Rational(2, 5) * sp.Matrix([[1, -6], [3, 2]]).T
    DNiDxe = sp.Rational(1, 4) * sp.Matrix([[-1, 1, 1, -1], [-1, -1, 1, 1]])

    DNiDxy = J_inv @ DNiDxe

    sp.pprint(DNiDxy)

    a = np.array([0.3, 0.2, 0.5, 0.1, 0.2, 0.4, 0.4, 0.3])
    B = (1 / 10) * np.array([[-4, 0, -2, 0, 4, 0, 2, 0],
                            [0, 4, 0, -8, 0, -4, 0, 8],
                            [4, -4, -8, -2, -4, 4, 8, 2]])
    strain = B @ a
    print('Strain')
    print(strain)

    E, v = 1e9, 0.2
    d11 = E / (1 - v**2)
    d12 = (E * v) / (1 - v**2)
    d33 = E / (2 * (1 + v))
    D = np.array([[d11, d12, 0], [d12, d11, 0], [0, 0, d33]])

    stress = D @ strain
    print('Stress')
    print(stress)
