
import sympy as sp
import numpy as np


def N():
    xi, eta = sp.symbols('xi, eta')
    return sp.Rational(1, 4) * sp.Matrix([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])


if __name__ == "__main__":

    from rich import print

    half = sp.Rational(1, 2)
    a = sp.Matrix([[0, 0], [1, 0], [half, half], [0, 1]])
    b = sp.Matrix([[-1, half], [0, 0], [1, 0], [1, 1]])
    c = sp.Matrix([[0, 0], [1, 0], [3 * half, 1], [-half, 1]])
    d = sp.Matrix([[0, 0], [half, 0], [half, half], [0, half]])
    e = sp.Matrix([[-1, -1], [1, 0], [0, 0], [-half, 1]])

    for coords in [a, b, c, d, e]:
        print('-' * 80)
        sp.pprint(N().dot(coords[:, 0]).simplify())
        sp.pprint(N().dot(coords[:, 1]).simplify())
