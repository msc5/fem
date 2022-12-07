
import numpy as np
import matplotlib.pyplot as plt

from rich import print


E = 200e9       # Pa
L = 2           # m
A_o = 15e-4     # m^2
A_l = 5e-4      # m^2
q_o = 750e3     # N / m


def A(x: float) -> float:
    return ((A_l - A_o) / L) * x + A_o


def A_mid(x_i: float, x_j: float) -> float:
    return A(x_i + (x_j - x_i) / 2)


def q(x: float) -> float:
    return q_o * x


def q_mid(x_i: float, x_j: float) -> float:
    return q(x_i + (x_j - x_i) / 2)


def stiffness(x_i: float, x_j: float):
    K_c = E * A_mid(x_i, x_j) / (x_j - x_i)
    return K_c * np.array([[1, -1], [-1, 1]])


def dist_load(x_i: float, x_j: float):
    q_c = (q_mid(x_i, x_j) * (x_j - x_i)) / 2
    return q_c * np.ones((2, 1))


def u_exact(x: float) -> float:
    a = (A_l - A_o) / L
    b = A_o
    u = a * x + b
    sq = ((a * L)**2 - b**2) / (a**3)
    c = sq * np.log(b)
    return - (q_o / (2 * E)) * (
        # (u**2) / (2 * a**3) +
        # (2 * b * u) / (a**3) -
        (a * x**2 - 2 * b * x) / (2 * a**2) -
        sq * np.log(np.abs(u)) + c
    )


def FEM(n_elements: int):

    print('')
    s = 'FEM n = ' + str(n_elements) + ' '
    print(s + '-' * (80 - len(s)))
    print('')

    n_nodes = n_elements + 1

    u_c = np.array([False] + [True] * n_elements)

    X = np.linspace(0, L, n_nodes)
    EFT = np.array([np.array([0, 1]) + i for i in range(n_elements)])

    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros((n_nodes, 1))
    for member in EFT:
        nodes = X[member]
        K_e = stiffness(*nodes)
        K[np.ix_(member, member)] += K_e
        F_e = dist_load(*nodes)
        F[member] += F_e

    print('K\n', K, K.shape)
    print('F\n', F, F.shape)
    K = K[np.ix_(u_c, u_c)]
    F = F[u_c]

    u = np.zeros((n_nodes, 1))
    u[u_c] += np.linalg.solve(K, F)
    print('u\n', u)

    return X, u


if __name__ == '__main__':

    N_elements = [1, 2, 4]
    # N_elements = np.arange(1, 5)
    colors = plt.cm.plasma(np.linspace(0, 1, len(N_elements) * 2))
    X_exact = np.linspace(0, L, 1000)
    fig, ax = plt.subplots()
    for i, n in enumerate(N_elements):
        ax.plot(*FEM(n), color=colors[i])
    ax.plot(X_exact, u_exact(X_exact), color='black')
    ax.set_title('FEM vs. Exact Solution for Tapered Bar')
    ax.set_ylabel('Axial Displacement (m)')
    ax.set_xlabel('Position Along Bar (m)')
    ax.legend(['FEM Solution n = ' + str(n) for n in N_elements] +
              ['Exact Solution'])

    plt.show()
