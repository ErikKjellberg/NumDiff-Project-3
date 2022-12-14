import numpy as np
import scipy.linalg


def RMS_norm(u, dx):
    return np.linalg.norm(u) * np.sqrt(dx)


def get_Sdx(dx, N):
    col = np.zeros(N)
    col[1] = -1 / (2 * dx)
    col[-1] = 1 / (2 * dx)
    return scipy.linalg.circulant(col)


def get_Tdx(dx, N):
    col = np.zeros(N)
    col[0] = -2 / (dx * dx)
    col[1] = 1 / (dx * dx)
    col[-1] = 1 / (dx * dx)
    return scipy.linalg.circulant(col)


# Trap. rule
# y_(n+1)=y_n+h/2*(A*y_n+A*y_(n+1))
# (I-A*h/2)y_(n+1)=(I+A*h/2)y_n
# y_(n+1)=(I-A*h/2)^(-1)(I+A*h/2)y_n
def get_trap(A, dt):
    N = A.shape[0]
    return np.dot(np.linalg.inv(np.eye(N) - A * dt / 2), (np.eye(N) + A * dt / 2))
