import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from util import get_Tdx, get_trap


def first_diff(u, dx):
    u1 = np.concatenate(([u[-1]], u[:-1]))
    u2 = np.concatenate((u[:-1], [u[0]]))
    return (u1 - u2) / (2 * dx)


def second_diff(u, dx):
    u_before = np.concatenate(([u[-1]], u[:-1]))
    u_after = np.concatenate((u[1:], [u[0]]))
    return 1 / (dx * dx) * (u_before - 2 * u + u_after)


def inviscid_burger_scheme(u, dx, dt):
    ux = first_diff(u, dx)
    uxx = second_diff(u, dx)
    uux = np.multiply(u, ux)
    uxux = np.multiply(ux, ux)
    uuxx = np.multiply(u, uxx)
    return u - dt * uux + dt * dt / 2 * np.multiply((2 * uxux + uuxx), u)


def viscous_burger_scheme(u, d, dx, dt):
    global Tdx, ITdx
    return np.dot(ITdx, inviscid_burger_scheme(u, dx, dt) + d * dt / 2 * np.dot(Tdx, u))


x0 = 0
x1 = 1
N = 250  # amount of grid points on x between 0 and 1
dx = (x1 - x0) / N
x_grid = np.linspace(x0, x1, N + 1)
print(dx, x_grid[1] - x_grid[0])
t0 = 0
t1 = 10
M = 1000
t_grid = np.linspace(t0, t1, M + 1)
dt = (t1 - t0) / M

g = lambda x: np.exp(-100 * (x - 0.5) ** 2)
Tdx = get_Tdx(dx, N)
d = 0.002
ITdx = np.linalg.inv(np.eye(N) - d * dt / 2 * Tdx)

uold = [g(x) for x in x_grid[:-1]]
u = [uold + [uold[0]]]
uold = np.array(uold)

for i, t in enumerate(t_grid[1:]):
    if i % 50 == 0:
        print(i)
    u_new = viscous_burger_scheme(uold, d, dx, dt)
    u.append(np.concatenate((u_new, [u_new[0]])))
    uold = u_new
u = np.array(u)
view = "anim"

# Show solution in 3d plot
if view == "plot":
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x, t = np.meshgrid(x_grid, t_grid)

    ax.set_zlim(-0.3, 0.3)
    surf = ax.plot_surface(x, t, u, cmap=cm.coolwarm)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$u(t,x)$")
    ax.set_title("Lax-Wendroff method with ")

    plt.show()

elif view == "anim":
    # Animate instead!
    def animate(i):
        line.set_ydata(u[i, :])
        return (line,)

    fig, ax = plt.subplots()

    (line,) = ax.plot(x_grid, u[0, :])

    ani = animation.FuncAnimation(
        fig, animate, interval=int(1000 * (t1 - t0) / M), frames=u.shape[0]
    )
    ax.set_title("Viscous Burger equation")
    plt.show()
