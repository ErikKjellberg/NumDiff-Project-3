import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from matplotlib import cm

from util import RMS_norm, get_Sdx, get_Tdx, get_trap

x0 = 0
x1 = 1
N = 900  # amount of grid points on x between 0 and 1
dx = (x1 - x0) / N
x_grid = np.linspace(x0, x1, N + 1)
print(dx, x_grid[1] - x_grid[0])
t0 = 0
t1 = 10
M = 5000
t_grid = np.linspace(t0, t1, M + 1)
dt = (t1 - t0) / M

g = lambda x: np.exp(-100 * (x - 0.5) ** 2)
f = (
    lambda x: int(0 < x <= 1 / 4) * (1 + (4 * x + 1) * (4 * x - 1))
    + int(1 / 4 < x <= 3 / 4) * (2 - (4 * x - 2) ** 2)
    + int(3 / 4 < x <= 1) * (1 + (4 * x - 3) * (4 * x - 5))
)

a = 1
d = 0.1
Pe = abs(a / d)
# High Pe -> Convection dominated, and so on and so on.
mu = dt / dx
print(f"Peclet number {Pe}")

print(get_Sdx(1, N))
print(get_Tdx(1, N))
A = get_trap(d * get_Tdx(dx, N) - a * get_Sdx(dx, N), dt)

uold = [g(x) for x in x_grid[:-1]]
u = [uold + [uold[0]]]
uold = np.array(uold)

for i, t in enumerate(t_grid[1:]):
    u_new = np.dot(A, uold)
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

    ax.set_title(f"Lax-Wendroff method with $a \Delta t / \Delta x = {a * mu}$")
    plt.show()

elif view == "RMS":
    u_RMS = np.array([RMS_norm(u[i, :], dx) for i in range(t_grid.shape[0])])
    plt.title("RMS norm of solution for $a \Delta t / \Delta x = 1$")
    plt.plot(t_grid, u_RMS)
    plt.show()
