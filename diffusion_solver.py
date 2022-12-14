import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg
import matplotlib.animation as animation

# Euler step function for y'=Ay, so y'(t)=f(t,y(t)) with f(t,y(t))=A*y(t)
def euler_step(A, y_old, h):
    return y_old + h*np.matmul(A, y_old)

# Kuranku Nikurusonu
def crank_nicolson(uold, T_dx, dt, N):
    return np.linalg.solve(np.identity(N) - dt / 2 * T_dx, np.dot(np.identity(N) + dt / 2 * T_dx, uold))


def construct_T_dx(alpha, beta, L, N):
    dx = L / (N + 1)
    c = np.zeros(N)
    c[0] = -2
    c[1] = 1
    return scipy.linalg.toeplitz(c, c)/dx/dx

# Diffusion equation:
# u_xx = u_t, with homogeneous BC,
# u(t,0) = u(t,1) = 0, and IC
# u(0,t) = g(t)
#
# Discretization:
# u'=T_(dx)*u

x0 = 0
x1 = 1
N = 20 # amount of grid points on x between 0 and 1
dx = (x1-x0)/(N+1)
x_grid = np.linspace(x0, x1, N + 2)
t0 = 0
t1 = 0.1
M = int(N*t1/0.1)
t_grid = np.linspace(t0, t1, M)
dt = t_grid[1]-t_grid[0]

g = lambda x: (x-0.4) * int(0.4 <= x <= 0.7)
uold = np.array([g(x) for x in x_grid])
u = [uold]
T_dx = construct_T_dx(0,0,x1-x0,N)
for i,t in enumerate(t_grid[1:]):
    #u_new = np.concatenate(([g(x0)], euler_step(T_dx, uold[1:-1], dt), [g(x1)]))
    u_new = np.concatenate(([g(x0)], crank_nicolson(uold[1:-1], T_dx, dt, N), [g(x1)]))
    u.append(u_new)
    uold = u_new

tx_grid = np.meshgrid(t_grid,x_grid)
u = np.array(u)

view = "plot"

# Animate instead!
def animate(i):
    line.set_ydata(u[i][:])
    return line,

# Show solution in 3d plot
if view == "plot":
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x, t = np.meshgrid(x_grid, t_grid)

    ax.set_zlim(-0.3, 0.3)
    surf = ax.plot_surface(x,t, u, cmap=cm.coolwarm)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$u(t,x)$")
    ax.set_title("Crank-Nicolson method with $M=N$")
    print(dt, dx)
    plt.show()

elif view == "anim":

    fig, ax = plt.subplots()

    line, = ax.plot(x_grid, u[0][:])


    ani = animation.FuncAnimation(fig, animate, interval=int(1000 * (t1 - t0) / M), frames=u.shape[0])

    plt.show()
