import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg
import matplotlib.animation as animation

# Euler step function for y'=Ay, so y'(t)=f(t,y(t)) with f(t,y(t))=A*y(t)
def euler_step(A, y_old, h):
    return y_old + h*np.matmul(A,y_old)

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
x_grid = np.linspace(x0, x1, N)
t0 = 0
t1 = 10
M = 10000 # amount of time steps
t_grid = np.linspace(t0, t1, M)
dt = t_grid[1]-t_grid[0]

g = lambda x : (x-0.4) * int(0.4 <= x <= 0.7)
u0 = [g(x) for x in x_grid]
u = [np.array(u0)]
T_dx = construct_T_dx(0,0,x1-x0,N)
for i,t in enumerate(t_grid[1:]):
    u_new = euler_step(T_dx, u[-1], dt)
    u.append(u_new)

tx_grid = np.meshgrid(t_grid,x_grid)
u = np.array(u)

# Show solution in 3d plot
"""fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x, t = np.meshgrid(x_grid, t_grid)

ax.set_zlim(-1.01, 1.01)
surf = ax.plot_surface(x,t, u, cmap=cm.coolwarm)

plt.show()"""

fig, ax = plt.subplots()

line, = ax.plot(x_grid, u[0][:])
# Animate instead!
def animate(i):
    line.set_ydata(u[i][:])
    return line,

ani = animation.FuncAnimation(fig, animate, interval=60, frames=u.shape[0])

plt.show()
