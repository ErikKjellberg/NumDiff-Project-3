import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg
import matplotlib.animation as animation

x0 = 0
x1 = 1
N = 100 # amount of grid points on x between 0 and 1
dx = (x1-x0) / N
x_grid = np.linspace(x0, x1, N + 1)
print(dx, x_grid[1] - x_grid[0])
t0 = 0
t1 = 5
M = 1000
t_grid = np.linspace(t0, t1, M+1)
dt = (t1 - t0) / M

g = lambda x : np.exp(-100*(x-0.5)**2)
g = lambda x : int(0 < x <= 1/4)*(1+(4*x+1)*(4*x-1)) + \
int(1/4 < x <= 3/4)*(2-(4*x-2)**2) + \
int(3/4 < x <= 1)*(1+(4*x-3)*(4*x-5))

a = 1
mu = dt/dx


print(a * mu)
def get_circer(a, mu, N):
    xi = a * mu
    col = np.zeros(N)
    col[0] = 1 - xi * xi
    col[1] = xi / 2 * (1 + xi)

    col[-1] = -xi / 2 * (1 - xi)
    return scipy.linalg.circulant(col)

uold = [g(x) for x in x_grid[:-1]]
circ = get_circer(a, mu, N)
u = [uold + [uold[0]]]
uold = np.array(uold)

for i,t in enumerate(t_grid[1:]):
    u_new = np.dot(circ, uold) 
    u.append(np.concatenate((u_new, [u_new[0]])))
    uold = u_new
u = np.array(u)
view = "anim"
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

    plt.show()

elif view == "anim":

    fig, ax = plt.subplots()

    line, = ax.plot(x_grid, u[0][:])


    ani = animation.FuncAnimation(fig, animate, interval=int(1000 * (t1 - t0) / M), frames=u.shape[0])

    plt.show()
