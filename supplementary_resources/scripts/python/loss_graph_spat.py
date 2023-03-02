import tkinter
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

def loss(x, y):
    return np.power(x, 2) + np.power(y, 2)

num = 100

X_vals = np.linspace(0, 140, num)
Y_vals = np.linspace(0, 120, num)

X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)

loss_vals = loss(X_grid, Y_grid)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X_grid, Y_grid, loss_vals, cmap='coolwarm', alpha = 0.7)
cntr = ax.contour3D(X_grid, Y_grid, loss_vals, 100, cmap='coolwarm')

loss_zero = ax.plot_surface(X_grid, Y_grid, loss_vals*0, facecolor="black", alpha = 0.2)
loss_zero = ax.plot_surface(X_grid, Y_grid, loss_vals*0 + 15000, facecolor="black", alpha = 0.2)


ax.set_xlabel(r'$\Delta Y_{lon}$')
ax.set_ylabel(r'$\Delta Y_{lat}$')
ax.set_zlabel('Loss')

ax.set_title("Squared Euclidean Distance")

ax.set_zlim(-1, np.max(loss_vals))
ax.set_xlim(0, 140)
ax.set_ylim(0, 120)

fig.colorbar(surf,  shrink=0.9, pad=0.1, location="left")

ax.view_init(30, 164)

plt.savefig("loss_graph_spat.png", dpi=600)

plt.show()


