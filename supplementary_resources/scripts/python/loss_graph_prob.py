import tkinter
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

def loss(D, sigma):
    numerator = np.exp(-D**2 / (2 * sigma))
    denominator = 2 * np.pi * sigma
    return -np.log(numerator / denominator)


num = 100
D_vals = np.linspace(0, 1, num)
sigma_vals = np.linspace(1 / (2 * np.pi), 1, num)
D_grid, sigma_grid = np.meshgrid(D_vals, sigma_vals)
loss_vals = loss(D_grid, sigma_grid)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(D_grid, sigma_grid, loss_vals, cmap='coolwarm', alpha = 0.7)
cntr = ax.contour3D(D_grid, sigma_grid, loss_vals, 100, cmap='coolwarm')

sigma_const = 1 / (2 * np.pi)
sigma_vals_off = np.linspace(0.001, sigma_const, num)

D_grid_off, sigma_grid_off = np.meshgrid(D_vals, sigma_vals_off)
loss_vals_off = loss(D_grid_off, sigma_grid_off)
loss_vals_off_clipped = np.clip(loss_vals_off, -1, np.max(loss_vals))

cntr_off = ax.contour3D(D_grid_off, sigma_grid_off, loss_vals_off_clipped, 100, cmap='coolwarm', alpha=0.5)


loss_wall = np.linspace(-1, np.max(loss_vals), num)
D, L = np.meshgrid(D_grid, loss_wall)

bound = ax.plot_surface(D, np.ones_like(D) * sigma_const, L, facecolor="black", alpha=0.2)

sigma_vals_full = np.linspace(0.001, 1, num)
D_grid_full, sigma_grid_full = np.meshgrid(D_vals, sigma_vals_full)
loss_zero = ax.plot_surface(D_grid_full, sigma_grid_full, loss_vals*0, facecolor="black", alpha = 0.2)


ax.set_xlabel(r'$D^2$')
ax.set_ylabel(r'$\sigma$')
ax.set_zlabel('Loss')

ax.set_title("Negative Log-Likelihood Loss")

ax.text(-0.2, sigma_const, -1, r'$\frac{1}{2\pi}$', color='red', fontsize=14, ha='center', va='center')
ax.text(-0.2, -0.2, 0, 'min', color='red', fontsize=12, ha='center', va='center')


ax.set_zlim(-1, np.max(loss_vals))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.colorbar(surf,  shrink=0.9, pad=0.1, location="left")

ax.view_init(30, 164)

plt.savefig("loss_graph.png", dpi=600)

plt.show()


