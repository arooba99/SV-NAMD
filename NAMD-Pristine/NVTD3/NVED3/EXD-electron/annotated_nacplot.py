#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, subprocess
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call

# Load data
nactxt = np.loadtxt('NATXT', dtype=float)
eigtxt = np.loadtxt('EIGTXT', dtype=float)

nbasis = eigtxt.shape[1]
nactxt = nactxt.reshape((-1, nbasis, nbasis))

hbar = 0.6582119281559802
potim = 1.0

# NAC in unit of meV
a_nac = np.average(np.abs(nactxt), axis=0) * hbar / (2 * potim) * 1000
a_eig = np.average(eigtxt, axis=0)

# Save `a_nac` to .dat files
np.savetxt('nac_data.dat', a_nac, fmt='%.6f', delimiter='\t')

# Plotting and saving the image
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches((4, 3))
plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.95, wspace=0.20, hspace=0.20)
ax.set_aspect(1.0)

# Plot the heatmap
img = ax.imshow(a_nac, origin='lower', interpolation='none', cmap='Blues')
cbar = plt.colorbar(img, pad=0.02)
cbar.ax.tick_params(axis='y', which='both', labelsize='medium')
label = cbar.ax.yaxis.label
label.set_rotation(270)
label.set_va('center')
label.set_ha('left')
label.set_position((-1.5, 0.5))
# cbar.ax.text(1.15, 1.0, fontsize='medium', ha='left', va='top', transform=cbar.ax.transAxes)
cbar.set_label('NAC (meV)', fontsize=14, labelpad=7)

# Add annotations
for i in range(a_nac.shape[0]):
    for j in range(a_nac.shape[1]):
        text_color = "white" if a_nac[i, j] > (a_nac.max() / 2) else "black"
        ax.text(j, i, f"{a_nac[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=10)
        # ax.text(j, i, None)

# ax.set_xticks([])
# ax.set_yticks([])

labels = ['CBM', 'CBM+1', 'CBM+2']
positions = np.arange(len(labels))

ax.set_xticks(positions)
ax.set_xticklabels(labels)

ax.set_yticks(positions)
ax.set_yticklabels(labels)

# plt.tight_layout(pad=0.2)
plt.tight_layout()
plt.savefig('nac.png', dpi=1080)
