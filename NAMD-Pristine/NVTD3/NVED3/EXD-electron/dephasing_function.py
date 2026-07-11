#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
#from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# def gaussian(t, c, tau):
#     return np.exp(-0.5 * (t / tau)**2)

def gaussian(x,c):
    result = np.exp(-x**2/(2*c**2))
    return result

def dephase(Et, dt=1.0):
    r'''
    Calculate the autocorrelation function (ACF), dephasing function, and FT of
    ACF.

    The dephasing function was calculated according to the following formula:

    G(t) = (1 / hbar**2) \int_0^{t_1} dt_1 \int_0^{t_2} dt_2 <E(t_2)E(0)>
    D(t) = exp(-G(t))

    Fourier Transform (FT) of the normalized ACF gives the phonon influence
    spectrum, also known as the phonon spectral density.

    I(\omega) \propto | FT(Ct / Ct[0]) |**2
    '''

    from scipy.fftpack import fft

    hbar = 0.6582119513926019  # eV fs

    Et = np.asarray(Et)
    Et -= np.average(Et)

    # Autocorrelation Function (ACF) of Et
    Ct = np.correlate(Et, Et, 'full')[Et.size:] / Et.size
    
    # Cumulative integration of the ACF
    Gt = cumulative_trapezoid(cumulative_trapezoid(Ct, dx=dt, initial=0), dx=dt, initial=0)
    Gt /= hbar**2
    
    # Dephasing function
    Dt = np.exp(-Gt)
    
    # FT of normalized ACF
    Iw = np.abs(fft(Ct / Ct[0]))**2

    return Ct, Dt, Gt, Iw


energy = np.loadtxt('EIGTXT')
nbasis = energy.shape[1]
matrix = np.zeros((nbasis, nbasis), dtype=float)
dt = 1.0 # fs

############################################################
# Initialize accumulators for averaging
T_all = None
Et_all = None
Ct_all = None
Dt_all = None
Gt_all = None
Dt_fit_all = None
Iw_all = None

count = 0  # To keep track of how many pairs are accumulated

############################################################
# Loop through each pair of basis states
# Loop through each pair of basis states
for ii in range(nbasis):
    for jj in range(ii):
        Et = energy[:, ii] - energy[:, jj]
        T = np.arange(Et.size-1) * dt

        # Perform dephasing calculations
        Ct, Dt, Gt, Iw = dephase(Et)

        # Fit dephasing function using Gaussian
        popt, pcov = curve_fit(gaussian, T, Dt, maxfev=10000)
        Dt_fit = gaussian(T, *popt)
        matrix[ii, jj] = popt[0]
        matrix[jj, ii] = matrix[ii, jj]

        # Accumulate data for averaging
        if T_all is None:
            # Initialize accumulators
            T_all = T
            Et_all = Et[:-1]
            Ct_all = Ct
            Dt_all = Dt
            Gt_all = Gt  # Initialize Gt_all accumulator
            Dt_fit_all = Dt_fit
            Iw_all = Iw
        else:
            # Accumulate by summing
            Et_all += Et[:-1]
            Ct_all += Ct
            Dt_all += Dt
            Gt_all += Gt  # Accumulate Gt
            Dt_fit_all += Dt_fit
            Iw_all += Iw
        
        count += 1  # Increment pair count

############################################################
# Calculate means by dividing the accumulators by the number of pairs
Et_avg = Et_all / count
Ct_avg = Ct_all / count
Dt_avg = Dt_all / count
Gt_avg = Gt_all / count  # Calculate Gt average
Dt_fit_avg = Dt_fit_all / count
Iw_avg = Iw_all / count

############################################################
np.savetxt('Et_avg.dat', np.column_stack([T_all, Et_avg]), fmt='%12.6f', header='Time_fs Et_avg_eV')
np.savetxt('Ct_avg.dat', np.column_stack([T_all, Ct_avg]), fmt='%12.6f', header='Time_fs Ct_avg')
np.savetxt('Dt_avg.dat', np.column_stack([T_all, Dt_avg]), fmt='%12.6f', header='Time_fs Dt_avg')
np.savetxt('Iw_avg.dat', np.column_stack([T_all, Iw_avg]), fmt='%12.6f', header='Time_fs Iw_avg')
np.savetxt('DEPHTIME', matrix, fmt='%10.4f')

import matplotlib.pyplot as plt

data_labels = [
    ("Et_avg.dat", "Energy Difference (ΔE)", r'$\Delta E$ (eV)', 'k'),
    ("Ct_avg.dat", "Autocorrelation Function (ACF)", "ACF", 'r'),
    ("Dt_avg.dat", "Dephasing Function", "Dephasing Function", 'b'),
    ("Iw_avg.dat", "Spectral Density", "Spectral Density", 'g'),
]

for fname, title, ylabel, color in data_labels:
    arr = np.loadtxt(fname)
    t, y = arr[:, 0], arr[:, 1]
    plt.figure(figsize=(6, 4))
    plt.plot(t, y, color=color)
    plt.xlabel("Time (fs)", fontsize=14)
    plt.xlim(0, 1000)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    outname = fname.replace('.dat', '.png')
    plt.savefig(outname, dpi=1080)
    

import numpy as np
import matplotlib.pyplot as plt

dephtime = np.loadtxt('DEPHTIME', dtype=float)
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches((4, 3))
plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.95, wspace=0.20, hspace=0.20)
ax.set_aspect(1.0)

# Plot the heatmap
img = ax.imshow(dephtime, origin='lower', interpolation='none', cmap='Blues')
cbar = plt.colorbar(img, pad=0.02)
cbar.ax.tick_params(axis='y', which='both', labelsize='medium')
label = cbar.ax.yaxis.label
label.set_rotation(270)
label.set_va('center')
label.set_ha('left')
label.set_position((-1.5, 0.5))
# cbar.ax.text(1.15, 1.0, fontsize='medium', ha='left', va='top', transform=cbar.ax.transAxes)
cbar.set_label('Dephasing time (fs)', fontsize=14, rotation=270, labelpad=9)

# Add annotations
for i in range(dephtime.shape[0]):
    for j in range(dephtime.shape[1]):
        text_color = "white" if dephtime[i, j] > (dephtime.max() / 2) else "black"
        ax.text(j, i, f"{dephtime[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=12)
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
plt.savefig('deph.png', dpi=1080)
