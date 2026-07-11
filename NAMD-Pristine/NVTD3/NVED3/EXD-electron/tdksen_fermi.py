#!/usr/bin/env python3
############################################################
import os, re
import numpy as np
from glob import glob

import matplotlib as mpl
mpl.use('agg')
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
############################################################
def WeightFromPro(infile='PROCAR', whichAtom=None, spd=None):
    """
    Contribution of selected atoms to the each KS orbital
    """
    print(f"Reading file: {infile}")
    assert os.path.isfile(infile), f'{infile} cannot be found!'

    with open(infile, 'r') as f:
        FileContents = [line for line in f.readlines() if line.strip()]

    nkpts, nbands, nions = [int(xx) for xx in re.sub('[^0-9]', ' ', FileContents[1]).split()]

    if spd:
        Weights = np.asarray([line.split()[1:-1] for line in FileContents
                              if not re.search('[a-zA-Z]', line)], dtype=float)
        Weights = np.sum(Weights[:, spd], axis=1)
    else:
        Weights = np.asarray([line.split()[-1] for line in FileContents
                              if not re.search('[a-zA-Z]', line)], dtype=float)

    nspin = Weights.shape[0] // (nkpts * nbands * nions)
    Weights.resize(nspin, nkpts, nbands, nions)

    Energies = np.asarray([line.split()[-4] for line in FileContents
                            if 'occ.' in line], dtype=float)
    Energies.resize(nspin, nkpts, nbands)

    if whichAtom is None:
        return Energies, np.sum(Weights, axis=-1)
    else:
        return Energies, np.sum(Weights[:, :, :, whichAtom], axis=-1) 
def parallel_wht(runDirs, whichAtoms, nproc=None):
    '''
    calculate localization of some designated in parallel.
    '''
    import multiprocessing
    import numpy as np

    nproc = multiprocessing.cpu_count() if nproc is None else nproc
    pool = multiprocessing.Pool(processes=nproc)

    results = []
    for rd in runDirs:
        res = pool.apply_async(WeightFromPro, (rd + '/PROCAR', whichAtoms, None,))
        results.append(res)

    enr = []
    wht = []
    for ii in range(len(results)):
        tmp_enr, tmp_wht = results[ii].get()
        enr.append(tmp_enr)
        wht.append(tmp_wht)
        
    print("enr:", np.array(enr))
    print("wht:", np.array(wht))

    return np.array(enr), np.array(wht)

############################################################
# calculate spatial localization
############################################################
nsw     = 2000
dt      = 1.0
nproc   = 32
prefix  = '../run/'
runDirs = [prefix + '/{:04d}'.format(ii + 1) for ii in range(nsw)]
# which spin, index starting from 0
whichS  = 0
# which k-point, index starting from 0
whichK  = 0 
# which atoms, index starting from 0

whichA = np.arange(27)      # Indices 0 to 26 for the first 27 atoms (Compound B)
#whichB = range(27, 63)      # Indices 27 to 62 for the next 36 atoms (Compound A)

Alabel  = r'Zn$_2$S$_2$'
Blabel  = r'CdCl$_2$'

# whichA = np.arange(27)      # Indices 0 to 26 for the first 27 atoms (Compound A)
# #whichB = range(27, 44)      # Indices 27 to 44 for the next 18 atoms (Compound B)

# # whichB = np.arange(84)
# # whichB = np.arange(84, 124)

# Alabel  = 'WSeTe'
# Blabel  = 'BAs'

E_F = -1.7068

if os.path.isfile('all_wht.npy'):
    Wht = np.load('all_wht.npy')
    Enr = np.load('all_en.npy')
    # Shift energy levels relative to Fermi energy
    Enr_shifted = Enr - E_F

    # Identify VBM and CBM correctly
    VBM = np.max(Enr_shifted[Enr_shifted < 0])  # Highest occupied state
    CBM = np.min(Enr_shifted[Enr_shifted > 0])  # Lowest unoccupied state

    # Print updated energy values
    # print(f"VBM: {VBM:.4f} eV, CBM: {CBM:.4f} eV, Fermi Level (E_F): 0.0000 eV")

else:
    # for gamma point version, no-spin
    Enr, Wht = parallel_wht(runDirs, whichA, nproc=nproc)
    Enr = Enr[:, whichS,whichK, :]
    Wht = Wht[:, whichS,whichK, :]
    Enr_shifted = Enr - E_F
    # Enr, Wht1 = parallel_wht(runDirs, whichA, nproc=nproc)
    # Enr, Wht2 = parallel_wht(runDirs, whichB, nproc=nproc)
    # Enr = Enr[:, whichS,whichK, :]
    # Wht1 = Wht1[:, whichS,whichK, :]
    # Wht2 = Wht2[:, whichS,whichK, :]
    # Wht = Wht1 / (Wht1 + Wht2)

    np.save('all_wht.npy', Wht)
    np.save('all_en.npy', Enr)

############################################################
fig = plt.figure()
fig.set_size_inches(6, 3.5)

########################################
ax      = plt.subplot()
nband   = Enr.shape[1]
T, dump = np.mgrid[0:nsw:dt, 0:nband]
sFac    = 8

############################################################
# METHOD 1.
############################################################
# use scatter to plot the band 
# ax.scatter( T, Enr, s=Wht / Wht.max() * sFac, color='red', lw=0.0, zorder=1)
# for ib in range(nband):
#     ax.plot(T[:,ib], Enr[:,ib], lw=0.5, color='k', alpha=0.5)

############################################################
# METHOD 2.
############################################################
# use colored scatter to plot the band 

from matplotlib.colors import LinearSegmentedColormap

# Define RGB colors for min and max values
# Replace these RGB tuples with your desired colors
# min_color = (85/255, 181/255, 183/255)  # Min (W)
# max_color = (251/255, 111/255, 111/255)  # Max (B)
min_color = (124/255, 129/255, 175/255)  # Min (Zn) 124, 129, 175
max_color = (179/255, 255/255, 173/255)  # Max (Cl) 179, 255, 173
# Create a custom colormap using these two colors
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', [min_color, max_color])

# Use this custom colormap in the scatter plot
img = ax.scatter(T, Enr_shifted, s=1.0, c=Wht, lw=0.0, zorder=1,
                 vmin=Wht.min(),
                 vmax=Wht.max(),
                 cmap=custom_cmap)

#img = ax.scatter(T, Enr, s=1.0, c=Wht, lw=0.0, zorder=1,
#                 vmin=Wht.min(),
#                 vmax=Wht.max(),
#                 cmap='viridis')
#for ib in range(nband):
#    ax.plot(T[:,ib], Enr[:,ib], lw=0.5, color='k', alpha=0.5)

divider = make_axes_locatable(ax)
ax_cbar = divider.append_axes('right', size='5%', pad=0.02)
cbar = plt.colorbar(img, cax=ax_cbar,
                    orientation='vertical')
cbar.set_ticks([Wht.min(), Wht.max()])
cbar.set_ticklabels([Alabel, Blabel])

############################################################
# METHOD 3.
############################################################
# # use color strip to plot the band
#
# LW    = 1.0
# DELTA = 0.3
# norm  = mpl.colors.Normalize(vmin=Wht.min(),
#                              vmax=Wht.max())
# # create a ScalarMappable and initialize a data structure
# s_m   = mpl.cm.ScalarMappable(cmap='jet_r', norm=norm)
# s_m.set_array([Wht])
#
# x     = np.arange(0, nsw, dt)
# # for iband in range(nband):
# for iband in range(100, 110):
#     print('Processing band: {:4d}...'.format(iband))
#     y = Enr[:,iband]
#     z = Wht[:,iband]
#
#     ax.plot(x, y,
#             lw=LW + 2 * DELTA,
#             color='gray', zorder=1)
#
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments,
#                         colors=[s_m.to_rgba(ww) for ww in (z[1:] + z[:-1])/2.]
#                         )
#     # lc.set_array((z[1:] + z[:-1]) / 2)
#     lc.set_linewidth(LW)
#     ax.add_collection(lc)
#
#     divider = make_axes_locatable(ax)
#     ax_cbar = divider.append_axes('right', size='5%', pad=0.02)
#     cbar = plt.colorbar(s_m, cax=ax_cbar,
#                         # ticks=[Wht.min(), Wht.max()],
#                         orientation='vertical')
#     cbar.set_ticks([Wht.min(), Wht.max()])
#     cbar.set_ticklabels([Alabel, Blabel])

ax.set_xlim(0, 2000)
ax.set_ylim(-4.0, 4.0)


# Set the font as Helvetica
# mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['font.family'] = 'Candara'
ax.set_xlabel('Time (fs)',   fontsize=16, labelpad=5)
ax.set_ylabel('Energy (eV)', fontsize=16, labelpad=8)
# fontsize=14 or None
ax.tick_params(which='both', labelsize='large', direction='in')

########################################
plt.tight_layout(pad=0.2)
plt.savefig('ksen_fermi.png', dpi=360)

