'''
    This code is used to replot Figure 1 of the main text.
'''
# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import os, sys

# some additional packages that are needed
from scipy import io, signal
from scipy import ndimage
from multiprocessing import Pool

# import plotting-related packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# import external dictionaries
os.chdir("../src")
from preprocess import *
from relevance import *

# define function to follow curve in the H[K]-H[s] space
def follow_total_relevance(zipped_data):
    total_time, spikes, bin_time = zipped_data
    N_max = np.round(np.log(total_time - (0.01*total_time))/np.log(10), 2);
    N_parts = np.unique(np.logspace(0.4,N_max,100).astype("int"))
    N_partitions = np.append(N_parts,[spikes.size])
    input_data = [(total_time, spikes, i) for i in N_parts]
    pool = Pool()
    res = pool.map_async(follow_curve,input_data)
    pool.close(); pool.join()
    data = np.array(res.get())
    data = np.append(data,np.array([[0.0, 0.0], [0.0, 1.0]]),axis=0)
    return data[np.lexsort((data[:,0],data[:,1]))]

# Load spike train data
filenames = [fname.rstrip('\n') for fname in open(os.path.join('../Flekken_Data','cell_filenames'))]
spike_times = [io.loadmat(os.path.join('../Flekken_Data',filenames[i]), squeeze_me=True)['cellTS'] for i in np.arange(len(filenames))]

# Binarize spike train data
binning_time = 10e-3
unfiltered_spike_trains, time_bins = binning(binning_time,spike_times,True)

# Load the positions and calculate speeds
pos = io.loadmat(os.path.join('../Flekken_Data','BEN/BEN_pos.mat'), squeeze_me=True)
positions = np.array([pos['post'],(pos['posx']+pos['posx2'])/2.,(pos['posy']+pos['posy2'])/2.])

# Convert the coordinates into actual spatial coordinates
tight_range = ((-74.5, 74.5), (-74.5, 74.5))
positions[1], positions[2], info = transform(positions[1],positions[2],range_=tight_range,translate=True,rotate=True)

rat_speed = calculate_speed(positions[0], positions[1], positions[2], 0.0) # Calculate speed
x_t = np.interp(time_bins, positions[0], positions[1]); y_t = np.interp(time_bins, positions[0], positions[2]) # Interpolate the positions of the LEDs
rat_speed = np.interp(time_bins, positions[0], rat_speed) # Interpolate the speeds of the rat

# Filter for speed
min_speed = 5.0; speed_filter = np.where(rat_speed[:-1]<min_speed); speed_mask = rat_speed[:-1]<min_speed
# Set to zero the spike patterns in which the rat is moving less than the minimum speed
spike_trains = np.copy(unfiltered_spike_trains)
for i in np.arange(len(spike_trains)): spike_trains[i][speed_filter] = 0
# Mask time points when the rat is moving less than the minimum speed
x_t = np.copy(x_t[:-1]); y_t = np.copy(y_t[:-1]); time_step = np.copy(time_bins[:-1])

# follow the curves for the grid cell and interneuron of interest
data_gridcell = follow_total_relevance((time_step.size, unfiltered_spike_trains[6], binning_time))
data_interneuron = follow_total_relevance((time_step.size, unfiltered_spike_trains[7], binning_time))

# plot Figure 1
fig = plt.figure(dpi=300)
fig.set_size_inches(12,6)
gs = gridspec.GridSpec(2, 3)

axHofK = plt.subplot(gs[0:,1:3])
axHofK.plot(data_gridcell[:,1], data_gridcell[:,0], "go-", alpha=0.75, label=r"Grid Cell 7 (T02C01)")
axHofK.plot(data_interneuron[:,1], data_interneuron[:,0], "ro-", alpha=0.75, label=r"Interneuron 8 (T02C02)")
axHofK.set_ylabel(r'$H[K]/\log M$ (Mats)')
axHofK.set_xlabel(r'$H[\underline{s}]/\log M$ (Mats)')
axHofK.patch.set_facecolor('white')
axHofK.spines['bottom'].set_color("black"); axHofK.spines['bottom'].set_linewidth(0.5)
axHofK.spines['left'].set_color("black"); axHofK.spines['left'].set_linewidth(0.5)
axHofK.set_xlim(left=-0.01)
axHofK.set_ylim(bottom=-0.008, top=0.5)
axHofK.legend(loc='upper left')
axHofK.fill_between(data_gridcell[:,1], 0, data_gridcell[:,0],color="g",alpha=0.25)
axHofK.fill_between(data_interneuron[:,1], 0, data_interneuron[:,0],color="r",alpha=0.25)
axHofK.text(-0.1, 1.065, "C", transform=axHofK.transAxes, fontsize=18, fontweight='bold', va='top')

axGridMap = plt.subplot(gs[0,0])
N_bins = 200; neuron_index = 6
occupational_probability, xedges, yedges = np.histogram2d(x_t, y_t, bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))
spike_map = np.ma.array(np.histogram2d(np.repeat(x_t,spike_trains[neuron_index]),
                                       np.repeat(y_t,spike_trains[neuron_index]),
                                       bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))[0])/np.ma.array(occupational_probability)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
blur = ndimage.filters.gaussian_filter(spike_map, 8.0, mode="reflect", truncate = 4.0)
heatmap = axGridMap.imshow(100.*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet)
plt.colorbar(heatmap)
axGridMap.grid(b=False, which='major')
axGridMap.grid(b=False, which='minor')
axGridMap.set_title(r"Grid Cell 7 (T02C01)")
axGridMap.text(-0.2, 1.15, "A", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')

axInterMap = plt.subplot(gs[1,0])
N_bins = 200; neuron_index = 7
occupational_probability, xedges, yedges = np.histogram2d(x_t, y_t, bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))
spike_map = np.ma.array(np.histogram2d(np.repeat(x_t,spike_trains[neuron_index]),
                                       np.repeat(y_t,spike_trains[neuron_index]),
                                       bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))[0])/np.ma.array(occupational_probability)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
blur = ndimage.filters.gaussian_filter(spike_map, 8.0, mode="reflect", truncate = 4.0)
heatmap = axInterMap.imshow(100.*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet)
plt.colorbar(heatmap)
axInterMap.grid(b=False, which='major')
axInterMap.grid(b=False, which='minor')
axInterMap.set_title(r"Interneuron 8 (T02C02)")
axInterMap.text(-0.2, 1.15, "B", transform=axInterMap.transAxes, fontsize=18, fontweight='bold', va='top')
gs.update(wspace=0.3, hspace=0.4)

plt.savefig("Figures/Figure1.pdf", bbox_inches="tight", dpi=600)
