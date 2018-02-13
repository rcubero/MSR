'''
    This code is used to replot Figure 2 of the main text.
'''
# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import os, sys

# some additional packages that are needed
from scipy import io, signal
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.stats import pearsonr, spearmanr
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
from spatial_quantities import *
from HD_quantities import *

# Load spike train data
filenames = [fname.rstrip('\n') for fname in open(os.path.join('../Flekken_Data','cell_filenames'))]
spike_times = [io.loadmat(os.path.join('../Flekken_Data',filenames[i]), squeeze_me=True)['cellTS'] for i in np.arange(len(filenames))]

# Binarize spike train data
binning_time = 10e-3
unfiltered_spike_trains, time_bins = binning(binning_time,spike_times,True)

# Load the positions and calculate speeds
pos = io.loadmat(os.path.join('../Flekken_Data','BEN/BEN_pos.mat'), squeeze_me=True)
positions = np.array([pos['post'],(pos['posx']+pos['posx2'])/2.,(pos['posy']+pos['posy2'])/2.])
positions_r = np.array([pos['post'],pos['posx'],pos['posy']])
positions_g = np.array([pos['post'],pos['posx2'],pos['posy2']])

# Load cell names
cellnames = np.loadtxt("../Flekken_Data/cell_names",dtype=bytes,delimiter='\n').astype(str)

# Convert the coordinates into actual spatial coordinates
tight_range = ((-74.5, 74.5), (-74.5, 74.5))
positions[1], positions[2], info = transform(positions[1],positions[2],range_=tight_range,translate=True,rotate=True)
positions_r[1], positions_r[2], info = transform(positions_r[1],positions_r[2],range_=tight_range,translate=True,rotate=True)
positions_g[1], positions_g[2], info = transform(positions_g[1],positions_g[2],range_=tight_range,translate=True,rotate=True)

rat_speed = calculate_speed(positions[0], positions[1], positions[2], 0.0) # Calculate speed
x_t = np.interp(time_bins, positions[0], positions[1]); y_t = np.interp(time_bins, positions[0], positions[2]) # Interpolate the midpoint positions of the LEDs
x_r = np.interp(time_bins, positions_r[0], positions_r[1]); y_r = np.interp(time_bins, positions_r[0], positions_r[2]) # Interpolate the locations of the red LED
x_g = np.interp(time_bins, positions_g[0], positions_g[1]); y_g = np.interp(time_bins, positions_g[0], positions_g[2]) # Interpolate the locations of the green LED
rat_speed = np.interp(time_bins, positions[0], rat_speed) # Interpolate the speeds of the rat
movementAngle = calculate_movement_direction(x_t, y_t); head_direction = calculate_head_direction(x_r, x_g, y_r, y_g) # calculate movement direction and head direction

# Filter for speed
min_speed = 5.0; speed_filter = np.where(rat_speed[:-1]<min_speed); speed_mask = rat_speed[:-1]<min_speed
# Set to zero the spike patterns in which the rat is moving less than the minimum speed
spike_trains = np.copy(unfiltered_spike_trains)
for i in np.arange(len(spike_trains)): spike_trains[i][speed_filter] = 0
# Mask time points when the rat is moving less than the minimum speed
unmasked_x_t = np.copy(x_t[:-1]); unmasked_y_t = np.copy(y_t[:-1])
x_t = np.ma.array(x_t[:-1], mask=speed_mask); y_t = np.ma.array(y_t[:-1], mask=speed_mask)
time_step = np.ma.array(time_bins[:-1], mask=speed_mask)
movementAngle = movementAngle[:-1]; head_direction = head_direction[:-1]

# load neuron index
grid = [7,9,11,13,15,17,19,20,23,24,25,27,28,33,36,37,39,40,41,42,52,60,61,62,63,64,65]
interneuron = [8,12,16,22,50]

interneuron_index = np.array(interneuron)-1
grid_index = np.array(grid)-1
bordercell_index = np.array([43])
other_neuron = np.delete(np.arange(65),np.sort(np.append(interneuron_index,np.append(grid_index,bordercell_index))))

# calculate the multi-scale relevance for each neuron
try:
    unfiltered_relevance = np.loadtxt("Data_Output/unfiltered_relevance.d")
except:
    unfiltered_relevance = np.zeros(spike_trains.shape[0])
    for i in np.arange(spike_trains.shape[0]):
        unfiltered_relevance[i] = parallelized_total_relevance((time_step.size, unfiltered_spike_trains[i]))
    np.savetxt("Data_Output/unfiltered_relevance.d", unfiltered_relevance)

N_neurons = 65
try:
    spatial_info = np.loadtxt("Data_Output/spatial_info.d")
    spatial_sp = np.loadtxt("Data_Output/spatial_sp.d")
    spatial_mean = np.loadtxt("Data_Output/spatial_meanspike.d")
    randomized_spatial_info, randomized_std_spatial_info = np.loadtxt("Data_Output/randomized_spatial_info.d")
    spatial_gridscore = np.loadtxt("Data_Output/grid_score.d")

    HD_tuningcurve = HD_tuningcurves(N_neurons, unfiltered_spike_trains, head_direction, binning_time)
    smoothened_firingrates = HD_tuningcurves(N_neurons, unfiltered_spike_trains, head_direction, binning_time, N_bins=360)
    smoothened_firingrates = np.array([gaussian_filter1d(smoothened_firingrates[i], np.sqrt(20), truncate=4.0, mode='wrap') for i in np.arange(N_neurons)])
    HD_info = np.loadtxt("Data_Output/HD_info.d")
    HD_sp = np.loadtxt("Data_Output/HD_sp.d")
    HD_meanvectorlength = np.loadtxt("Data_Output/HD_meanvectorlength.d")
    randomized_HD_info, randomized_std_HD_info = np.loadtxt("Data_Output/randomized_HD_info.d")

except:
    # calculate spatial quantities
    spatial_info = spatial_information(N_neurons, spike_trains, unmasked_x_t, unmasked_y_t, binning_time, output_name="Data_Output/spatial_info.d")
    spatial_sp = spatial_sparsity(N_neurons, spike_trains, unmasked_x_t, unmasked_y_t, binning_time, output_name="Data_Output/spatial_sp.d")
    spatial_mean = spatial_meanspike(N_neurons, spike_trains, unmasked_x_t, unmasked_y_t, binning_time, output_name="Data_Output/spatial_meanspike.d")
    randomized_spatial_info, randomized_std_spatial_info = randomized_spatial_information(N_neurons, spike_trains, unmasked_x_t, unmasked_y_t, binning_time, output_name="Data_Output/randomized_spatial_info.d")
    spatial_gridscore = np.loadtxt("Data_Output/grid_score.d") # pre-calculated grid scores taken from Dunn (2017)

    # calculate head directional quantities
    HD_tuningcurve = HD_tuningcurves(N_neurons, unfiltered_spike_trains, head_direction, binning_time)
    smoothened_firingrates = HD_tuningcurves(N_neurons, unfiltered_spike_trains, head_direction, binning_time, N_bins=360)
    smoothened_firingrates = np.array([gaussian_filter1d(smoothened_firingrates[i], np.sqrt(20), truncate=4.0, mode='wrap') for i in np.arange(N_neurons)])
    HD_info = HD_information(N_neurons, unfiltered_spike_trains, head_direction, binning_time, output_name="Data_Output/HD_info.d")
    HD_sp = HD_sparsity(N_neurons, unfiltered_spike_trains, head_direction, binning_time, output_name="Data_Output/HD_sp.d")
    HD_meanvectorlength = mean_vector_length(N_neurons, unfiltered_spike_trains, head_direction, binning_time, output_name="Data_Output/HD_meanvectorlength.d")
    randomized_HD_info, randomized_std_HD_info = randomized_HD_information(N_neurons, unfiltered_spike_trains, head_direction, binning_time, output_name="Data_Output/randomized_HD_info.d")

# plot Figure 2
fig = plt.figure(dpi=300)
fig.set_size_inches(20,32)
gs0 = gridspec.GridSpec(4, 1, hspace=0.2)

gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
axScatterPlot = plt.subplot(gs[0,:])
x_data = np.copy(unfiltered_relevance)
y_data = np.copy((spatial_info - randomized_spatial_info))
s_data = np.copy(spatial_sp)
n_data = np.array(["%d"%(u+1) for u in np.arange(N_neurons)]).astype("str")

red_scatter = axScatterPlot.scatter(x_data[grid_index], y_data[grid_index], s=600*s_data[grid_index], c="green", marker="o", alpha=0.8)
blue_scatter = axScatterPlot.scatter(x_data[interneuron_index], y_data[interneuron_index], s=600*s_data[interneuron_index], c="red", marker="s", alpha=0.8)
purple_scatter = axScatterPlot.scatter(x_data[bordercell_index], y_data[bordercell_index], s=600*s_data[bordercell_index], c="purple", marker="*", alpha=0.8)
grey_scatter = axScatterPlot.scatter(x_data[other_neuron], y_data[other_neuron], s=600*s_data[other_neuron], c="dimgrey", marker="^", alpha=1.0)
for u in np.arange(len(x_data)):
    axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.001),fontsize=12)
r_pearson, p_pearson = pearsonr(x_data, y_data)
r_spearman, p_spearman = spearmanr(x_data, y_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.26,0.8),fontsize=18)

red_scatter = axScatterPlot.plot([], [], c="green", marker="o", alpha=0.8, label="grid cells");
blue_scatter = axScatterPlot.plot([], [], c="red", marker="s", alpha=0.8, label="interneurons");
purple_scatter = axScatterPlot.plot([], [], c="purple", marker="*", alpha=0.8, label="border cell");
grey_scatter = axScatterPlot.plot([], [], c="dimgrey", marker="^", alpha=1.0, label="unclassified neurons")

axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'spatial information, $I(s, \textrm{\textbf{x}})$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.text(-0.025, 1.05, "A", transform=axScatterPlot.transAxes, fontsize=18, fontweight='bold', va='top')
axScatterPlot.tick_params(labelsize=14)

gs = gridspec.GridSpecFromSubplotSpec(2, 10, subplot_spec=gs0[1], hspace=0.2)
Rank = iter(np.arange(10))
for neuron_index in np.argsort(-x_data)[0:10]:
    r_neuron = next(Rank)
    if neuron_index+1 in interneuron: neuron_name = str("Interneuron ")+str(neuron_index + 1)
    elif neuron_index+1 in grid: neuron_name = str("Grid Cell ")+str(neuron_index + 1)
    elif neuron_index+1 in [44]: neuron_name = str("Border Cell ")+str(neuron_index + 1)
    else: neuron_name = str("Neuron ")+str(neuron_index + 1)

    axGridMap = plt.subplot(gs[0,r_neuron])
    N_bins = 200
    occupational_probability, xedges, yedges = np.histogram2d(x_t, y_t, bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))
    spike_map = np.ma.array(np.histogram2d(np.repeat(x_t,spike_trains[neuron_index]), np.repeat(y_t,spike_trains[neuron_index]),bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))[0])/np.ma.array(occupational_probability)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    blur = gaussian_filter(spike_map, 8.0, mode="reflect", truncate = 4.0)
    heatmap = axGridMap.imshow(100.0*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet)
    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    axGridMap.grid(b=False, which='major')
    axGridMap.grid(b=False, which='minor')
    axGridMap.set_title("%s \n (%s) \n $sp_{\\textrm{\\textbf{x}}}$ = %.4f \n $\overline{\lambda}$ = %.2f Hz"%(neuron_name, cellnames[neuron_index], spatial_sp[neuron_index], np.sum(unfiltered_spike_trains[neuron_index])/(binning_time*len(unfiltered_spike_trains[neuron_index]))), fontsize=16)
    if r_neuron == 0: axGridMap.text(-0.2, 1.15, "B", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')
    axGridMap.tick_params(labelsize=12)

Rank = iter(np.arange(10))
for neuron_index in np.argsort(x_data)[0:10]:
    r_neuron = next(Rank)
    if neuron_index+1 in interneuron: neuron_name = str("Interneuron ")+str(neuron_index + 1)
    elif neuron_index+1 in grid: neuron_name = str("Grid Cell ")+str(neuron_index + 1)
    elif neuron_index+1 in [44]: neuron_name = str("Border Cell ")+str(neuron_index + 1)
    else: neuron_name = str("Neuron ")+str(neuron_index + 1)

    axGridMap = plt.subplot(gs[1,r_neuron])
    N_bins = 200
    occupational_probability, xedges, yedges = np.histogram2d(x_t, y_t, bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))
    spike_map = np.ma.array(np.histogram2d(np.repeat(x_t,spike_trains[neuron_index]), np.repeat(y_t,spike_trains[neuron_index]),bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))[0])/np.ma.array(occupational_probability)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    blur = gaussian_filter(spike_map, 8.0, mode="reflect", truncate = 4.0)
    heatmap = axGridMap.imshow(100.0*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet)
    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    axGridMap.grid(b=False, which='major')
    axGridMap.grid(b=False, which='minor')
    axGridMap.set_title("%s \n (%s) \n $sp_{\\textrm{\\textbf{x}}}$ = %.4f \n $\overline{\\lambda}$ = %.2f Hz"%(neuron_name, cellnames[neuron_index], spatial_sp[neuron_index], np.sum(unfiltered_spike_trains[neuron_index])/(binning_time*len(unfiltered_spike_trains[neuron_index]))), fontsize=16)
    if r_neuron == 0: axGridMap.text(-0.2, 1.15, "C", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')
    axGridMap.tick_params(labelsize=12)

gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[2], hspace=0.01)
axScatterPlot = plt.subplot(gs[0,:])
x_data = np.copy(unfiltered_relevance)
y_data = np.copy((HD_info - randomized_HD_info))
s_data = np.copy(HD_meanvectorlength)
n_data = np.array(["%d"%(u+1) for u in np.arange(N_neurons)]).astype("str")

red_scatter = axScatterPlot.scatter(x_data[grid_index], y_data[grid_index], s=250*s_data[grid_index], c="green", marker="o", alpha=0.8)
blue_scatter = axScatterPlot.scatter(x_data[interneuron_index], y_data[interneuron_index], s=250*s_data[interneuron_index], c="red", marker="s", alpha=0.8)
purple_scatter = axScatterPlot.scatter(x_data[bordercell_index], y_data[bordercell_index], s=250*s_data[bordercell_index], c="purple", marker="*", alpha=0.8)
grey_scatter = axScatterPlot.scatter(x_data[other_neuron], y_data[other_neuron], s=250*s_data[other_neuron], c="dimgrey", marker="^", alpha=1.0)
for u in np.arange(len(x_data)):
    axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.001),fontsize=12)
r_pearson, p_pearson = pearsonr(x_data, y_data)
r_spearman, p_spearman = spearmanr(x_data, y_data)
axScatterPlot.annotate('$\\rho_p=$ %.3f, $P=$ %.2e \n $\\rho_s=$ %.3f, $P=$ %.2e'%(r_pearson, p_pearson, r_spearman, p_spearman),(0.261,0.045),fontsize=18)

red_scatter = axScatterPlot.plot([], [], c="green", marker="o", alpha=0.8, label="grid cells");
blue_scatter = axScatterPlot.plot([], [], c="red", marker="s", alpha=0.8, label="interneurons");
purple_scatter = axScatterPlot.plot([], [], c="purple", marker="*", alpha=0.8, label="border cell");
grey_scatter = axScatterPlot.plot([], [], c="dimgrey", marker="^", alpha=1.0, label="unclassified neurons")

axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional information, $I(s, \theta)$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.text(-0.025, 1.05, "D", transform=axScatterPlot.transAxes, fontsize=18, fontweight='bold', va='top')
axScatterPlot.tick_params(labelsize=14)

gs = gridspec.GridSpecFromSubplotSpec(2, 10, subplot_spec=gs0[3], hspace=0.1)
Rank = iter(np.arange(10))
for neuron_index in np.argsort(-unfiltered_relevance)[0:10]:
    r_neuron = next(Rank)
    if neuron_index+1 in interneuron: neuron_name = str("Interneuron ")+str(neuron_index + 1)
    elif neuron_index+1 in grid: neuron_name = str("Grid Cell ")+str(neuron_index + 1)
    elif neuron_index+1 in [44]: neuron_name = str("Border Cell ")+str(neuron_index + 1)
    else: neuron_name = str("Neuron ")+str(neuron_index + 1)

    axGridMap = plt.subplot(gs[0,r_neuron],projection='polar')
    axGridMap.plot(np.linspace(0, 2*np.pi, 40), HD_tuningcurve[neuron_index], 'o')
    axGridMap.plot(np.linspace(0, 2*np.pi, 360), smoothened_firingrates[neuron_index], '-')
    axGridMap.set_theta_zero_location("N")
    axGridMap.patch.set_facecolor("white")
    axGridMap.grid(True,color="k",alpha=0.4)
    axGridMap.set_title("%s \n (%s) \n $R$=%.3f \n $sp_{\\theta}$ = %.3f"%(neuron_name, cellnames[neuron_index], HD_meanvectorlength[neuron_index], HD_sp[neuron_index]), fontsize=16)
    if r_neuron == 0: axGridMap.text(-0.2, 1.15, "E", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')
    axGridMap.tick_params(labelsize=12)

Rank = iter(np.arange(10))
for neuron_index in np.argsort(unfiltered_relevance)[0:10]:
    r_neuron = next(Rank)
    if neuron_index+1 in interneuron: neuron_name = str("Interneuron ")+str(neuron_index + 1)
    elif neuron_index+1 in grid: neuron_name = str("Grid Cell ")+str(neuron_index + 1)
    elif neuron_index+1 in [44]: neuron_name = str("Border Cell ")+str(neuron_index + 1)
    else: neuron_name = str("Neuron ")+str(neuron_index + 1)

    axGridMap = plt.subplot(gs[1,r_neuron],projection='polar')
    axGridMap.plot(np.linspace(0, 2*np.pi, 40), HD_tuningcurve[neuron_index], 'o')
    axGridMap.plot(np.linspace(0, 2*np.pi, 360), smoothened_firingrates[neuron_index], '-')
    axGridMap.set_theta_zero_location("N")
    axGridMap.patch.set_facecolor("white")
    axGridMap.grid(True,color="k",alpha=0.4)
    axGridMap.set_title("%s \n (%s) \n $R$=%.3f \n $sp_{\\theta}$ = %.3f"%(neuron_name, cellnames[neuron_index], HD_meanvectorlength[neuron_index], HD_sp[neuron_index]), fontsize=16)
    if r_neuron == 0: axGridMap.text(-0.2, 1.15, "F", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')
    axGridMap.tick_params(labelsize=12)

plt.savefig("Figures/Figure2.pdf", bbox_inches="tight", dpi=600)
