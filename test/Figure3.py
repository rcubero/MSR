'''
    This code is used to replot Figure 3 of the main text.
'''
# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import os, sys

# some additional packages that are needed
from scipy import io, signal
from scipy.ndimage.filters import gaussian_filter1d
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
cellnames = np.loadtxt(os.path.join('../Flekken_Data','cell_names'),dtype=bytes,delimiter='\n').astype(str)

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

# follow the curves for the grid cell and interneuron of interest
interneuron_index = np.array([8,12,16,22,50])-1
grid_index = np.array([7,9,11,13,15,17,19,20,23,24,25,27,28,33,36,37,39,40,41,42,52,60,61,62,63,64,65])-1
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
    spatial_sp = np.loadtxt("Data_Output/spatial_sp.d")
    spatial_gridscore = np.loadtxt("Data_Output/grid_score.d")
    HD_sp = np.loadtxt("Data_Output/HD_sp.d")
    HD_meanvectorlength = np.loadtxt("Data_Output/HD_meanvectorlength.d")

except:
    spatial_sp = spatial_sparsity(N_neurons, spike_trains, unmasked_x_t, unmasked_y_t, binning_time, output_name="Data_Output/spatial_sp.d")
    spatial_gridscore = np.loadtxt("Data_Output/grid_score.d") # pre-calculated grid scores taken from Dunn (2017)
    HD_sp = HD_sparsity(N_neurons, unfiltered_spike_trains, head_direction, binning_time, output_name="Data_Output/HD_sp.d")
    HD_meanvectorlength = mean_vector_length(N_neurons, unfiltered_spike_trains, head_direction, binning_time, output_name="Data_Output/HD_meanvectorlength.d")

# plot Figure 3
relevance_range = np.linspace(np.amin(unfiltered_relevance), np.amax(unfiltered_relevance)+0.001,6)
population = [len(spatial_sp[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
inds = 4.0*np.arange(5)+2.5
label = ["%0.3f"%(relevance_range[i]) for i in np.arange(6)]
inds_ticks = 4.0*np.arange(6)+0.35

fig, ax = plt.subplots(1,2, dpi=600)
fig.set_size_inches(20,8)

mean_scores = [np.mean(spatial_gridscore[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
std_scores = [np.std(spatial_gridscore[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
bar1 = ax[0].bar(inds-1.0, mean_scores, width=1.3, color="red", edgecolor="grey", yerr=std_scores, alpha=0.6, error_kw=dict(lw=2, capsize=4, capthick=2), label=r'grid score, $g$')
ax[0].set_xticks(inds_ticks)
ax[0].set_xticklabels(label)
ax[0].tick_params(labelsize=14)
ax[0].set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=18)

pearson_r = pearsonr(spatial_gridscore, unfiltered_relevance)
spearman_r = spearmanr(spatial_gridscore, unfiltered_relevance)
ax[0].bar(inds+0.5, [-100 for i in np.arange(len(mean_scores))], width=1.3, color="white", edgecolor="white", label='$\\rho_p=$ %.3f, $P=$ %.2e'%(pearson_r[0], pearson_r[1]))
ax[0].bar(inds+0.5, [-100 for i in np.arange(len(mean_scores))], width=1.3, color="white", edgecolor="white", label='$\\rho_s=$ %.3f, $P=$ %.2e'%(spearman_r[0], spearman_r[1]))

mean_scores = [np.mean(HD_meanvectorlength[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
std_scores = [np.std(HD_meanvectorlength[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
bar1 = ax[0].bar(inds+0.5, mean_scores, width=1.3, color="yellow", edgecolor="grey", yerr=std_scores, alpha=0.6, error_kw=dict(lw=2, capsize=4, capthick=2), label=r'mean vector length, $R$')
ax[0].set_xticks(inds_ticks)
ax[0].set_xticklabels(label)
ax[0].tick_params(labelsize=14)
ax[0].set_ylim(bottom=-0.20, top=0.9)

pearson_r = pearsonr(HD_meanvectorlength, unfiltered_relevance)
spearman_r = spearmanr(HD_meanvectorlength, unfiltered_relevance)
ax[0].bar(inds+0.5, [-100 for i in np.arange(len(mean_scores))], width=1.3, color="white", edgecolor="white", label='$\\rho_p=$ %.3f, $P=$ %.2e'%(pearson_r[0], pearson_r[1]))
ax[0].bar(inds+0.5, [-100 for i in np.arange(len(mean_scores))], width=1.3, color="white", edgecolor="white", label='$\\rho_s=$ %.3f, $P=$ %.2e'%(spearman_r[0], spearman_r[1]))

ax[0].legend(loc="upper left", fontsize=16)
for i, v in enumerate(population):
    ax[0].text(inds[i]-0.8, -0.19, r'$n=$'+str(v), color='black', fontsize=12)

ax[0].text(-0.1, 1.06, "A", transform=ax[0].transAxes, fontsize=18, fontweight='bold', va='top')

mean_scores = [np.mean(spatial_sp[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
std_scores = [np.std(spatial_sp[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
bar1 = ax[1].bar(inds-1.0, mean_scores, width=1.3, color="orange", edgecolor="grey", yerr=std_scores, alpha=0.6, error_kw=dict(lw=2, capsize=4, capthick=2), label=r'spatial sparsity, $sp_{\textrm{\textbf{x}}}$')
ax[1].set_xticks(inds_ticks)
ax[1].set_xticklabels(label)
ax[1].tick_params(labelsize=14)
ax[1].set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=18)

pearson_r = pearsonr(spatial_sp, unfiltered_relevance)
spearman_r = spearmanr(spatial_sp, unfiltered_relevance)
ax[1].bar(inds+0.5, [-100 for i in np.arange(len(mean_scores))], width=1.3, color="white", edgecolor="white", label='$\\rho_p=$ %.3f, $P=$ %.2e'%(pearson_r[0], pearson_r[1]))
ax[1].bar(inds+0.5, [-100 for i in np.arange(len(mean_scores))], width=1.3, color="white", edgecolor="white", label='$\\rho_s=$ %.3f, $P=$ %.2e'%(spearman_r[0], spearman_r[1]))

mean_scores = [np.mean(HD_sp[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
std_scores = [np.std(HD_sp[np.where((unfiltered_relevance>=relevance_range[i])*(unfiltered_relevance<relevance_range[i+1]))[0]]) for i in np.arange(len(relevance_range)-1)]
bar1 = ax[1].bar(inds+0.5, mean_scores, width=1.3, color="purple", edgecolor="grey", yerr=std_scores, alpha=0.6, error_kw=dict(lw=2, capsize=4, capthick=2), label=r'head directional sparsity, $sp_{\theta}$')
ax[1].set_xticks(inds_ticks)
ax[1].set_xticklabels(label)
ax[1].tick_params(labelsize=14)
ax[1].set_ylim(bottom=-0.09, top=0.8)

pearson_r = pearsonr(HD_sp, unfiltered_relevance)
spearman_r = spearmanr(HD_sp, unfiltered_relevance)
ax[1].bar(inds+0.5, [-100 for i in np.arange(len(mean_scores))], width=1.3, color="white", edgecolor="white", label='$\\rho_p=$ %.3f, $P=$ %.2e'%(pearson_r[0], pearson_r[1]))
ax[1].bar(inds+0.5, [-100 for i in np.arange(len(mean_scores))], width=1.3, color="white", edgecolor="white", label='$\\rho_s=$ %.3f, $P=$ %.2e'%(spearman_r[0], spearman_r[1]))

ax[1].legend(loc="upper left", fontsize=16)
for i, v in enumerate(population):
    ax[1].text(inds[i]-0.8, -0.08, r'$n=$'+str(v), color='black', fontsize=12)

ax[1].text(-0.1, 1.06, "B", transform=ax[1].transAxes, fontsize=18, fontweight='bold', va='top')

plt.savefig("Figures/Figure3.pdf", bbox_inches="tight", dpi=600)
