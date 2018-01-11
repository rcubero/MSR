'''
    This code is used to replot Figure 6 of the main text.
'''
# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import os, sys

# some additional packages that are needed
from scipy import io, signal
from scipy.ndimage.filters import gaussian_filter1d
from multiprocessing import Pool

# import plotting-related packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# import external dictionaries
from preprocess import *
from relevance import *
from spatial_quantities import *
from HD_quantities import *
from resampling import *

# Load spike train data
filenames = [fname.rstrip('\n') for fname in open(os.path.join('../Flekken_Data','cell_filenames'))]
spike_times = [io.loadmat(os.path.join('../Flekken_Data',filenames[i]), squeeze_me=True)['cellTS'] for i in np.arange(len(filenames))]

# Initialize dictionary to be passed into the resampling function
spike_data = {}

# Binarize spike train data
binning_time = 1e-3
smaller_spike_trains, t_smaller = binning(small_bins,spike_times,True)

# Load the positions and calculate speeds
pos = io.loadmat(os.path.join('../Flekken_Data','BEN/BEN_pos.mat'), squeeze_me=True)
positions = np.array([pos['post'],(pos['posx']+pos['posx2'])/2.,(pos['posy']+pos['posy2'])/2.])
positions_r = np.array([pos['post'],pos['posx'],pos['posy']])
positions_g = np.array([pos['post'],pos['posx2'],pos['posy2']])

# Load cell names
cellnames = np.loadtxt("cell_names",dtype=bytes,delimiter='\n').astype(str)

# Convert the coordinates into actual spatial coordinates
tight_range = ((-74.5, 74.5), (-74.5, 74.5))
positions[1], positions[2], info = transform(positions[1],positions[2],range_=tight_range,translate=True,rotate=True)
positions_r[1], positions_r[2], info = transform(positions_r[1],positions_r[2],range_=tight_range,translate=True,rotate=True)
positions_g[1], positions_g[2], info = transform(positions_g[1],positions_g[2],range_=tight_range,translate=True,rotate=True)
rat_speed = calculate_speed(positions[0], positions[1], positions[2], 0.0)

# load neuron index
interneuron_index = np.array([8,12,16,22,50])-1
grid_index = np.array([7,9,11,13,15,17,19,20,23,24,25,27,28,33,36,37,39,40,41,42,52,60,61,62,63,64,65])-1
bordercell_index = np.array([43])
other_neuron = np.delete(np.arange(65),np.sort(np.append(interneuron_index,np.append(grid_index,bordercell_index))))

# Filter for speed
min_speed = 5.0; speed_filter = np.where(rat_speed[:-1]<min_speed)

# Resample the position, head direction and speed data into 1-ms bins
t_smaller = np.arange(positions[0][0], positions[0][-1]+(0.1*small_bins),small_bins)
x_smaller = np.interp(t_smaller, positions[0], positions[1]); y_smaller = np.interp(t_smaller, positions[0], positions[2])
xr_smaller = np.interp(t_smaller, positions[0], positions_r[1]); yr_smaller = np.interp(t_smaller, positions[0], positions_r[2])
xg_smaller = np.interp(t_smaller, positions[0], positions_g[1]); yg_smaller = np.interp(t_smaller, positions[0], positions_g[2])
hd_smaller = calculate_head_direction(xr_smaller, xg_smaller, yr_smaller, yg_smaller)
s_smaller = np.interp(t_smaller, positions[0], rat_speed[:-1])
del xr_smaller, yr_smaller, xg_smaller, yg_smaller

min_speed = 5.0; speed_filter = np.where(s_smaller<min_speed)[0]
masked_x_smaller = np.ma.array(x_smaller, mask=(s_smaller<min_speed))
masked_y_smaller = np.ma.array(y_smaller, mask=(s_smaller<min_speed))

xmax = 75.; xmin = -75.; N_bins = 50; th_N_bins = 40;
occupational_probability, x_edges, y_edges = np.histogram2d(masked_x_smaller.compressed(), masked_y_smaller.compressed(), bins=N_bins, range=((-75.0,75.0),(-75.0,75.0)))

# Fill out dictionary to be passed into the resampling function
spike_data['binning_time'] = small_bins
spike_data['speed_filter'] = speed_filter
spike_data['pos_x'] = masked_x_smaller
spike_data['pos_y'] = masked_y_smaller
spike_data['theta'] = hd_smaller
spike_data['x_limits'] = xmax, xmin
spike_data['y_limits'] = xmax, xmin
spike_data['N_bins'] = N_bins
spike_data['HD_bins'] = th_N_bins
spike_data['x_edges'] = x_edges
spike_data['y_edges'] = y_edges

# Prepare the triweight kernel
sigma = 4.2;
kernel_x = np.linspace(-75.0,75.0,N_bins+1); kernel_x = np.diff(kernel_x)/2 + kernel_x[:-1]
kernel_y = np.linspace(-75.0,75.0,N_bins+1); kernel_y = np.diff(kernel_y)/2 + kernel_y[:-1]
kernel_x, kernel_y = np.meshgrid(kernel_x,kernel_y)
triweight_kernel = (4.*np.power(1.-(np.power(kernel_x,2)+np.power(kernel_y,2))/(9.*np.power(sigma,2)),3))/(9.*np.pi*np.power(sigma,2))
support = (np.sqrt(np.power(kernel_x,2)+np.power(kernel_y,2))<3.*sigma).astype("float")
triweight_kernel = triweight_kernel*support

resampled_relevance = {}
resampled_relevance_rhd = {}

for neuron_of_interest in np.arange(65):
    smaller_spike = np.copy(smaller_spike_trains[neuron_of_interest])
    spike_data['spikes'] = np.copy(smaller_spike)

    ratemap = firing_ratemap(1, smaller_spike, masked_x_smaller, masked_y_smaller, binning_time)
    ratemap[np.isnan(ratemap)] = 0
    ratemap = ratemap.flatten()
    spike_data['spatial_rate'] = ratemap

    HD_tuningcurve = HD_tuningcurves(1, smaller_spike, head_direction, binning_time, N_bins=40)
    HD_tuningcurve = gaussian_filter1d(HD_tuningcurve, np.sqrt(20./(360./th_N_bins)), truncate=4.0, mode='wrap')
    spike_data['HD_rate'] = HD_tuningcurve

    spike_data['n'] = 10
    if neuron_of_interest != 46:
        spike_data['calc_option'] = [True,False,False] # resample position only, head direction only and position+head direction
        resampled_relevance[neuron_of_interest] = resample_ratemap(spike_data)
    else if neuron_of_interest == 46:
        spike_data['calc_option'] = [True,False,True]
        resampled_relevance[neuron_of_interest], resampled_relevance_rhd[neuron_of_interest] = resample_ratemap(spike_data)

    # get spikes and trajectories
    if neuron_of_interest == 46:
        smaller_spike_noi = np.copy(smaller_spike_trains[neuron_of_interest])
        
        Code = [(i,chr(ord('A')+i)) for i in np.arange(25)] + [(i+25,chr(ord('a')+i)) for i in np.arange(25)]
        x_edges = x_edges[:-1] + np.diff(x_edges)/2.; y_edges = y_edges[:-1] + np.diff(y_edges)/2.
        x_edges, y_edges = np.meshgrid(x_edges, y_edges, indexing='ij')
        x_edges = x_edges.flatten(); y_edges = y_edges.flatten()
    
        step = (xmax-xmin)/N_bins;
        x_alloc_codes = np.floor((x_edges-xmin)/step); x_codes = [Code[int(i)][1] for i in x_alloc_codes]
        y_alloc_codes = np.floor((y_edges-xmin)/step); y_codes = [Code[int(i)][1] for i in y_alloc_codes]
        binning_codes = np.array([x_codes[i]+y_codes[i] for i in np.arange(len(y_codes))]).astype("str")
        x_trajectory = (np.floor((x_smaller-xmin)/step)).astype("int"); x_trajectory_codes = [Code[int(i)][1] for i in x_trajectory]
        y_trajectory = (np.floor((y_smaller-xmin)/step)).astype("int"); y_trajectory_codes = [Code[int(i)][1] for i in y_trajectory]
        trajectory_codes = np.array([np.where(binning_codes==x_trajectory_codes[i]+y_trajectory_codes[i])[0] for i in np.arange(len(x_trajectory))])
    
        # Get head direction trajectory codes
        th_max = 2.*np.pi; th_min = 0; th_step = (th_max-th_min)/th_N_bins
        hd_trajectory = (np.floor((hd_smaller-th_min)/th_step)).astype("int")

        r_trajectory = np.array([ratemap[trajectory_codes[i]]*small_bins for i in np.arange(len(trajectory_codes))])
        synthetic_noi = np.array([np.random.binomial(1, min(r_trajectory[i],1.0)) for i in np.arange(len(r_trajectory))])
        synthetic_noi[speed_filter] = 0

        rhd_trajectory = np.array([ratemap[trajectory_codes[i]]*smoothened_firingrates[hd_trajectory[i]]*small_bins for i in np.arange(len(trajectory_codes))])
        synthetic_rhd_noi = np.array([np.random.binomial(1, min(rhd_trajectory[i],1.0)) for i in np.arange(len(rhd_trajectory))])
        synthetic_rhd_noi[speed_filter] = 0

# Load first data on spatial information and spatial sparsity
try:
    unfiltered_relevance = np.loadtxt("Data_Output/unfiltered_relevance.d")
    spatial_info = np.loadtxt("Data_Output/spatial_info.d")
    spatial_sp = np.loadtxt("Data_Output/spatial_sp.d")
    randomized_spatial_info, randomized_std_spatial_info = np.loadtxt("Data_Output/randomized_spatial_info.d")
raise:
    print("Run first Figure2.py")


fig = plt.figure(dpi=300)
fig.set_size_inches(18,20)

gs0 = gridspec.GridSpec(2, 1, height_ratios=[4,2], hspace=0.15)

neuron_name = "Neuron 47 Spike Map"
gs = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[0], width_ratios = [3,0.5,0.5], wspace=0.15, hspace=0.1)
ax = plt.subplot(gs[:,0])
ax.plot(x_smaller,-y_smaller,alpha=0.3, label="Trajectory")
spikes = np.copy(smaller_spike_noi); spikes[speed_filter] = 0
ax.scatter(np.repeat(x_smaller,spikes.astype("int")),np.repeat(y_smaller,spikes.astype("int")), s=100, label='Original spikes')
spikes = np.copy(synthetic_noi); spikes[speed_filter] = 0
ax.scatter(np.repeat(x_smaller,spikes.astype("int")),np.repeat(y_smaller,spikes.astype("int")),  s=100, label=r'Resampled spikes, $p(s)=\lambda(x)\Delta t$', alpha=0.5)
spikes = np.copy(synthetic_rhd_noi); spikes[speed_filter] = 0
ax.scatter(np.repeat(x_smaller,spikes.astype("int")),np.repeat(y_smaller,spikes.astype("int")),  s=100, label=r'Resampled spikes, $p(s)=\lambda(x)\lambda(\theta)\Delta t$', alpha=0.5)
ax.set_xlim(left=-75, right=75); ax.set_ylim(bottom=-75, top=75); ax.legend(loc="best", fontsize=18); ax.tick_params(labelsize=12)
ax.set_title(neuron_name, fontsize=18)
ax.text(-0.05, 1.00, "A", transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')

ax = plt.subplot(gs[0,1])
occupational_probability, xedges, yedges = np.histogram2d(masked_x_smaller.compressed(), masked_y_smaller.compressed(), bins=200,range=((-75.0,75.0),(-75.0,75.0)))
spikes = np.copy(smaller_spike_noi); spikes[speed_filter] = 0
spike_map = np.ma.array(np.histogram2d(np.repeat(masked_x_smaller,spikes.astype("int")).compressed(), np.repeat(masked_y_smaller,spikes.astype("int")).compressed(), bins=200,range=((-75.0,75.0),(-75.0,75.0)))[0])/np.ma.array(occupational_probability)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]; blur = ndimage.filters.gaussian_filter(spike_map, 8.0, mode="reflect", truncate = 4.0)
heatmap = ax.imshow(1000.0*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet); plt.colorbar(heatmap, fraction=0.046, pad=0.04)
ax.grid(b=False, which='major'); ax.grid(b=False, which='minor'); ax.tick_params(labelsize=12)
ax.text(-0.3, 1.1, "B.1", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
ax.text(0.7, 1.35, r'Original ratemap', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

ax = plt.subplot(gs[0,2], projection="polar")
HD_occ, angle_vertices = np.histogram(hd_smaller, 40, range=(0,2.*np.pi))
HD_ratemap = np.histogram(np.repeat(hd_smaller,unfiltered_spikes), 40, range=(0,2.*np.pi))[0]
HD_ratemap = sensibly_divide(HD_ratemap, HD_occ*small_bins); HD_ratemap[np.isnan(HD_ratemap)] = 0
angle_vertices = (0.5*(angle_vertices[0:(-1)]+angle_vertices[1:])) * 360. / (2.*np.pi); smoothened_angles = np.arange(0, 360.1, 1)
smoothened_firingrates = np.zeros(smoothened_angles.shape[0]); smoothingWindow = 20
for i in range(len(smoothened_angles)):
    dd = np.abs(smoothened_angles[i] - angle_vertices); dd[dd>180.] = 360. - dd[dd>180.]
    weights = (1./( np.sqrt(2.*smoothingWindow*np.pi) )) * np.exp( -dd**2 / (2.*smoothingWindow) )
    smoothened_firingrates[i] = sum(weights*HD_ratemap) / sum(weights)
ax.plot(angle_vertices*np.pi/180., HD_ratemap, 'o')
ax.plot(smoothened_angles*np.pi/180., smoothened_firingrates, '-')
ax.set_theta_zero_location("N"); ax.patch.set_facecolor("white"); ax.grid(True,color="k",alpha=0.4); ax.tick_params(labelsize=12)
ax.text(-0.3, 1.06, "B.2", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')


ax = plt.subplot(gs[1,1])
spikes = np.copy(synthetic_noi); spikes[speed_filter] = 0
occupational_probability, xedges, yedges = np.histogram2d(masked_x_smaller.compressed(), masked_y_smaller.compressed(), bins=200,range=((-75.0,75.0),(-75.0,75.0)))
spike_map = np.ma.array(np.histogram2d(np.repeat(masked_x_smaller,spikes.astype("int")).compressed(), np.repeat(masked_y_smaller,spikes.astype("int")).compressed(), bins=200,range=((-75.0,75.0),(-75.0,75.0)))[0])/np.ma.array(occupational_probability)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]; blur = ndimage.filters.gaussian_filter(spike_map, 8.0, mode="reflect", truncate = 4.0)
heatmap = ax.imshow(1000.0*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet); plt.colorbar(heatmap, fraction=0.046, pad=0.04)
ax.grid(b=False, which='major'); ax.grid(b=False, which='minor'); ax.tick_params(labelsize=12)
ax.text(-0.3, 1.1, "C.1", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
ax.text(0.35, 1.35, r'Resampled ratemap, $p(s) = \lambda(\textrm{\textbf{x}})\Delta t$', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

ax = plt.subplot(gs[1,2], projection="polar")
HD_occ, angle_vertices = np.histogram(hd_smaller, 40, range=(0,2.*np.pi))
HD_ratemap = np.histogram(np.repeat(hd_smaller,synthetic_noi), 40, range=(0,2.*np.pi))[0]
HD_ratemap = sensibly_divide(HD_ratemap, HD_occ*small_bins); HD_ratemap[np.isnan(HD_ratemap)] = 0
angle_vertices = (0.5*(angle_vertices[0:(-1)]+angle_vertices[1:])) * 360. / (2.*np.pi); smoothened_angles = np.arange(0, 360.1, 1)
smoothened_firingrates = np.zeros(smoothened_angles.shape[0]); smoothingWindow = 20
for i in range(len(smoothened_angles)):
    dd = np.abs(smoothened_angles[i] - angle_vertices); dd[dd>180.] = 360. - dd[dd>180.]
    weights = (1./( np.sqrt(2.*smoothingWindow*np.pi) )) * np.exp( -dd**2 / (2.*smoothingWindow) )
    smoothened_firingrates[i] = sum(weights*HD_ratemap) / sum(weights)
ax.plot(angle_vertices*np.pi/180., HD_ratemap, 'o')
ax.plot(smoothened_angles*np.pi/180., smoothened_firingrates, '-')
ax.set_theta_zero_location("N"); ax.patch.set_facecolor("white"); ax.grid(True,color="k",alpha=0.4); ax.tick_params(labelsize=12)
ax.text(-0.3, 1.06, "C.2", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')


ax = plt.subplot(gs[2,1])
spikes = np.copy(synthetic_rhd_noi); spikes[speed_filter] = 0
occupational_probability, xedges, yedges = np.histogram2d(masked_x_smaller.compressed(), masked_y_smaller.compressed(), bins=200,range=((-75.0,75.0),(-75.0,75.0)))
spike_map = np.ma.array(np.histogram2d(np.repeat(masked_x_smaller,spikes.astype("int")).compressed(), np.repeat(masked_y_smaller,spikes.astype("int")).compressed(), bins=200,range=((-75.0,75.0),(-75.0,75.0)))[0])/np.ma.array(occupational_probability)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]; blur = ndimage.filters.gaussian_filter(spike_map, 8.0, mode="reflect", truncate = 4.0)
heatmap = ax.imshow(1000.0*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet); plt.colorbar(heatmap, fraction=0.046, pad=0.04)
ax.grid(b=False, which='major'); ax.grid(b=False, which='minor'); ax.tick_params(labelsize=12)
ax.text(-0.3, 1.1, "D.1", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
ax.text(0.23, 1.35, r'Resampled ratemap, $p(s) = \lambda(\textrm{\textbf{x}})\lambda(\theta)\Delta t$', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

ax = plt.subplot(gs[2,2], projection="polar")
HD_occ, angle_vertices = np.histogram(hd_smaller, 40, range=(0,2.*np.pi))
HD_ratemap = np.histogram(np.repeat(hd_smaller,synthetic_rhd_noi), 40, range=(0,2.*np.pi))[0]
HD_ratemap = sensibly_divide(HD_ratemap, HD_occ*small_bins); HD_ratemap[np.isnan(HD_ratemap)] = 0
angle_vertices = (0.5*(angle_vertices[0:(-1)]+angle_vertices[1:])) * 360. / (2.*np.pi); smoothened_angles = np.arange(0, 360.1, 1)
smoothened_firingrates = np.zeros(smoothened_angles.shape[0]); smoothingWindow = 20
for i in range(len(smoothened_angles)):
    dd = np.abs(smoothened_angles[i] - angle_vertices); dd[dd>180.] = 360. - dd[dd>180.]
    weights = (1./( np.sqrt(2.*smoothingWindow*np.pi) )) * np.exp( -dd**2 / (2.*smoothingWindow) )
    smoothened_firingrates[i] = sum(weights*HD_ratemap) / sum(weights)
ax.plot(angle_vertices*np.pi/180., HD_ratemap, 'o')
ax.plot(smoothened_angles*np.pi/180., smoothened_firingrates, '-')
ax.set_theta_zero_location("N"); ax.patch.set_facecolor("white"); ax.grid(True,color="k",alpha=0.4); ax.tick_params(labelsize=12)
ax.text(-0.3, 1.06, "D.2", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')


ax = plt.subplot(gs[3,1:3])
barlist = ax.bar([1,1.4,1.8], [unfiltered_relevance[46], np.mean(resampled_relevance[46]), np.mean(resampled_relevance_rhd[46])], width=0.25, yerr=[0,np.std(resampled_relevance[46]),np.std(resampled_relevance_rhd[46])])
barlist[0].set_color("#077187"); barlist[1].set_color("#077187"); barlist[2].set_color("#077187")
ax.set_ylabel(r'multiscale relevance',fontsize=13)
ax.set_ylim(bottom=0.26, top=0.305)
ax.set_xticks([1,1.4,1.8]);
ax.set_xticklabels(('Original Ratemap', r'Resampled Ratemap', r'Resampled Ratemap'),fontsize=10)
ax.text(0.45, -0.13, r'$\lambda(\textrm{\textbf{x}})\Delta t$', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
ax.text(0.75, -0.13, r'$\lambda(\textrm{\textbf{x}})\lambda(\theta)\Delta t$', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')
ax.text(-0.12, 1.05, "E", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')


gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], wspace=0.2)
mean_resampled_relevance = np.zeros(65)
for neuron_of_interest in np.arange(65):
    mean_resampled_relevance[neuron_of_interest] = np.mean(resampled_relevance[neuron_of_interest])
resampling_difference = unfiltered_relevance - mean_resampled_relevance

y_data = np.copy((spatial_info - randomized_spatial_info))
ax = plt.subplot(gs[:,0])
ax.scatter(resampling_difference[grid_index], y_data[grid_index], c="green", marker="o")
ax.scatter(resampling_difference[interneuron_index], y_data[interneuron_index], c="red", marker="s")
ax.scatter(resampling_difference[bordercell_index], y_data[bordercell_index], c="purple", marker="*")
ax.scatter(resampling_difference[other_neuron], y_data[other_neuron], c="grey", marker="^")
for u in np.arange(len(n_data)): ax.annotate(n_data[u],(resampling_difference[u]+0.00008, y_data[u]+0.001),fontsize=8)
ax.set_ylabel(r'spatial information $I(s,\textrm{\textbf{x}})$ (bits per spike)')
ax.set_xlabel(r'differential MSR, $\mathcal{R}_t^{\textrm{original}} - \mathcal{R}_t^{\textrm{resampled}}$')
ax.text(-0.08, 1.05, "F", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

y_data = np.copy(spatial_sp)
ax = plt.subplot(gs[:,1])
ax.scatter(resampling_difference[grid_index], y_data[grid_index], c="green", marker="o")
ax.scatter(resampling_difference[interneuron_index], y_data[interneuron_index], c="red", marker="s")
ax.scatter(resampling_difference[bordercell_index], y_data[bordercell_index], c="purple", marker="*")
ax.scatter(resampling_difference[other_neuron], y_data[other_neuron], c="grey", marker="^")
for u in np.arange(len(n_data)): ax.annotate(n_data[u],(resampling_difference[u]+0.00008, y_data[u]+0.001),fontsize=8)
ax.set_ylabel(r'spatial sparsity $sp_{\textrm{\textbf{x}}}$')
ax.set_xlabel(r'differential MSR, $\mathcal{R}_t^{\textrm{original}} - \mathcal{R}_t^{\textrm{resampled}}$')
ax.text(-0.08, 1.05, "G", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

plt.savefig("Figure6.pdf", bbox_inches="tight", dpi=600)
plt.show()
