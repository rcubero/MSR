'''
    This code is used to replot Figure 7 of the main text.
'''
# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import os, sys, glob

# some additional packages that are needed
from scipy import *
from scipy import io, signal
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr, mode
from multiprocessing import Pool

# import plotting-related packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid.inset_locator import inset_axes
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# import external dictionaries
os.chdir("../src")
from preprocess import *
from relevance import *
from spatial_quantities import *
from HD_quantities import *
from decoding import *

def getdata(indicatorname):
    fileindicator = '%s%s'%(datadirectory,indicatorname)
    filename = (glob.glob(fileindicator))[0]
    f = open(filename, 'r')
    data = f.readlines()
    data = ravel(array(data))
    f.close()
    return data

# ------------------------------------------ #
#  Decoding using the mEC neurons            #
# ------------------------------------------ #
# Load spike train data
filenames = [fname.rstrip('\n') for fname in open(os.path.join('../Flekken_Data','cell_filenames'))]
spike_times = [io.loadmat(os.path.join('../Flekken_Data',filenames[i]), squeeze_me=True)['cellTS'] for i in np.arange(len(filenames))]

# Load the positions and calculate speeds
pos = io.loadmat(os.path.join('/Users/rcubero/Dropbox/CVSintheBrain/BEN','BEN_pos.mat'), squeeze_me=True)
positions = np.array([pos['post'],(pos['posx']+pos['posx2'])/2.,(pos['posy']+pos['posy2'])/2.])
positions_r = np.array([pos['post'],pos['posx'],pos['posy']])
positions_g = np.array([pos['post'],pos['posx2'],pos['posy2']])
rat_speed = calculate_speed(positions[0], positions[1], positions[2], 0.0)
    
N_neurons = 65
binomial_decoding_bins = 20e-3
binomial_decoding_spike, binomial_t_decode = binning(binomial_decoding_bins,spike_times,True)
binomial_decoding_spike = binomial_decoding_spike.astype("bool").astype("int")

# get idealized firing rate map
binomial_t_decode = binomial_t_decode[:-1] + np.diff(binomial_t_decode)/2.
binomial_x_decode = np.interp(binomial_t_decode, positions[0], positions[1])
binomial_y_decode = np.interp(binomial_t_decode, positions[0], positions[2])
binomial_s_decode = np.interp(binomial_t_decode, positions[0], rat_speed)

# Filter for speed
min_speed = 5.0;
binomial_decode_filter = np.where(binomial_s_decode<min_speed);
binomial_decode_mask = binomial_s_decode<min_speed

# Take out the spike patterns in which the rat is moving less than the minimum speed
for i in np.arange(len(binomial_decoding_spike)): binomial_decoding_spike[i][binomial_decode_filter] = 0

# Take out the positions of the rat in which it is moving less than the minimum speed
binomial_x_decode = np.ma.array(binomial_x_decode, mask=binomial_decode_mask); binomial_y_decode = np.ma.array(binomial_y_decode, mask=binomial_decode_mask)

binomial_N_bins = 20.
binomial_unsmoothened_occupation, binomial_ux_edges, binomial_uy_edges = np.histogram2d(binomial_x_decode.compressed(), binomial_y_decode.compressed(), bins=binomial_N_bins, range=((-75.0,75.0),(-75.0,75.0)))
binomial_unsmoothened_occupation = binomial_unsmoothened_occupation.flatten()
binomial_unsmoothened_occupation = binomial_unsmoothened_occupation/np.sum(binomial_unsmoothened_occupation)
binomial_unsmoothened_occupation[binomial_unsmoothened_occupation==0] = sys.float_info.min

xmax = 75.; xmin = -75.; binomial_N_bins = 20; binomial_step = (xmax-xmin)/binomial_N_bins;
binomial_ux_edges = binomial_ux_edges[:-1] + np.diff(binomial_ux_edges)/2.;
binomial_uy_edges = binomial_uy_edges[:-1] + np.diff(binomial_uy_edges)/2.
binomial_ux_edges, binomial_uy_edges = np.meshgrid(binomial_ux_edges, binomial_uy_edges, indexing='ij')
binomial_ux_edges = binomial_ux_edges.flatten(); binomial_uy_edges = binomial_uy_edges.flatten()

binomial_position_mask = binomial_x_decode.mask
binomial_unmasked_x = np.ma.copy(binomial_x_decode); binomial_unmasked_x.mask = False
binomial_unmasked_y = np.ma.copy(binomial_y_decode); binomial_unmasked_y.mask = False
binomial_x_alloc = np.floor((binomial_unmasked_x-xmin)/binomial_step);
binomial_y_alloc = np.floor((binomial_unmasked_y-xmin)/binomial_step);
binomial_x_center = (binomial_x_alloc*binomial_step)+xmin+(binomial_step/2.);
binomial_y_center = (binomial_y_alloc*binomial_step)+xmin+(binomial_step/2.);

binomial_unsmoothened_ratemap = []
for neuron_index in np.arange(N_neurons):
    occupational_probability, xedges, yedges = np.histogram2d(binomial_x_decode.compressed(),binomial_y_decode.compressed(), bins=binomial_N_bins,range=((-75.0,75.0),(-75.0,75.0)))
    spike_map = np.histogram2d(np.repeat(binomial_x_decode,binomial_decoding_spike[neuron_index]).compressed(),np.repeat(binomial_y_decode,binomial_decoding_spike[neuron_index]).compressed(),bins=binomial_N_bins,range=((-75.0,75.0),(-75.0,75.0)))[0]
    spike_map = sensibly_divide(spike_map, occupational_probability)
    spike_map[np.isnan(spike_map)] = 0
    spike_map[spike_map==0] = 1e-15
    binomial_unsmoothened_ratemap.append(spike_map)
binomial_unsmoothened_rates = [binomial_unsmoothened_ratemap[i].flatten() for i in np.arange(N_neurons)]

try:
    res_temporal_top_ml_weighted = np.loadtxt("Data_Output/res_temporal_top_ml_weighted.d")
    res_skaggs_top_ml_weighted = np.loadtxt("Data_Output/res_skaggs_top_ml_weighted.d")
    res_grids_ml_weighted = np.loadtxt("Data_Output/res_grids_ml_weighted.d")
    res_temporal_bot_ml_weighted = np.loadtxt("Data_Output/res_temporal_bot_ml_weighted.d")
    res_skaggs_bot_ml_weighted = np.loadtxt("Data_Output/res_skaggs_bot_ml_weighted.d")

except:
    # Load neuron index
    unfiltered_relevance = np.loadtxt("Data_Output/unfiltered_relevance.d")

    trikernel_skaggs = np.loadtxt("Data_Output/spatial_info.d")
    randomized_skaggs, randomized_skaggs_std = np.loadtxt("Data_Output/randomized_spatial_info.d")
    normed_skaggs = (trikernel_skaggs - randomized_skaggs)

    grid = [7,9,11,13,15,17,19,20,23,24,25,27,28,33,36,37,39,40,41,42,52,60,61,62,63,64,65]
    grid_index = np.array(grid)-1

    # Begin spatial decoding for mEC neurons
    res_temporal_top_ml_weighted = parallelized_binomial_decoding(np.argsort(-unfiltered_relevance)[0:20], binomial_unsmoothened_rates, binomial_decoding_spike, binomial_N_bins, binomial_unsmoothened_occupation, binomial_x_center, binomial_y_center, binomial_position_mask, binomial_ux_edges, binomial_uy_edges)
    res_skaggs_top_ml_weighted = parallelized_binomial_decoding(np.argsort(-normed_skaggs)[0:20], binomial_unsmoothened_rates, binomial_decoding_spike, binomial_N_bins, binomial_unsmoothened_occupation, binomial_x_center, binomial_y_center, binomial_position_mask, binomial_ux_edges, binomial_uy_edges)
    res_grids_ml_weighted = parallelized_binomial_decoding(grid_index, binomial_unsmoothened_rates, binomial_decoding_spike, binomial_N_bins, binomial_unsmoothened_occupation, binomial_x_center, binomial_y_center, binomial_position_mask, binomial_ux_edges, binomial_uy_edges)
    res_temporal_bot_ml_weighted = parallelized_binomial_decoding(np.argsort(-unfiltered_relevance)[45:65], binomial_unsmoothened_rates, binomial_decoding_spike, binomial_N_bins, binomial_unsmoothened_occupation, binomial_x_center, binomial_y_center, binomial_position_mask, binomial_ux_edges, binomial_uy_edges)
    res_skaggs_bot_ml_weighted = parallelized_binomial_decoding(np.argsort(-normed_skaggs)[45:65], binomial_unsmoothened_rates, binomial_decoding_spike, binomial_N_bins, binomial_unsmoothened_occupation, binomial_x_center, binomial_y_center, binomial_position_mask, binomial_ux_edges, binomial_uy_edges)

    np.savetxt("Data_Output/res_temporal_top_ml_weighted.d", res_temporal_top_ml_weighted)
    np.savetxt("Data_Output/res_skaggs_top_ml_weighted.d", res_skaggs_top_ml_weighted)
    np.savetxt("Data_Output/res_grids_ml_weighted.d", res_grids_ml_weighted)
    np.savetxt("Data_Output/res_temporal_bot_ml_weighted.d", res_temporal_bot_ml_weighted)
    np.savetxt("Data_Output/res_skaggs_bot_ml_weighted.d", res_skaggs_bot_ml_weighted)


# ------------------------------------------ #
#  Decoding using the ADn neurons            #
# ------------------------------------------ #

# decoding head directions for Mouse12
try:
    relevance_decoder = np.loadtxt("Data_Output/Mouse12-120806-HD-relevance_decoder.d")
    informative_decoder = np.loadtxt("Data_Output/Mouse12-120806-HD-informative_decoder.d")

    cm_error = np.linspace(0,2.0*np.pi,100)*180/np.pi
    random_cm_error = np.zeros((500,len(cm_error)))
    for i in np.arange(500):
        random_decoder = np.loadtxt("Data_Output/HD_randomization/random_decoder"+str(i)+".d")
        random_cm_error[i] = np.array([len(np.where(random_decoder*180/np.pi<=cm_e)[0])/len(random_decoder) for cm_e in cm_error])

    random_mean_cm_error = np.mean(random_cm_error,axis=0)
    random_std_cm_error = np.std(random_cm_error,axis=0)

except:
    data_name = "Mouse12-120806"
    datadirectory = '/Users/rcubero/Dropbox/Peyrache_Data_NEVR3004/%s/'%(data_name)
    outputname = '/Users/rcubero/Dropbox/Peyrache_Data_NEVR3004/Results/%s-HD'%(data_name)

    fileindicator = '%s%s'%(datadirectory,"Mouse*states.Wake")
    filename = (glob.glob(fileindicator))[0]
    awakeStart, awakeEnd = np.loadtxt(filename)
    minimumSpikesForToBother = 100.
    smoothingWindow = 20 # smoothing in degrees

    numfiles = 13 # number of files (clu and res) to read in
    ephysAcquisitionRate = 20000. # number of samples per second for the acquisition system
    angleAcquisitionRate = 1250./32. #EEG sampling rate divided by 32 = 39.0625 Hz
    binningtime = 1./angleAcquisitionRate

    # Get the spike data
    cellnames = []; cellspikes = []
    for i in range(numfiles):
        clusters = getdata( "Mouse*.clu.%d"%(i+1) )
        clusters = clusters.astype(np.int)
        timestamps = getdata( "Mouse*.res.%d"%(i+1) )
        timestamps = timestamps.astype(np.float)
        timestamps = 1000. * timestamps / ephysAcquisitionRate #convert time stamps to milliseconds

        numclusters = clusters[0]
        clusters = clusters[1:] ## chop off the first line!
        for j in range(numclusters):
            cellnames.append('T%dC%d'%(i+1,j+1)) #e.g. T2C4 is tetrode 2, cell 4
            cellspikes.append(timestamps[clusters==j+1]) # this is an array of all the time stamps in seconds

    # Get the angle data
    angledata = getdata( "Mouse*.ang" )
    angledata = angledata.astype(np.float) # is currently in radians (0, 2pi)
    angletime = 1000. * arange(len(angledata)) / angleAcquisitionRate # Convert time to milliseconds
    angledata[ angledata < -0.01 ] = np.nan # They labeled their missed points with negative numbers

    # Chop out the data of interest (here only the time points the animal was awake)
    awakeAngleData = angledata[ (angletime>=awakeStart*1000.) * (angletime<=awakeEnd*1000.) ]
    awakeAngleTime = angletime[ (angletime>=awakeStart*1000.) * (angletime<=awakeEnd*1000.) ]

    # Resample the angular data at 1ms bins
    binnedAwakeTime = np.arange(awakeAngleTime[0], awakeAngleTime[-1]+0.000001, 1)
    resampledAwakeAngleData = np.zeros(len(binnedAwakeTime))
    resampledAwakeAngleData = np.nan
    for i in range(len(binnedAwakeTime)):
        idx = np.searchsorted(awakeAngleTime, binnedAwakeTime[i], side="left")
        closestTime = awakeAngleTime[idx]
        if(abs(binnedAwakeTime[i] - closestTime) < 0.5): #time points are close enough
            resampledAwakeAngleData[i] = awakeAngleData[idx]
            continue
        if(binnedAwakeTime[i] > closestTime and idx+1 < len(awakeAngleTime)):
            resampledAwakeAngleData[i] = local_angle_interpolation(awakeAngleData, awakeAngleTime, binnedAwakeTime[i], idx, idx+1)
            continue
        if(binnedAwakeTime[i] <= closestTime and idx-1 > 0):
            resampledAwakeAngleData[i] = local_angle_interpolation(awakeAngleData, awakeAngleTime, binnedAwakeTime[i], idx-1, idx)
            continue

    celldata = np.zeros((len(cellnames), len(binnedAwakeTime)))
    for i in range(len(cellnames)):
        for j in range(len(cellspikes[i])):
            # note 1ms binning means that number of ms from start is the correct index
            tt = int(floor((cellspikes[i])[j] - binnedAwakeTime[0]) )
            if(tt>len(binnedAwakeTime)-1 or tt<0): # check if outside bounds of the awake time
                continue
            celldata[i,tt] += 1 # add a spike to the thing

    # Resample into 100 ms time bins
    n = 100
    decoding_spikes = []
    for neuron_number in np.arange(celldata.shape[0]):
        s = [celldata[neuron_number][n-i::n] for i in range(n)]
        decoding_spikes.append(np.array([sum(r) for r in zip_longest(*s, fillvalue=0)]).astype("int"))
    decoding_spikes = np.array(decoding_spikes)

    xmax = 2.0*np.pi; xmin = 0.0; N_bins = 40; step = (xmax-xmin)/N_bins;
    unmasked_theta = np.copy(resampledAwakeAngleData)
    unmasked_theta[np.where(unmasked_theta==-1)[0]] = np.nan
    theta_alloc = np.floor((unmasked_theta-xmin)/step);
    s = [np.array(theta_alloc[n-i::n]) for i in range(n)]
    theta_alloc = np.array([mode(np.array(r),nan_policy='omit')[0][0] if mode(np.array(r),nan_policy='omit')[1][0]>0 else np.nan for r in zip_longest(*s, fillvalue=0)])
    theta_alloc = np.array(theta_alloc)
    theta_center = (theta_alloc*step)+xmin+(step/2.)
    nothing_to_decode = np.where(np.isnan(theta_alloc))[0]

    unfiltered_relevance = np.loadtxt('Data_Output/%s-unfiltered_relevance.d'%(data_name))
    HD_info = np.loadtxt('Data_Output/%s-HD_info.d'%(data_name))
    randomized_HD_info, randomized_std_HD_info = np.loadtxt('Data_Output/%s-randomized_HD_info.d'%(data_name))
    nonnan_quantities = np.where(~np.isnan(unfiltered_relevance))[0]

    N_neurons = len(cellspikes)
    cellspikes = [cellspikes[i][ (cellspikes[i]>=awakeStart*1000.) * (cellspikes[i]<=awakeEnd*1000.) ] - awakeStart*1000 for i in np.arange(len(cellspikes))]
    spikedata = zeros((len(cellspikes), len(awakeAngleTime)))
    for i in np.arange(len(cellspikes)):
        for j in np.arange(len(cellspikes[i])):
            tt = int(floor(cellspikes[i][j]/(binningtime*1000)))
            if(tt>len(awakeAngleTime)-1 or tt<0): # check if outside bounds of the awake time
                continue
            spikedata[i,tt] += 1 # add a spike to the thing
    HD_tuningcurve = HD_tuningcurves(N_neurons, spikedata, awakeAngleData, binningtime)
    occupational_probability = HD_counts(awakeAngleData)
    occupational_probability = occupational_probability/np.sum(occupational_probability)

    angles = arange(0, 2.*pi+0.0001, pi/20)
    anglevertices = (0.5*(angles[0:(-1)]+angles[1:])) * 360. / (2.*pi)

    relevance_decoder = parallelized_poisson_decoding(nonnan_quantities[np.argsort(-unfiltered_relevance[nonnan_quantities])[0:30]], HD_tuningcurve, decoding_spikes, occupational_probability, theta_center, nothing_to_decode, (anglevertices*np.pi)/180.)
    informative_decoder = parallelized_poisson_decoding(nonnan_quantities[np.argsort(-(HD_info[nonnan_quantities]-randomized_HD_info[nonnan_quantities]))[0:30]], HD_tuningcurve, decoding_spikes, occupational_probability, theta_center, nothing_to_decode, (anglevertices*np.pi)/180.)

    np.savetxt("Data_Output/Mouse12-120806-HD-relevance_decoder.d", relevance_decoder)
    np.savetxt("Data_Output/Mouse12-120806-HD-informative_decoder.d", informative_decoder)

    random_cm_error = np.zeros((500,len(cm_error)))
    cm_error = np.linspace(0,2.0*np.pi,100)*180/np.pi
    for i in np.arange(1000):
        random_decoder = parallelized_poisson_decoding(np.random.choice(nonnan_quantities, 30, replace=False), HD_tuningcurve, decoding_spikes, occupational_probability, theta_center, nothing_to_decode, (anglevertices*np.pi)/180.)
        random_cm_error[i] = np.array([len(np.where(random_decoder*180/np.pi<=cm_e)[0])/len(random_decoder) for cm_e in cm_error])

    random_mean_cm_error = np.mean(random_cm_error,axis=0)
    random_std_cm_error = np.std(random_cm_error,axis=0)

# decoding head directions for Mouse28
try:
    relevance_decoder_PoS = np.loadtxt("Data_Output/Mouse28-140313-HD-relevance_decoder_30_ADnOnly.d")
    informative_decoder_PoS = np.loadtxt("Data_Output/Mouse28-140313-HD-informative_decoder_30_ADnOnly.d")

    relevance_decoder_ADn = np.loadtxt("Data_Output/Mouse28-140313-HD-relevance_decoder_30_PoSOnly.d")
    informative_decoder_ADn = np.loadtxt("Data_Output/Mouse28-140313-HD-informative_decoder_30_PoSOnly.d")

    cm_error = np.linspace(0,2.0*np.pi,100)*180/np.pi
    random_cm_error = np.zeros((1000,len(cm_error)))
    for i in np.arange(1000):
        random_decoder = np.loadtxt("Data_Output/HD_randomization/Mouse28-140313_random_decoder"+str(i)+".d")
        random_cm_error[i] = np.array([len(np.where(random_decoder*180/np.pi<=cm_e)[0])/len(random_decoder) for cm_e in cm_error])

    random_mean_cm_error_m28 = np.mean(random_cm_error,axis=0)
    random_std_cm_error_m28 = np.std(random_cm_error,axis=0)

except:
    pass

fig = plt.figure(dpi=300)
fig.set_size_inches(12,15)

gs0 = gridspec.GridSpec(3, 1, height_ratios=[0.9,1.5,1.2])

gs = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=[2,1], subplot_spec=gs0[0], hspace=0.5)

ax = plt.subplot(gs[:, :-1], projection='3d')
spikes = binomial_decoding_spike[np.sort(np.argsort(-unfiltered_relevance)[0:20]),500]
ratemap_stacked = [binomial_unsmoothened_rates[neuron_index] for neuron_index in np.sort(np.argsort(-unfiltered_relevance)[0:20])]
argument = multinomial_distribution(spikes, ratemap_stacked, binomial_unsmoothened_occupation)
probability = np.exp(argument); probability[probability<=1e-100] = 0;
normalizer = np.sum(probability); probability = probability/normalizer; probability = probability.reshape(20,20)
X = binomial_ux_edges.reshape(20,20); Y = binomial_uy_edges.reshape(20,20)
surf = ax.plot_surface(X, Y, probability, cmap=cm.viridis, linewidth=0, antialiased=True)
ax.set_xlabel(r'$x$',fontsize=16); ax.set_ylabel(r'$y$',fontsize=16)
ax.xaxis.set_tick_params(size=14); ax.yaxis.set_tick_params(size=14)
ax.zaxis.set_major_locator(LinearLocator(10)); ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
ax.set_zlabel(r'$p( x \vert s)$', fontsize=16); ax.zaxis.set_tick_params(size=14)
fig.colorbar(surf, shrink=0.5, aspect=7, pad = 0.1)

ax = plt.subplot(gs[:-1, -1])
ax = plt.subplot(gs[-1, -1])

gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[1], hspace=0.3)

ax = plt.subplot(gs[:, :])
cm_error = np.arange(5,215,5)
ax.plot(cm_error, [len(np.where(res_temporal_top_ml_weighted[0]<=cm_e)[0])/len(res_temporal_top_ml_weighted[0]) for cm_e in cm_error], "s-", c="purple", label="Top 20 Temporally Relevant Neurons")
ax.plot(cm_error, [len(np.where(res_skaggs_top_ml_weighted[0]<=cm_e)[0])/len(res_skaggs_top_ml_weighted[0]) for cm_e in cm_error], "*-", c="y", label="Top 20 Spatially Informative Neurons")
ax.plot(cm_error, [len(np.where(res_temporal_bot_ml_weighted[0]<=cm_e)[0])/len(res_temporal_bot_ml_weighted[0]) for cm_e in cm_error], "s--", c="purple", markerfacecolor="none", label="Bottom 20 Temporally Relevant Neurons")
ax.plot(cm_error, [len(np.where(res_skaggs_bot_ml_weighted[0]<=cm_e)[0])/len(res_skaggs_bot_ml_weighted[0]) for cm_e in cm_error], "*--", c="y", markerfacecolor="none", label="Bottom 20 Spatially Informative Neurons")
ax.plot(cm_error, [len(np.where(res_grids_ml_weighted[0]<=cm_e)[0])/len(res_grids_ml_weighted[0]) for cm_e in cm_error], ">-", c="orange", label="Grid Cells Only")
ax.legend(loc="upper left", fontsize=10)
ax.set_xlabel("$x$ (cm)", fontsize=16)
ax.set_ylabel(r'$P( \| \hat{X} - X_{true} \| \leq x)$', fontsize=16)
ax.set_ylim(top=1.01)
ax.text(-0.05, 1.08, "A", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')


gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[2], hspace=0.5)

ax = plt.subplot(gs[:, :-1])
cm_error = np.linspace(0,2.0*np.pi,100)*180/np.pi
ax.plot(cm_error, [len(np.where(relevance_decoder*180/np.pi<=cm_e)[0])/len(relevance_decoder) for cm_e in cm_error], "s-", c="purple", markersize=3, label="Top 30 Temporally Relevant Neurons")
ax.plot(cm_error, [len(np.where(informative_decoder*180/np.pi<=cm_e)[0])/len(informative_decoder) for cm_e in cm_error], "*-", c="y", markersize=3, label="Top 30 HD Informative Neurons")

ax.plot(cm_error, random_mean_cm_error, "-", c="grey", alpha=0.2, label="30 Random Neurons")
ax.fill_between(cm_error, random_mean_cm_error-random_std_cm_error, random_mean_cm_error+random_std_cm_error, color="grey", alpha=0.1)

ax.legend(loc="lower right", fontsize=10)
ax.set_xlabel("$\\theta$ (deg)", fontsize=16)
ax.set_ylabel(r'$P( \| \hat{\theta} - \theta_{true} \| \leq \theta)$', fontsize=16)
ax.set_ylim(top=1.01)
ax.text(-0.155, 1.05, "B", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')


ax = plt.subplot(gs[:, -1])
cm_error = np.linspace(0,2.0*np.pi,100)*180/np.pi

ax.plot(cm_error, [len(np.where(relevance_decoder_ADn*180/np.pi<=cm_e)[0])/len(relevance_decoder_ADn) for cm_e in cm_error], "o--", c="purple", markersize=3, label="Top 30 (ADn) Temporally Relevant Neurons")
ax.plot(cm_error, [len(np.where(informative_decoder_ADn*180/np.pi<=cm_e)[0])/len(informative_decoder_ADn) for cm_e in cm_error], "o--", c="y", markersize=3, label="Top 30 (ADn) HD Informative Neurons")

ax.plot(cm_error, [len(np.where(relevance_decoder_PoS*180/np.pi<=cm_e)[0])/len(relevance_decoder_PoS) for cm_e in cm_error], "+--", c="purple", markersize=3, label="Top 30 (PoS) Temporally Relevant Neurons")
ax.plot(cm_error, [len(np.where(informative_decoder_PoS*180/np.pi<=cm_e)[0])/len(informative_decoder_PoS) for cm_e in cm_error], "+--", c="y", markersize=3, label="Top 30 (PoS) HD Informative Neurons")

ax.plot(cm_error, random_mean_cm_error_m28, "-", c="grey", alpha=0.2, label="30 Random Neurons (ADn and PoS)")
ax.fill_between(cm_error, random_mean_cm_error_m28-random_std_cm_error_m28, random_mean_cm_error_m28+random_std_cm_error_m28, color="grey", alpha=0.1)

ax.legend(loc="lower right", fontsize=10)
ax.set_xlabel("$\\theta$ (deg)", fontsize=16)
ax.set_ylabel(r'$P( \| \hat{\theta} - \theta_{true} \| \leq \theta)$', fontsize=16)
ax.set_ylim(top=1.01)
ax.text(-0.155, 1.05, "C", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')

plt.savefig("Figures/Figure7.pdf", bbox_inches="tight", dpi=600)



