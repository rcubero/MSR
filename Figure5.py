'''
    This code is used to replot Figure 5 of the main text.
'''
# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import glob, os

# some additional packages that are needed
from scipy import *
from scipy import io, signal
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
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

def getdata(indicatorname):
    fileindicator = '%s%s'%(datadirectory,indicatorname)
    filename = (glob.glob(fileindicator))[0]
    f = open(filename, 'r')
    data = f.readlines()
    data = ravel(array(data))
    f.close()
    return data

def getposdata(indicatorname):
    fileindicator = '%s%s'%(datadirectory,indicatorname)
    filename = (glob.glob(fileindicator))[0]
    f = open(filename, 'r')
    data = f.readlines()
    data = ravel(array(data))
    numberofrows = len(data)
    numberofcolumns = len( (data[0]).split() )
    if(numberofcolumns == 1): data = data.astype(np.float)
    else:
        fulldata = zeros((numberofrows, numberofcolumns))
        for i in range(numberofrows):
            col = ravel(array(  (data[i]).split()  ))
            col = col.astype(np.float)
            if (len(col)==numberofcolumns): fulldata[i,:] = col + 0.
            else:
                fulldata[i,:] = np.array([-1]*4)
        data = fulldata
    
    f.close()
    return data

# -----IMPORTANT-----
# These entries need to be changed
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

# Get the position data
originalpositiondata = getposdata( "Mouse*.pos" )
originalpositiondata = originalpositiondata.astype(np.float)
originalpositiondata[ originalpositiondata < -0.01 ] = np.nan # They labeled their missed points with negative numbers

# Chop out the data of interest (here only the time points the animal was awake)
awakeAngleData = angledata[ (angletime>=awakeStart*1000.) * (angletime<=awakeEnd*1000.) ]
awakePosData = originalpositiondata[ (angletime>=awakeStart*1000.) * (angletime<=awakeEnd*1000.) ]
awakeAngleTime = angletime[ (angletime>=awakeStart*1000.) * (angletime<=awakeEnd*1000.) ]

awake_x_pos = (awakePosData[:,0] + awakePosData[:,2])/2.
awake_y_pos = (awakePosData[:,1] + awakePosData[:,3])/2.

# Mask timepoints with missing information
awakeAngleData = np.ma.array(awakeAngleData, mask=np.isnan(awakeAngleData))
awake_x_pos = np.ma.array(awake_x_pos, mask=np.isnan(awake_x_pos))
awake_y_pos = np.ma.array(awake_y_pos, mask=np.isnan(awake_y_pos))

# Bin the spike time for MSR calculation
cellspikes = [cellspikes[i][ (cellspikes[i]>=awakeStart*1000.) * (cellspikes[i]<=awakeEnd*1000.) ] - awakeStart*1000 for i in np.arange(len(cellspikes))]
celldata, binnedAwakeTime = binning(10, cellspikes, True, t_stop=(awakeEnd-awakeStart)*1000) # binned at 10ms

# Calculate MSR
try:
    unfiltered_relevance = np.loadtxt('Data_Output/%s-unfiltered_relevance.d'%(data_name))
except:
    unfiltered_relevance = np.zeros(celldata.shape[0])
    unfiltered_relevance[:] = np.nan
    N_neurons, T_total = celldata.shape
    for i in np.arange(N_neurons):
        if(np.sum(celldata[i])>minimumSpikesForToBother):
            unfiltered_relevance[i] = parallelized_total_relevance((T_total, celldata[i]))
    np.savetxt('Data_Output/%s-unfiltered_relevance.d'%(data_name), unfiltered_relevance)

# Resample the spikes to correspond with tracking data
spikedata = zeros((len(cellspikes), len(awakeAngleTime)))
for i in np.arange(len(cellspikes)):
    for j in np.arange(len(cellspikes[i])):
        tt = int(floor(cellspikes[i][j]/(binningtime*1000)))
        if(tt>len(awakeAngleTime)-1 or tt<0): # check if outside bounds of the awake time
            continue
        spikedata[i,tt] += 1 # add a spike to the thing

tight_range = ((-26.5,26.5),(-23.0,23.0))
awake_x_pos, awake_y_pos, info = transform(awake_x_pos,awake_y_pos,range_=tight_range,translate=True,rotate=True)
mask_data = awake_x_pos.mask

try:
    Nbins = (25,20)
    N_neurons = len(cellspikes)
    
    spatial_info = np.loadtxt('Data_Output/%s-spatial_info.d'%(data_name))
    spatial_sp = np.loadtxt('Data_Output/%s-spatial_sp.d'%(data_name))
    spatial_mean = np.loadtxt('Data_Output/%s-spatial_meanspike.d'%(data_name))
    randomized_spatial_info, randomized_std_spatial_info = np.loadtxt('Data_Output/%s-randomized_spatial_info.d'%(data_name))
    
    HD_tuningcurve = HD_tuningcurves(N_neurons, spikedata, awakeAngleData, binningtime)
    smoothened_firingrates = HD_tuningcurves(N_neurons, spikedata, awakeAngleData, binningtime, N_bins=360)
    smoothened_firingrates = np.array([gaussian_filter1d(smoothened_firingrates[i], np.sqrt(20), truncate=4.0, mode='wrap') for i in np.arange(N_neurons)])
    HD_info = np.loadtxt('Data_Output/%s-HD_info.d'%(data_name))
    HD_sp = np.loadtxt('Data_Output/%s-HD_sp.d'%(data_name))
    HD_meanvectorlength = np.loadtxt('Data_Output/%s-HD_meanvectorlength.d'%(data_name))
    randomized_HD_info, randomized_std_HD_info = np.loadtxt('Data_Output/%s-randomized_HD_info.d'%(data_name))

except:
    Nbins = (25,20)
    N_neurons = len(cellspikes)

    # create the smoothing kernel
    kernel_x = np.linspace(tight_range[0][0],tight_range[0][1],Nbins[0]+1); kernel_x = np.diff(kernel_x)/2 + kernel_x[:-1]
    kernel_y = np.linspace(tight_range[1][0],tight_range[1][1],Nbins[1]+1); kernel_y = np.diff(kernel_y)/2 + kernel_y[:-1]
    sigma = 4.2
    kernel_x, kernel_y = np.meshgrid(kernel_x,kernel_y, indexing="ij")
    triweight_kernel = (4.*np.power(1.-(np.power(kernel_x,2)+np.power(kernel_y,2))/(9.*np.power(sigma,2)),3))/(9.*np.pi*np.power(sigma,2))
    support = (np.sqrt(np.power(kernel_x,2)+np.power(kernel_y,2))<3.*sigma).astype("float")
    triweight_kernel = triweight_kernel*support

    # calculate spatial quantities
    spatial_info = spatial_information(N_neurons, spikedata.astype("int"), awake_x_pos, awake_y_pos, binningtime, N_bins=Nbins, range=tight_range, kernel=triweight_kernel, output_name='Data_Output/%s-spatial_info.d'%(data_name))
    spatial_sp = spatial_sparsity(N_neurons, spikedata.astype("int"), awake_x_pos, awake_y_pos, binningtime, N_bins=Nbins, range=tight_range, kernel=triweight_kernel,  output_name='Data_Output/%s-spatial_sp.d'%(data_name))
    spatial_mean = spatial_meanspike(N_neurons, spikedata.astype("int"), awake_x_pos, awake_y_pos, binningtime, N_bins=Nbins, range=tight_range, kernel=triweight_kernel, output_name='Data_Output/%s-spatial_meanspike.d'%(data_name))
    randomized_spatial_info, randomized_std_spatial_info = randomized_spatial_information(N_neurons, spikedata, awake_x_pos, awake_y_pos, binningtime, N_bins=Nbins, range=tight_range, kernel=triweight_kernel, output_name='Data_Output/%s-randomized_spatial_info.d'%(data_name))

    # calculate head directional quantities
    HD_tuningcurve = HD_tuningcurves(N_neurons, spikedata.astype("int"), awakeAngleData, binningtime)
    smoothened_firingrates = HD_tuningcurves(N_neurons, spikedata.astype("int"), awakeAngleData, binningtime, N_bins=360)
    smoothened_firingrates = np.array([gaussian_filter1d(smoothened_firingrates[i], np.sqrt(20), truncate=4.0, mode='wrap') for i in np.arange(N_neurons)])
    HD_info = HD_information(N_neurons, spikedata.astype("int"), awakeAngleData, binningtime, output_name='Data_Output/%s-HD_info.d'%(data_name))
    HD_sp = HD_sparsity(N_neurons, spikedata.astype("int"), awakeAngleData, binningtime, output_name='Data_Output/%s-HD_sp.d'%(data_name))
    HD_meanvectorlength = mean_vector_length(N_neurons, spikedata.astype("int"), awakeAngleData, binningtime, output_name='Data_Output/%s-HD_meanvectorlength.d'%(data_name))
    randomized_HD_info, randomized_std_HD_info = randomized_HD_information(N_neurons, spikedata, awakeAngleData, binningtime, output_name='Data_Output/%s-randomized_HD_info.d'%(data_name))

# Note which are neurons from the ADn and which ones are not
Mouse_ToPlot = np.zeros(len(unfiltered_relevance)).astype("bool")
Mouse_ToPlot_NonADn = np.zeros(len(unfiltered_relevance)).astype("bool")
Mouse_ToConsider = data_name[0:data_name.find("-")]
for i in np.arange(len(unfiltered_relevance)):
    terminal_index = cellnames[i]
    terminal_index = terminal_index[terminal_index.find("T")+1 : terminal_index.find("C")]
    if ~np.isnan(unfiltered_relevance[i]):
        if (int(terminal_index) in np.arange(1,9,1)): Mouse_ToPlot[i] = True
        else: Mouse_ToPlot_NonADn[i] = True

# plot Figure 5
fig = plt.figure(dpi=300)
fig.set_size_inches(20,32)
gs0 = gridspec.GridSpec(4, 1, hspace=0.3)

gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0])

axScatterPlot = plt.subplot(gs[0,0])
# Plot ADn neurons
non_nan = np.where(Mouse_ToPlot)[0]
x_data = np.copy(unfiltered_relevance[non_nan])
y_data = np.copy((HD_info[non_nan] - randomized_HD_info[non_nan]))
s_data = np.copy(HD_meanvectorlength[non_nan])
n_data = np.array(["%d"%(u+1) for u in non_nan]).astype("str")
red_scatter = axScatterPlot.scatter(x_data, y_data, s=250*s_data, marker="o", edgecolors="red", facecolors="None", linewidth='3', alpha=0.8, label="ADn neurons, M12 (120809)")
for u in np.arange(len(x_data)): axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.0005),fontsize=12)
# Plot non-ADn neurons
non_nan = np.where(Mouse_ToPlot_NonADn)[0]
x_data = np.copy(unfiltered_relevance[non_nan])
y_data = np.copy((HD_info[non_nan] - randomized_HD_info[non_nan]))
s_data = np.copy(HD_meanvectorlength[non_nan])
n_data = np.array(["%d"%(u+1) for u in non_nan]).astype("str")
grey_scatter = axScatterPlot.scatter(x_data, y_data, s=250*s_data, marker="s", edgecolors="grey", facecolors="None", linewidth='3', alpha=0.8, label="Non-ADn neurons, M12 (120806)")
for u in np.arange(len(x_data)): axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.0005),fontsize=12)
# Plot labels and specifics
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional information, $I(s, \theta)$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.set_ylim(bottom=-0.05,top=1.8)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.text(-0.025, 1.05, "A", transform=axScatterPlot.transAxes, fontsize=18, fontweight='bold', va='top')
axScatterPlot.tick_params(labelsize=14)

axScatterPlot = plt.subplot(gs[0,1])
# Plot ADn neurons
non_nan = np.where(Mouse_ToPlot)[0]
x_data = np.copy(unfiltered_relevance[non_nan])
y_data = np.copy(HD_sp[non_nan])
s_data = np.copy(HD_meanvectorlength[non_nan])
n_data = np.array(["%d"%(u+1) for u in non_nan]).astype("str")
red_scatter = axScatterPlot.scatter(x_data, y_data, s=250*s_data, marker="o", edgecolors="red", facecolors="None", linewidth='3', alpha=0.8, label="ADn neurons, M12 (120809)")
for u in np.arange(len(x_data)): axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.0005),fontsize=12)
# Plot non-ADn neurons
non_nan = np.where(Mouse_ToPlot_NonADn)[0]
x_data = np.copy(unfiltered_relevance[non_nan])
y_data = np.copy(HD_sp[non_nan])
s_data = np.copy(HD_meanvectorlength[non_nan])
n_data = np.array(["%d"%(u+1) for u in non_nan]).astype("str")
grey_scatter = axScatterPlot.scatter(x_data, y_data, s=250*s_data, marker="s", edgecolors="grey", facecolors="None", linewidth='3', alpha=0.8, label="Non-ADn neurons, M12 (120806)")
for u in np.arange(len(x_data)): axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.0005),fontsize=12)
# Plot labels and specifics
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'head directional sparsity, $sp_{\theta}$', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.set_ylim(bottom=-0.05,top=1.8)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.text(-0.025, 1.05, "B", transform=axScatterPlot.transAxes, fontsize=18, fontweight='bold', va='top')
axScatterPlot.tick_params(labelsize=14)

non_nan = np.where(Mouse_ToPlot)[0]
gs = gridspec.GridSpecFromSubplotSpec(2, 10, subplot_spec=gs0[1], hspace=0.0)
Rank = iter(np.arange(10))
for neuron_index in non_nan[np.argsort(-unfiltered_relevance[non_nan])[0:10]]:
    r_neuron = next(Rank)
    neuron_name = str("Neuron ")+str(neuron_index + 1)

    axGridMap = plt.subplot(gs[0,r_neuron],projection='polar')
    axGridMap.plot(np.linspace(0, 2*np.pi, 40), HD_tuningcurve[neuron_index], 'o')
    axGridMap.plot(np.linspace(0, 2*np.pi, 360), smoothened_firingrates[neuron_index], '-')
    axGridMap.set_theta_zero_location("N")
    axGridMap.patch.set_facecolor("white")
    axGridMap.grid(True,color="k",alpha=0.4)
    axGridMap.set_title("%s \n (%s) \n $R$=%.3f \n $sp_{\\theta}$ = %.3f"%(neuron_name, cellnames[neuron_index], HD_meanvectorlength[neuron_index], HD_sp[neuron_index]), fontsize=16)
    if r_neuron == 0: axGridMap.text(-0.2, 1.15, "C", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')
    axGridMap.tick_params(labelsize=12)

Rank = iter(np.arange(10))
for neuron_index in non_nan[np.argsort(unfiltered_relevance[non_nan])[0:10]]:
    r_neuron = next(Rank)
    neuron_name = str("Neuron ")+str(neuron_index + 1)

    axGridMap = plt.subplot(gs[1,r_neuron],projection='polar')
    axGridMap.plot(np.linspace(0, 2*np.pi, 40), HD_tuningcurve[neuron_index], 'o')
    axGridMap.plot(np.linspace(0, 2*np.pi, 360), smoothened_firingrates[neuron_index], '-')
    axGridMap.set_theta_zero_location("N")
    axGridMap.patch.set_facecolor("white")
    axGridMap.grid(True,color="k",alpha=0.4)
    axGridMap.set_title("%s \n (%s) \n $R$=%.3f \n $sp_{\\theta}$ = %.3f"%(neuron_name, cellnames[neuron_index], HD_meanvectorlength[neuron_index], HD_sp[neuron_index]), fontsize=16)
    if r_neuron == 0: axGridMap.text(-0.2, 1.15, "D", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')
    axGridMap.tick_params(labelsize=12)

gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[2])

axScatterPlot = plt.subplot(gs[0,0])
# Plot ADn neurons
non_nan = np.where(Mouse_ToPlot)[0]
x_data = np.copy(unfiltered_relevance[non_nan])
y_data = np.copy((spatial_info[non_nan] - randomized_spatial_info[non_nan]))
s_data = np.copy(HD_sp[non_nan])
n_data = np.array(["%d"%(u+1) for u in non_nan]).astype("str")
red_scatter = axScatterPlot.scatter(x_data, y_data, s=250*s_data, marker="o", edgecolors="red", facecolors="None", linewidth='3', alpha=0.8, label="ADn neurons, M12 (120809)")
for u in np.arange(len(x_data)): axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.0005),fontsize=12)
# Plot non-ADn neurons
non_nan = np.where(Mouse_ToPlot_NonADn)[0]
x_data = np.copy(unfiltered_relevance[non_nan])
y_data = np.copy((spatial_info[non_nan] - randomized_spatial_info[non_nan]))
s_data = np.copy(HD_sp[non_nan])
n_data = np.array(["%d"%(u+1) for u in non_nan]).astype("str")
grey_scatter = axScatterPlot.scatter(x_data, y_data, s=250*s_data, marker="s", edgecolors="grey", facecolors="None", linewidth='3', alpha=0.8, label="Non-ADn neurons, M12 (120806)")
for u in np.arange(len(x_data)): axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.0005),fontsize=12)
# Plot labels and specifics
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'spatial information, $I(s, \textrm{\textbf{x}})$ (bits per spike)', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.set_ylim(bottom=-0.05,top=0.4)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.text(-0.025, 1.05, "E", transform=axScatterPlot.transAxes, fontsize=18, fontweight='bold', va='top')
axScatterPlot.tick_params(labelsize=14)

axScatterPlot = plt.subplot(gs[0,1])
# Plot ADn neurons
non_nan = np.where(Mouse_ToPlot)[0]
x_data = np.copy(unfiltered_relevance[non_nan])
y_data = np.copy(spatial_sp[non_nan])
s_data = np.copy(HD_sp[non_nan])
n_data = np.array(["%d"%(u+1) for u in non_nan]).astype("str")
red_scatter = axScatterPlot.scatter(x_data, y_data, s=250*s_data, marker="o", edgecolors="red", facecolors="None", linewidth='3', alpha=0.8, label="ADn neurons, M12 (120809)")
for u in np.arange(len(x_data)): axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.0005),fontsize=12)
# Plot non-ADn neurons
non_nan = np.where(Mouse_ToPlot_NonADn)[0]
x_data = np.copy(unfiltered_relevance[non_nan])
y_data = np.copy(spatial_sp[non_nan])
s_data = np.copy(HD_sp[non_nan])
n_data = np.array(["%d"%(u+1) for u in non_nan]).astype("str")
grey_scatter = axScatterPlot.scatter(x_data, y_data, s=250*s_data, marker="s", edgecolors="grey", facecolors="None", linewidth='3', alpha=0.8, label="Non-ADn neurons, M12 (120806)")
for u in np.arange(len(x_data)): axScatterPlot.annotate(n_data[u],(x_data[u]+0.00008,y_data[u]+0.0005),fontsize=12)
# Plot labels and specifics
axScatterPlot.set_xlabel(r'multiscale relevance, $\mathcal{R}_t$ (Mats$^2$)', fontsize=16)
axScatterPlot.set_ylabel(r'spatial sparsity, $s_{\textrm{\textbf{x}}}$', fontsize=16)
axScatterPlot.legend(loc="upper left", scatterpoints=1, fontsize=18)
axScatterPlot.set_xlim(right=0.305)
axScatterPlot.set_ylim(bottom=-0.05,top=0.4)
axScatterPlot.patch.set_facecolor("white")
axScatterPlot.text(-0.025, 1.05, "F", transform=axScatterPlot.transAxes, fontsize=18, fontweight='bold', va='top')
axScatterPlot.tick_params(labelsize=14)

non_nan = np.where(Mouse_ToPlot)[0]
gs = gridspec.GridSpecFromSubplotSpec(2, 10, subplot_spec=gs0[3], hspace=0.0)
Rank = iter(np.arange(10))
for neuron_index in non_nan[np.argsort(-unfiltered_relevance[non_nan])[0:10]]:
    r_neuron = next(Rank)
    neuron_name = str("Neuron ")+str(neuron_index + 1)

    axGridMap = plt.subplot(gs[0,r_neuron])
    N_bins = 200
    x_t = awake_x_pos[~awake_x_pos.mask]; y_t = awake_y_pos[~awake_x_pos.mask]; spike_trains = spikedata[neuron_index][~awake_x_pos.mask].astype("int")
    occupational_probability, xedges, yedges = np.histogram2d(x_t, y_t, bins=[100,80],range=tight_range)
    spike_map = np.ma.array(np.histogram2d(np.repeat(x_t,spike_trains), np.repeat(y_t,spike_trains),bins=[100,80],range=tight_range)[0])/np.ma.array(occupational_probability)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    blur = gaussian_filter(spike_map, 6.0, mode="reflect", truncate = 4.0)
    heatmap = axGridMap.imshow(100.0*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet)
    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    axGridMap.grid(b=False, which='major')
    axGridMap.grid(b=False, which='minor')
    axGridMap.set_title("%s \n (%s) \n $sp_{\\textrm{\\textbf{x}}}$ = %.4f \n $\overline{\\lambda}$ = %.2f Hz"%(neuron_name, cellnames[neuron_index], spatial_sp[neuron_index], spatial_mean[neuron_index]), fontsize=16)
    if r_neuron == 0: axGridMap.text(-0.2, 1.15, "G", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')
    axGridMap.tick_params(labelsize=12)

Rank = iter(np.arange(10))
for neuron_index in non_nan[np.argsort(unfiltered_relevance[non_nan])[0:10]]:
    r_neuron = next(Rank)
    neuron_name = str("Neuron ")+str(neuron_index + 1)

    axGridMap = plt.subplot(gs[1,r_neuron])
    N_bins = 200
    x_t = awake_x_pos[~awake_x_pos.mask]; y_t = awake_y_pos[~awake_x_pos.mask]; spike_trains = spikedata[neuron_index][~awake_x_pos.mask].astype("int")
    occupational_probability, xedges, yedges = np.histogram2d(x_t, y_t, bins=[100,80],range=tight_range)
    spike_map = np.ma.array(np.histogram2d(np.repeat(x_t,spike_trains), np.repeat(y_t,spike_trains),bins=[100,80],range=tight_range)[0])/np.ma.array(occupational_probability)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    blur = gaussian_filter(spike_map, 6.0, mode="reflect", truncate = 4.0)
    heatmap = axGridMap.imshow(100.0*blur.T,extent=extent,origin='lower',cmap=plt.cm.jet)
    plt.colorbar(heatmap, fraction=0.046, pad=0.04)
    axGridMap.grid(b=False, which='major')
    axGridMap.grid(b=False, which='minor')
    axGridMap.set_title("%s \n (%s) \n $sp_{\\textrm{\\textbf{x}}}$ = %.4f \n $\overline{\\lambda}$ = %.2f Hz"%(neuron_name, cellnames[neuron_index], spatial_sp[neuron_index], spatial_mean[neuron_index]), fontsize=16)
    if r_neuron == 0: axGridMap.text(-0.2, 1.15, "H", transform=axGridMap.transAxes, fontsize=18, fontweight='bold', va='top')
    axGridMap.tick_params(labelsize=12)

plt.savefig("Figures/Figure5.pdf", bbox_inches="tight", dpi=600)

