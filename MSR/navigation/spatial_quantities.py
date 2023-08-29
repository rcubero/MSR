'''
    This code is used to calculate spatial information, sparsity and other spatial quantities
'''

from __future__ import division
from multiprocessing import Pool
from scipy import signal
import numpy as np
from misc import *

def triweight_kernel(N_bins=50, sigma=4.2):
    kernel_x = np.linspace(-75.0,75.0,N_bins+1); kernel_x = np.diff(kernel_x)/2 + kernel_x[:-1]
    kernel_y = np.linspace(-75.0,75.0,N_bins+1); kernel_y = np.diff(kernel_y)/2 + kernel_y[:-1]
    kernel_x, kernel_y = np.meshgrid(kernel_x,kernel_y)
    triweight_kernel = (4.*np.power(1.-(np.power(kernel_x,2)+np.power(kernel_y,2))/(9.*np.power(sigma,2)),3))/(9.*np.pi*np.power(sigma,2))
    support = (np.sqrt(np.power(kernel_x,2)+np.power(kernel_y,2))<3.*sigma).astype("float")
    return triweight_kernel*support

def position_counts(x_t, y_t, N_bins=50, range=((-75.0,75.0),(-75.0,75.0)), kernel=triweight_kernel(N_bins=50, sigma=4.2)):
    try:
        occupational_probability = np.histogram2d(x_t.compressed(), y_t.compressed(), bins=N_bins, range=((-75.0,75.0),(-75.0,75.0)))[0]
    except:
        occupational_probability = np.histogram2d(x_t, y_t, bins=N_bins, range=((-75.0,75.0),(-75.0,75.0)))[0]
    return signal.convolve2d(occupational_probability, kernel, mode="same")

def firing_ratemap(N_neurons, spike_trains, x_t, y_t, binning_time, N_bins=50, range=((-75.0,75.0),(-75.0,75.0)), kernel=triweight_kernel(N_bins=50, sigma=4.2)):
    occupational_probability = position_counts(x_t, y_t, N_bins, range, kernel)
    denom = binning_time*np.ma.array(occupational_probability,mask=np.isnan(occupational_probability))
    
    if N_neurons > 1:
        assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
        ratemap = []
        for neuron_index in np.arange(N_neurons):
            try: # do this when the positions are masked arrays
                spike_map = np.histogram2d(np.repeat(x_t,spike_trains[neuron_index].astype("int")).compressed(), np.repeat(y_t,spike_trains[neuron_index].astype("int")).compressed(), bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))[0]
            except:
                spike_map = np.histogram2d(np.repeat(x_t,spike_trains[neuron_index].astype("int")), np.repeat(y_t,spike_trains[neuron_index].astype("int")), bins=N_bins,range=range)[0]
            numer = signal.convolve2d(spike_map, kernel, mode="same")
            ratemap.append(sensibly_divide(numer, denom))

    elif N_neurons == 1:
        try: # do this when the positions are masked arrays
            spike_map = np.histogram2d(np.repeat(x_t,spike_trains.astype("int")).compressed(), np.repeat(y_t,spike_trains.astype("int")).compressed(), bins=N_bins,range=((-75.0,75.0),(-75.0,75.0)))[0]
        except:
            spike_map = np.histogram2d(np.repeat(x_t,spike_trains.astype("int")), np.repeat(y_t,spike_trains.astype("int")), bins=N_bins,range=range)[0]
        numer = signal.convolve2d(spike_map, kernel, mode="same")
        ratemap = sensibly_divide(numer, denom)

    else:
        print("Check number of neurons")
        ratemap = np.nan

    return ratemap

def spatial_information(N_neurons, spike_trains, x_t, y_t, binning_time, N_bins=50, range=((-75.0,75.0),(-75.0,75.0)), kernel=triweight_kernel(N_bins=50, sigma=4.2), output_name=None):
    # calculate ratemaps and non-normalized occupational probability
    ratemap = firing_ratemap(N_neurons, spike_trains, x_t, y_t, binning_time, N_bins, range, kernel)
    occupational_probability = position_counts(x_t, y_t, N_bins, range, kernel)
    denom = binning_time*np.ma.array(occupational_probability,mask=np.isnan(occupational_probability))
    
    if N_neurons > 1:
        assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
        spatial_info = np.zeros(N_neurons)
        for neuron_index in np.arange(N_neurons):
            spike_map = np.ma.array(ratemap[neuron_index],mask=np.isnan(ratemap[neuron_index]))
            ave_spikes = np.ma.sum(spike_map*denom)/np.ma.sum(denom) # the division is for normalization
            spatial_info[neuron_index] = (np.ma.sum(spike_map*(np.ma.log(spike_map) - np.log(ave_spikes))*(denom/np.ma.sum(denom))))/(ave_spikes)

    elif N_neurons == 1:
        spike_map = np.ma.array(ratemap,mask=np.isnan(ratemap))
        ave_spikes = np.ma.sum(spike_map*denom)/np.ma.sum(denom) # the division is for normalization
        spatial_info = (np.ma.sum(spike_map*(np.ma.log(spike_map) - np.log(ave_spikes))*(denom/np.ma.sum(denom))))/(ave_spikes)

    else:
        print("Check number of neurons")
        spatial_info = np.nan

    if output_name is not None:
        np.savetxt(output_name, spatial_info)

    return spatial_info

def spatial_sparsity(N_neurons, spike_trains, x_t, y_t, binning_time, N_bins=50, range=((-75.0,75.0),(-75.0,75.0)), kernel=triweight_kernel(N_bins=50, sigma=4.2), output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"

    # calculate ratemaps and non-normalized occupational probability
    ratemap = firing_ratemap(N_neurons, spike_trains, x_t, y_t, binning_time, N_bins, range, kernel)
    occupational_probability = position_counts(x_t, y_t, N_bins, range, kernel)
    denom = binning_time*np.ma.array(occupational_probability,mask=np.isnan(occupational_probability))
    
    sp_sparsity = np.zeros(N_neurons)
    for neuron_index in np.arange(N_neurons):
        spike_map = np.ma.array(ratemap[neuron_index],mask=np.isnan(ratemap[neuron_index]))
        sp_sparsity[neuron_index] = 1. - np.power(np.ma.sum(denom*spike_map)/np.ma.sum(denom),2)/(np.ma.sum(denom*np.power(spike_map,2))/np.ma.sum(denom))

    if output_name is not None:
        np.savetxt(output_name, sp_sparsity)

    return sp_sparsity

def spatial_meanspike(N_neurons, spike_trains, x_t, y_t, binning_time, N_bins=50, range=((-75.0,75.0),(-75.0,75.0)), kernel=triweight_kernel(N_bins=50, sigma=4.2), output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"

    # calculate ratemaps and non-normalized occupational probability
    ratemap = firing_ratemap(N_neurons, spike_trains, x_t, y_t, binning_time, N_bins, range, kernel)
    occupational_probability = position_counts(x_t, y_t, N_bins, range, kernel)
    denom = binning_time*np.ma.array(occupational_probability,mask=np.isnan(occupational_probability))
    
    sp_meanspike = np.zeros(N_neurons)
    for neuron_index in np.arange(N_neurons):
        spike_map = np.ma.array(ratemap[neuron_index],mask=np.isnan(ratemap[neuron_index]))
        sp_meanspike[neuron_index] = np.ma.sum(spike_map*denom)/np.ma.sum(denom)

    if output_name is not None:
        np.savetxt(output_name, sp_meanspike)
    
    return sp_meanspike

def randomization_procedure(data):
    randomized_spike, unmasked_x, unmasked_y, binning_time, N_bins, range, kernel = data
    randomized_spike = np.random.permutation(randomized_spike)
    return spatial_information(1, randomized_spike, unmasked_x, unmasked_y, binning_time, N_bins, range, kernel)

def randomized_spatial_information(N_neurons, spike_trains, x_t, y_t, binning_time, N_bins=50, range=((-75.0,75.0),(-75.0,75.0)), kernel=triweight_kernel(N_bins=50, sigma=4.2), output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"

    try:
        unmasked_x = x_t[np.where(~x_t.mask)[0]]
        unmasked_y = y_t[np.where(~y_t.mask)[0]]
    except:
        unmasked_x = np.copy(x_t); unmasked_y = np.copy(y_t);

    randomized_skaggs = np.zeros(N_neurons)
    randomized_std_skaggs = np.zeros(N_neurons)
    for n_queue in np.arange(N_neurons):
        try:
            randomized_spike = spike_trains[n_queue][np.where(~x_t.mask)[0]]
        except:
            randomized_spike = spike_trains[n_queue]

        assert len(unmasked_x) == len(randomized_spike), "Length of spike train do not correspond to length of positions"

        input_data = [(randomized_spike, unmasked_x, unmasked_y, binning_time, N_bins, range, kernel) for _ in np.arange(1000)]

        pool = Pool()
        res = pool.map_async(randomization_procedure,input_data)
        pool.close(); pool.join()
        dump_values = np.array(res.get())
    
        randomized_skaggs[n_queue] = np.mean(dump_values)
        randomized_std_skaggs[n_queue] = np.std(dump_values)

    if output_name is not None:
        np.savetxt(output_name, np.array((randomized_skaggs, randomized_std_skaggs)))
    
    return randomized_skaggs, randomized_std_skaggs

