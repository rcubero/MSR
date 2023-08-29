'''
    This code is used to calculate head directional information, sparsity
    and other head directional quantities
'''

from __future__ import division
from multiprocessing import Pool
import numpy as np
from misc import *

def HD_counts(head_direction, N_bins=40):
    try:
        return np.histogram(head_direction.compressed(), bins=N_bins, range=(0,2.0*np.pi))[0]
    except:
        return np.histogram(head_direction, bins=N_bins, range=(0,2.0*np.pi))[0]

def HD_tuningcurves(N_neurons, spike_trains, head_direction, binning_time, N_bins=40):
    # calculate non-normalized occupational probability
    occupational_probability = HD_counts(head_direction, N_bins)
    denom = binning_time*np.ma.array(occupational_probability, mask=np.isnan(occupational_probability))

    if N_neurons > 1:
        assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
        ratemap = []
        for neuron_index in np.arange(N_neurons):
            try: # do this when the positions are masked arrays
                numer = np.histogram(np.repeat(head_direction,spike_trains[neuron_index].astype("int")).compressed(), bins=N_bins, range=(0,2.0*np.pi))[0]
            except:
                numer = np.histogram(np.repeat(head_direction,spike_trains[neuron_index].astype("int")), bins=N_bins, range=(0,2.0*np.pi))[0]
            ratemap.append(sensibly_divide(numer, denom))

    elif N_neurons == 1:
        try: # do this when the positions are masked arrays
            numer = np.histogram(np.repeat(head_direction,spike_trains.astype("int")).compressed(), bins=N_bins, range=(0,2.0*np.pi))[0]
        except:
            numer = np.histogram(np.repeat(head_direction,spike_trains.astype("int")), bins=N_bins, range=(0,2.0*np.pi))[0]
        ratemap = sensibly_divide(numer, denom)

    else:
        ratemap = np.nan
    return ratemap

def HD_information(N_neurons, spike_trains, head_direction, binning_time, N_bins=40, output_name=None):
    # calculate ratemaps and non-normalized occupational probability
    ratemap = HD_tuningcurves(N_neurons, spike_trains, head_direction, binning_time, N_bins)
    occupational_probability = HD_counts(head_direction, N_bins)
    denom = binning_time*np.ma.array(occupational_probability, mask=np.isnan(occupational_probability))
    
    if N_neurons > 1:
        assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
        HD_info = np.zeros(spike_trains.shape[0])
        for neuron_index in np.arange(N_neurons):
            spike_map = np.ma.array(ratemap[neuron_index], mask=np.isnan(ratemap[neuron_index]))
            ave_spikes = np.ma.sum(spike_map*denom)/np.ma.sum(denom) # the division is for normalization
            HD_info[neuron_index] = (np.ma.sum(spike_map*(np.ma.log(spike_map) - np.log(ave_spikes))*(denom/np.ma.sum(denom))))/(ave_spikes)

    elif N_neurons == 1:
        spike_map = np.ma.array(ratemap, mask=np.isnan(ratemap))
        ave_spikes = np.ma.sum(spike_map*denom)/np.ma.sum(denom) # the division is for normalization
        HD_info = (np.ma.sum(spike_map*(np.ma.log(spike_map) - np.log(ave_spikes))*(denom/np.ma.sum(denom))))/(ave_spikes)

    if output_name is not None:
        np.savetxt(output_name, HD_info)
    
    return HD_info

def HD_sparsity(N_neurons, spike_trains, head_direction, binning_time, N_bins=40, output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
    
    # calculate ratemaps and non-normalized occupational probability
    ratemap = HD_tuningcurves(N_neurons, spike_trains, head_direction, binning_time, N_bins)
    occupational_probability = HD_counts(head_direction, N_bins)
    denom = binning_time*np.ma.array(occupational_probability, mask=np.isnan(occupational_probability))
    
    angular_sparsity = np.zeros(spike_trains.shape[0])
    for neuron_index in np.arange(N_neurons):
        spike_map = np.ma.array(ratemap[neuron_index], mask=np.isnan(ratemap[neuron_index]))
        angular_sparsity[neuron_index] = 1. - np.power(np.ma.sum(denom*spike_map)/np.ma.sum(denom),2)/(np.ma.sum(denom*np.power(spike_map,2))/np.ma.sum(denom))
    
    if output_name is not None:
        np.savetxt(output_name, angular_sparsity)
    
    return angular_sparsity

def mean_vector_length(N_neurons, spike_trains, head_direction, binning_time, N_bins=40, output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"

    HD_meanvectorlength = np.zeros(spike_trains.shape[0])
    for neuron_index in np.arange(N_neurons):
        neuron_angle = np.repeat(head_direction, spike_trains[neuron_index])
        HD_meanvectorlength[neuron_index] = np.sqrt(np.power(np.mean(np.cos(neuron_angle)),2)+np.power(np.mean(np.sin(neuron_angle)),2))

    if output_name is not None:
        np.savetxt(output_name, HD_meanvectorlength)
    
    return HD_meanvectorlength

def randomization_procedure(data):
    randomized_spike, head_direction, binning_time, N_bins = data
    randomized_spike = np.random.permutation(randomized_spike)
    return HD_information(1, randomized_spike, head_direction, binning_time, N_bins)

def randomized_HD_information(N_neurons, spike_trains, head_direction, binning_time, N_bins=40, output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
    unmasked_angle = np.copy(head_direction)

    randomized_HDinfo = np.zeros(spike_trains.shape[0])
    randomized_std_HDinfo = np.zeros(spike_trains.shape[0])
    for n_queue in np.arange(N_neurons):
        randomized_spike = spike_trains[n_queue]
        dump_values = []
        
        input_data = [(randomized_spike, head_direction, binning_time, N_bins) for _ in np.arange(1000)]
        
        pool = Pool()
        res = pool.map_async(randomization_procedure,input_data)
        pool.close(); pool.join()
        dump_values = np.array(res.get())
        del input_data
        
        randomized_HDinfo[n_queue] = np.mean(dump_values)
        randomized_std_HDinfo[n_queue] = np.std(dump_values)

    if output_name is not None:
        np.savetxt(output_name, np.array((randomized_HDinfo, randomized_std_HDinfo)))
    
    return randomized_HDinfo, randomized_std_HDinfo


