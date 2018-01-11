'''
    This code is used to calculate speed information, sparsity and other speed quantities
'''

from __future__ import division
from multiprocessing import Pool
import numpy as np

def speed_counts(speed_data, N_bins=20, range=(2,35)):
    return np.histogram(speed_data, bins=N_bins, range=range)[0]

def speed_tuningcurves(N_neurons, spike_trains, speed_data, binning_time, N_bins=20, range=(2,35)):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
    
    # calculate non-normalized occupational probability
    occupational_probability = speed_counts(speed_data, N_bins, range)
    denom = binning_time*occupational_probability
    
    ratemap = []
    for neuron_index in np.arange(N_neurons):
        try: # do this when the positions are masked arrays
            numer = np.histogram(np.repeat(speed_data,spike_trains[neuron_index].astype("int")).compressed(), bins=N_bins, range=range)[0]
        except:
            numer = np.histogram(np.repeat(speed_data,spike_trains[neuron_index].astype("int")), bins=N_bins, range=range)[0]
        ratemap.append(sensibly_divide(numer, denom))
    return ratemap

def speed_information(N_neurons, spike_trains, speed_data, binning_time, N_bins=20, range=(2,35), output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
    
    # calculate ratemaps and non-normalized occupational probability
    ratemap = speed_tuningcurves(N_neurons, spike_trains, speed_data, binning_time, N_bins, range)
    occupational_probability = speed_counts(speed_data, N_bins, range)
    denom = binning_time*occupational_probability
    
    speed_info = np.zeros(spike_trains.shape[0])
    for neuron_index in np.arange(N_neurons):
        ave_spikes = np.sum(ratemap[neuron_index]*denom)/np.ma.sum(denom) # the division is for normalization
        speed_info[neuron_index] = (np.ma.sum(ratemap[neuron_index]*(np.ma.log(ratemap[neuron_index]) - np.log(ave_spikes))*(denom/np.ma.sum(denom))))/np.ma.sum(ave_spikes)
    
    if output_name is not None:
        np.savetxt(output_name, speed_info)
    
    return speed_info


def speed_sparsity(N_neurons, spike_trains, speed_data, binning_time, N_bins=20, range=(2,35), output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
    
    # calculate ratemaps and non-normalized occupational probability
    ratemap = speed_tuningcurves(N_neurons, spike_trains, speed_data, binning_time, N_bins, range)
    occupational_probability = speed_counts(speed_data, N_bins, range)
    denom = binning_time*occupational_probability
    
    speedsparsity = np.zeros(spike_trains.shape[0])
    for neuron_index in np.arange(N_neurons):
        speedsparsity[neuron_index] = 1. - np.power(np.sum(denom*ratemap[neuron_index])/np.ma.sum(denom),2)/(np.sum(denom*np.power(ratemap[neuron_index],2))/np.ma.sum(denom))
    
    if output_name is not None:
        np.savetxt(output_name, speedsparsity)
    
    return speedsparsity


def randomized_speed_information(N_neurons, spike_trains, speed_data, binning_time, N_bins=20, range=(2,35), output_name=None):
    assert N_neurons == spike_trains.shape[0], "Number of spike trains do not correspond to number of neurons"
    
    unmasked_angle = np.copy(speed_data)
    
    randomized_speedinfo = np.zeros(spike_trains.shape[0])
    randomized_std_speedinfo = np.zeros(spike_trains.shape[0])
    for n_queue in np.arange(N_neurons):
        randomized_spike = spike_trains[n_queue]
        dump_values = []
        
        input_data = [(randomized_spike, speed_data, binning_time, N_bins, range) for _ in np.arange(1000)]
        
        def randomization_procedure(data):
            randomized_spike, speed_data, binning_time, N_bins, range = data
            randomized_spike = np.random.permutation(randomized_spike)
            return speed_information(1, randomized_spike, speed_data, binning_time, N_bins, range)
        
        pool = Pool()
        res = pool.map_async(follow_curve,input_data)
        pool.close(); pool.join()
        dump_values = np.array(res.get())
        del input_data
        
        randomized_speedinfo[n_queue] = np.mean(dump_values)
        randomized_std_speedinfo[n_queue] = np.std(dump_values)
    
    if output_name is not None:
        np.savetxt(output_name, np.array((randomized_speedinfo, randomized_std_speedinfo)))
    
    return randomized_speedinfo, randomized_std_speedinfo


def speed_score():
    pass
