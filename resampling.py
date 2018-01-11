'''
    This code is used to resample the firing ratemaps of neurons
'''

from __future__ import print_function, division

import numpy as np
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest

# import external dictionaries
from relevance import *
from misc import *

def resample_ratemap(spike_data):
    small_bins = spike_data['binning_time']
    smaller_spike = spike_data['spikes']
    speed_filter = spike_data['speed_filter']
    masked_x_smaller = spike_data['pos_x']
    masked_y_smaller = spike_data['pos_y']
    hd_smaller = spike_data['theta']
    spatial_rate = spike_data['spatial_rate']
    x_edges = spike_data['x_edges']
    y_edges = spike_data['y_edges']
    HD_rate = spike_data['HD_rate']
    xmax, xmin = spike_data['x_limits']
    ymax, ymin = spike_data['y_limits']
    N_bins = spike_data['N_bins']
    th_N_bins = spike_data['HD_bins']
    n = spike_data['n']
    calc_option = spike_data['calc_option']

    unfiltered_spikes = np.copy(smaller_spike)
    smaller_spike[speed_filter] = 0

    Code = [(i,chr(ord('A')+i)) for i in np.arange(25)] + [(i+25,chr(ord('a')+i)) for i in np.arange(25)]
    
    # Get spatial trajectory codes
    x_edges = x_edges[:-1] + np.diff(x_edges)/2.
    y_edges = y_edges[:-1] + np.diff(y_edges)/2.
    x_edges, y_edges = np.meshgrid(x_edges, y_edges, indexing='ij')
    x_edges = x_edges.flatten()
    y_edges = y_edges.flatten()
    
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

    returned_data = []
    
    if calc_option[0]:
        r_trajectory = np.array([ratemap[trajectory_codes[i]]*small_bins for i in np.arange(len(trajectory_codes))])
        synthetic = np.array([np.random.binomial(1, min(r_trajectory[i],1.0), 100) for i in np.arange(len(r_trajectory))])
        synthetic[speed_filter] = 0; synthetic = synthetic.T

        resampled_spikes = []
        for neuron_number in np.arange(synthetic.shape[0]):
            s = [synthetic[neuron_number][n-i::n] for i in np.arange(n)]
            resampled_spikes.append(np.array([sum(r) for r in zip_longest(*s, fillvalue=0)]).astype("int"))
        resampled_spikes = np.array(resampled_spikes)
        
        resampled_relevance = np.zeros(resampled_spikes.shape[0])
        for i in np.arange(resampled_spikes.shape[0]):
            resampled_relevance[i] = parallelized_total_relevance((resampled_spikes.shape[1], resampled_spikes[i]))

        returned_data.append(resampled_relevance)

    if calc_option[1]:
        hd_trajectory = np.array([smoothened_firingrates[hd_trajectory[i]]*small_bins for i in np.arange(len(trajectory_codes))])
        synthetic_hd = np.array([np.random.binomial(1, min(hd_trajectory[i],1.0), 100) for i in np.arange(len(hd_trajectory))])
        synthetic_hd[speed_filter] = 0; synthetic_hd = synthetic_hd.T
    
        resampled_hd_spikes = []

        for neuron_number in np.arange(synthetic_hd.shape[0]):
            s = [synthetic_hd[neuron_number][n-i::n] for i in np.arange(n)]
            resampled_hd_spikes.append(np.array([sum(r) for r in zip_longest(*s, fillvalue=0)]).astype("int"))
        resampled_hd_spikes = np.array(resampled_hd_spikes)
            
        resampled_hd_relevance = np.zeros(resampled_hd_spikes.shape[0])
        for i in np.arange(resampled_spikes.shape[0]):
            resampled_hd_relevance[i] = parallelized_total_relevance((resampled_hd_spikes.shape[1], resampled_hd_spikes[i]))
                
        returned_data.append(resampled_hd_relevance)
            
    if calc_option[2]:
        rhd_trajectory = np.array([ratemap[trajectory_codes[i]]*smoothened_firingrates[hd_trajectory[i]]*small_bins for i in np.arange(len(trajectory_codes))])
        synthetic_rhd = np.array([np.random.binomial(1, min(rhd_trajectory[i],1.0), 100) for i in np.arange(len(rhd_trajectory))])
        synthetic_rhd[speed_filter] = 0; synthetic_rhd = synthetic_rhd.T

        resampled_rhd_spikes = []

        for neuron_number in np.arange(synthetic_rhd.shape[0]):
            s = [synthetic_rhd[neuron_number][n-i::n] for i in np.arange(n)]
            resampled_rhd_spikes.append(np.array([sum(r) for r in zip_longest(*s, fillvalue=0)]).astype("int"))
        resampled_rhd_spikes = np.array(resampled_rhd_spikes)

        resampled_rhd_relevance = np.zeros(resampled_hd_spikes.shape[0])
        for i in np.arange(resampled_spikes.shape[0]):
            resampled_rhd_relevance[i] = parallelized_total_relevance((resampled_rhd_spikes.shape[1], resampled_rhd_spikes[i]))
                
        returned_data.append(resampled_rhd_relevance)

    if len(returned_data)==1: return returned_data[0]
    else: return returned_data
