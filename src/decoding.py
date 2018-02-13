'''
    This code is for decoding position or head direction under a binomial or Poisson neuron firing model.
'''

from __future__ import division
from multiprocessing import Pool
import numpy as np
import math


# ------------------------------------ #
#      Binomial neuron assumption      #
# ------------------------------------ #
def binomial_distribution(spike, p_spike):
    return np.log(np.power(p_spike,spike)*np.power(1.-p_spike,1.-spike))

def multinomial_distribution(spikes, p_spike, p_occupation):
    return np.sum([binomial_distribution(spikes[i], p_spike[i]) for i in np.arange(len(spikes))], axis=0) + np.log(p_occupation)

# --------- SPATIAL DECODING --------- #
def binomial_decoding(input_data):
    spikes, stacked_ratemap, p_occupation, x_true, y_true, x_bins, y_bins, time_index = input_data
    argument = multinomial_distribution(spikes, stacked_ratemap, p_occupation)
    
    probability = np.exp(argument); probability[probability<=1e-100] = 0.;
    normalizer = np.sum(probability)
    probability = probability/normalizer; argument = argument - np.log(normalizer)
    
    max_ind = np.where(probability==np.nanmax(probability))[0]
    
    average_error = np.sum(probability*(np.sqrt(np.power(x_true - x_bins,2)+np.power(y_true - y_bins,2))))
    entropy = -np.sum(probability*argument)
    distance_error = [np.sqrt(np.power(x_true - x_bins[index],2)+np.power(y_true - y_bins[index],2)) for index in max_ind]
    
    if len(max_ind)==1: distance_index = max_ind[0]
    else: distance_index = max_ind[np.argmax(distance_error)]
    
    return np.mean(distance_error), average_error, entropy, time_index, np.sum(spikes), distance_index, len(max_ind)

def parallelized_binomial_decoding(list_of_neurons, firing_map, neuron_spikes, number_of_bins, p_occupation,
                                   x_true, y_true, mask_on_position, x_bins, y_bins):
    input_data = []
    for i in np.arange(len(mask_on_position)):
        if ~mask_on_position[i]:
            spikes = neuron_spikes[np.sort(list_of_neurons),i]
            ratemap_stacked = [firing_map[neuron_index] for neuron_index in np.sort(list_of_neurons)]
            if np.sum(spikes)>0:
                input_data.append([spikes, ratemap_stacked, p_occupation, x_true[i], y_true[i], x_bins, y_bins, i])
    
    pool = Pool()
    res = pool.map_async(binomial_decoding,input_data)
    pool.close(); pool.join() # not optimal step but is safe to do
    data = np.array(res.get())
    return data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6]


# ------------------------------------ #
#      Poisson neuron assumption       #
# ------------------------------------ #
def poisson_distribution(spike, p_spike):
    prob = np.power(p_spike, spike)*np.exp(-p_spike)/math.factorial(spike)
    prob[prob<=1e-100] = 1e-100
    return np.log(prob)

def multipoisson_distribution(spikes, p_spike, p_occupation):
    poisson_prob = [poisson_distribution(spikes[i], p_spike[i]) for i in np.arange(len(spikes))]
    return np.sum(poisson_prob, axis=0) + np.log(p_occupation)

# --------- SPATIAL DECODING --------- #
def poisson_decoding(input_data):
    spikes, stacked_ratemap, p_occupation, dt, x_true, y_true, x_before, y_before, speed, x_bins, y_bins, time_index = input_data
    argument = multipoisson_distribution(spikes, stacked_ratemap, p_occupation)
    
    # smoothing prior
    smoothing_prior = np.exp(-( np.power(x_before - x_bins,2) + np.power(y_before - y_bins,2))/(2.*np.power(7.0*dt*speed,2)))
    smoothing_prior[smoothing_prior==0] = 1e-100
    argument = argument + np.log(smoothing_prior)
    
    probability = np.exp(argument); probability[probability<=1e-100] = 0
    normalizer = np.sum(probability)
    probability = probability/float(normalizer)
    argument = argument - np.log(normalizer)
    
    max_ind = np.where(argument==np.nanmax(argument))[0]
    
    average_error = np.sum(probability*(np.sqrt(np.power(x_true - x_bins,2)+np.power(y_true - y_bins,2))))
    entropy = -np.sum(probability*argument)
    distance_error = [np.sqrt(np.power(x_true - x_bins[index],2)+np.power(y_true - y_bins[index],2)) for index in max_ind]
    
    return np.mean(distance_error) , average_error, entropy, time_index, np.sum(spikes)

def parallelized_poisson_decoding(list_of_neurons, firing_map, neuron_spikes, number_of_bins, p_occupation, dt,
                                  x_true, y_true, x_before, y_before, speed, mask_on_position, x_bins, y_bins):
    input_data = []
    for i in np.arange(len(mask_on_position)):
        if ~mask_on_position[i]:
            spikes = neuron_spikes[np.sort(list_of_neurons),i]
            ratemap_stacked = [firing_map[neuron_index] for neuron_index in np.sort(list_of_neurons)]
            if np.sum(spikes.astype("bool").astype("int"))>0:
                input_data.append([spikes, ratemap_stacked, p_occupation, dt, x_true[i], y_true[i],
                                   x_before[i], y_before[i], speed[i], x_bins, y_bins, i])
    
    pool = Pool()
    res = pool.map_async(poisson_decoding,input_data)
    pool.close(); pool.join() # not optimal step but is safe to do
    data = np.array(res.get())
    return data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]

# --------- HEAD DIRECTIONAL DECODING --------- #
def HD_poisson_decoding(input_data):
    spikes, stacked_ratemap, p_occupation, theta_true, theta_bins, time_index = input_data
    argument = multipoisson_distribution(spikes, stacked_ratemap, p_occupation)
    
    max_ind = np.where(argument==np.nanmax(argument))[0]
    distance_error = [np.abs(np.arctan2(np.sin(theta_true-theta_bins[index]),np.cos(theta_true-theta_bins[index]))) for index in max_ind]
    return np.mean(distance_error)

def HD_parallelized_poisson_decoding(list_of_neurons, firing_map, neuron_spikes, p_occupation, theta_true, nondecodable_times, theta_bins):
    input_data = []
    for i in np.arange(len(theta_true)):
        if i not in nondecodable_times:
            spikes = neuron_spikes[np.sort(list_of_neurons),i]
            ratemap_stacked = [firing_map[neuron_index] for neuron_index in np.sort(list_of_neurons)]
            if np.sum(spikes.astype("bool").astype("int"))>0:
                input_data.append([spikes, ratemap_stacked, p_occupation, theta_true[i], theta_bins, i])
    
    pool = Pool()
    res = pool.map_async(HD_poisson_decoding,input_data)
    pool.close(); pool.join() # not optimal step but is safe to do
    return np.array(res.get())
