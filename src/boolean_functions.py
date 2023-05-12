"""
 This code is to build boolean functions with all possible pairs
 of neurons in the mEC and calculates its multiscale relevance.
"""

# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import os
import sys

# some additional packages that are needed
from scipy import io, signal
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.stats import pearsonr, spearmanr
from multiprocessing import Pool
from itertools import combinations

# import plotting-related packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

# import external dictionaries
from preprocess import *
from relevance import *
from spatial_quantities import *
from HD_quantities import *

# Load spike train data
filenames = [ fname.rstrip('\n') for fname in open(os.path.join('Flekken_Data', 'cell_filenames')) ]
spike_times = [ io.loadmat(os.path.join('Flekken_Data', filenames[i]), squeeze_me=True)['cellTS']
               for i in np.arange(len(filenames)) ]

# Binarize spike train data
binning_time = 1e-3
unfiltered_spike_trains, time_bins = binning(binning_time, spike_times, True)


results = {}

pair_list = np.arange(len(unfiltered_spike_trains))
results['neuron_list'] = pair_list

unfiltered_relevance = np.zeros(len(pair_list))
N_spins = np.zeros(len(pair_list))

for pair_index in np.arange(len(pair_list)):
    boolean_function = unfiltered_spike_trains[pair_index].astype('bool').astype('int')
    unfiltered_relevance[pair_index] = parallelized_total_relevance((len(boolean_function), boolean_function))
    N_spins[pair_index] = np.sum(boolean_function)

results['relevance'] = unfiltered_relevance
results['nspikes'] = N_spins

#io.savemat('%s.mat'%("mEC_booleanfunction_NAND_1ms.d"), results)
#io.savemat('%s.mat'%("mEC_booleanfunction_NOR_1ms.d"), results)
io.savemat('%s.mat'%("mEC_singleneuron_AND_1ms"), results)
