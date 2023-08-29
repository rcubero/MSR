'''
    This code is used for calculating the multiscale relevance.
'''

from __future__ import division

from collections import Counter
from multiprocessing import Pool
import numpy as np

# these are functions needed to calculate histograms
def _iter_baskets_contiguous(items, maxbaskets=3, item_count=None):
    '''
        generates balanced baskets from iterable, contiguous contents
        provide item_count if providing a iterator that doesn't support len()
        Taken from: http://stackoverflow.com/a/21767522
        '''
    item_count = item_count or len(items)
    baskets = min(item_count, maxbaskets)
    items = iter(items)
    floor = item_count // baskets
    ceiling = floor + 1
    stepdown = item_count % baskets
    for x_i in range(baskets):
        length = ceiling if x_i < stepdown else floor
        yield [next(items) for _ in range(length)]

# these are the functions that calculate the H[K] and the area under the curve
def _calculate_HofKS(mapping_ks):
    ks_counts = np.asarray(Counter(mapping_ks).most_common())
    positive_values = np.where(ks_counts[:,0]>0)[0]
    kq, mq = ks_counts[:,0][positive_values], ks_counts[:,1][positive_values]
    assert np.sum(kq*mq)==np.sum(mapping_ks)
    M = float(np.sum(kq*mq))
    return -np.sum(((kq*mq)/M)*np.log2((kq*mq)/M))/np.log2(M), -np.sum(((kq*mq)/M)*np.log2(kq/M))/np.log2(M)

def _calculate_area(data_points):
    # calculates integral using the trapezoid rule
    return np.sum(0.5*(np.abs(data_points[:,0][:-1]+data_points[:,0][1:]))*(np.abs(data_points[:,1][:-1]-data_points[:,1][1:])))

# these are the parallelized version of calculating total relevances
def follow_curve(data):
    total_time, spikes, partitions = data
    time_bins = np.asarray(list(_iter_baskets_contiguous(np.arange(total_time), partitions)))
    ks_map = np.array([np.sum(spikes[time_bins[i]]) for i in np.arange(len(time_bins))])
    return _calculate_HofKS(ks_map)

def parallelized_total_relevance(zipped_data):
    total_time, spikes = zipped_data
    N_max = np.round(np.log(total_time - (0.01*total_time))/np.log(10), 2);
    N_parts = np.unique(np.logspace(0.4,N_max,100).astype("int"))
    N_partitions = np.append(N_parts,[spikes.size])
    input_data = [(total_time, spikes, i) for i in N_partitions]
    pool = Pool()
    res = pool.map_async(follow_curve,input_data)
    pool.close(); pool.join() # not optimal step but is safe to do
    data = np.array(res.get())
    data = np.append(data,np.array([[0.0, 0.0], [0.0, 1.0]]),axis=0)
    data = data[np.lexsort((data[:,0],data[:,1]))]
    return _calculate_area(data)


