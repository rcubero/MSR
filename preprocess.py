'''
    This code is used to preprocess the spike times and position data.
    Some codes were adapted from https://github.com/danielwe/gridcell
'''

from __future__ import division
from multiprocessing import Pool
from scipy import optimize, signal, linalg, spatial
import numpy as np
from misc import *


def func_bin_spikes(data):
    spike_time, time_step = data
    return 1*np.histogram(spike_time, time_step)[0].astype('int')

def binning(size, spike_times, print_timestep, t_start=None, t_stop=None):
    if t_start is None:
        t_start = 0
    if t_stop is None:
        t_stop = max([spike_times[i].max() for i in range(len(spike_times))])
    
    time_step = np.arange(t_start, t_stop+size, size)

    # check if the binned times are indeed within the indicated start and end times
    if time_step[-2] > t_stop:
        time_step = time_step[:-1]
    if time_step[1] < t_start:
        time_step = time_step[1:]
    time_step[0] = t_start
    time_step[-1] = t_stop
    
    input_data = [(spike_times[i], time_step) for i in np.arange(len(spike_times))]
    pool = Pool()
    res = pool.map_async(func_bin_spikes, input_data)
    pool.close(); pool.join()
    binned_spikes = np.asarray(res.get())
    
    if print_timestep: return binned_spikes, time_step
    else: return binned_spikes

def transform(x, y, range_=None, translate=False, rotate=False):
    info = {'rotation_angle': 0.0, 'rotation_anchor': (0.0, 0.0),
        'translation_vector': (0.0, 0.0),
            'scaling_factor': 1.0, 'scaling_anchor': (0.0, 0.0)}
    if range is None:
        xo, yo = 0.0, 0.0
    else:
        xo, yo = _midpoint(range_[0]), _midpoint(range_[1])
    if rotate:
        angle = -_tilt(x, y)
        x, y = _rotate(x, y, angle)
        new_xc, new_yc = _midpoint(x), _midpoint(y)
        old_xc, old_yc = _rotate(new_xc, new_yc, -angle)
        x, y = x - new_xc + old_xc, y - new_yc + old_yc
        info['rotation_angle'] = angle
        info['rotation_anchor'] = (old_xc, old_yc)
    if translate:
        xc, yc = _midpoint(x), _midpoint(y)
        xtrans, ytrans = xo - xc, yo - yc
        x = x + xtrans
        y = y + ytrans
        info['translation_vector'] = (xtrans, ytrans)
    if range_ is not None:
        dxo, dyo = x - xo, y - yo
        candidate_factors = np.array([(range_[0][0] - xo) / np.nanmin(dxo),
                                      (range_[1][0] - yo) / np.nanmin(dyo),
                                      (range_[0][1] - xo) / np.nanmax(dxo),
                                      (range_[1][1] - yo) / np.nanmax(dyo)],)
        scaling_factor = np.amin(candidate_factors[candidate_factors >= 0.0])
                                      
        dxo *= scaling_factor
        dyo *= scaling_factor
        x, y = xo + dxo, yo + dyo
        info['scaling_factor'] = scaling_factor
        info['scaling_anchor'] = (xo, yo)
    return x, y, info

def _tilt(x, y):
    def _bbox_area(tilt):
        rot_x, rot_y = _rotate(x, y, -tilt)
        dx, dy = (np.ptp(rot_x[~np.isnan(rot_x)]), np.ptp(rot_y[~np.isnan(rot_y)]))
        return dx * dy
    pi_4 = .25 * np.pi
    tilt = optimize.minimize_scalar(_bbox_area, bounds=(-pi_4, pi_4), method='bounded').x
    return tilt

def _rotate(x, y, angle):
    rot_x = x * np.cos(angle) + y * np.sin(angle)
    rot_y = y * np.cos(angle) - x * np.sin(angle)
    return rot_x, rot_y

def _midpoint(x):
    return 0.5 * (np.nanmin(x) + np.nanmax(x))

def time_and_distance_weights(t, x, y):
    tsteps = np.diff(t)
    tweights = 0.5 * np.hstack((tsteps[0], tsteps[:-1] + tsteps[1:], tsteps[-1]))
    xsteps = np.ma.diff(x)
    ysteps = np.ma.diff(y)
    dsteps = np.ma.sqrt(xsteps * xsteps + ysteps * ysteps)
    dweights = 0.5 * np.ma.hstack((dsteps[0], dsteps[:-1] + dsteps[1:],dsteps[-1]))
    return tweights, dweights

def calculate_speed(t, x, y, speed_window):
    tweights, dweights = time_and_distance_weights(t, x, y)
    
    if not speed_window >= 0.0:
        raise ValueError("'speed_window' must be a non-negative number")
    
    mean_tstep = np.mean(np.diff(t))
    window_length = 2 * int(0.5 * speed_window / mean_tstep) + 1
    window_sequence = np.empty(window_length)
    window_sequence.fill(1.0 / window_length)
    
    dw_mask = np.ma.getmaskarray(dweights)
    dw_filled = np.ma.filled(dweights, fill_value=0.0)
    
    tweights_filt = sensibly_divide(signal.convolve(tweights, window_sequence, mode='same'),
                                    signal.convolve(np.ones_like(tweights), window_sequence,mode='same'))
    dweights_filt = sensibly_divide(signal.convolve(dw_filled, window_sequence, mode='same'),
                                    signal.convolve((~dw_mask).astype(np.float_),window_sequence, mode='same'), masked=True)
                                    
    return dweights_filt / tweights_filt

def calculate_movement_direction(x, y):
    xsteps = np.ma.diff(x)
    ysteps = np.ma.diff(y)
    xweights = 0.5 * np.ma.hstack((xsteps[0], xsteps[:-1] + xsteps[1:], xsteps[-1]))
    yweights = 0.5 * np.ma.hstack((ysteps[0], ysteps[:-1] + ysteps[1:], ysteps[-1]))
    return np.arctan2(yweights,xweights) + np.pi

def calculate_head_direction(x_1, x_2, y_1, y_2):
    xweights = x_1 - x_2
    yweights = y_1 - y_2
    return np.arctan2(-xweights,yweights) + np.pi

def local_angle_interpolation(angle_data, time_data, time, ind0, ind1):
    theta_0 = angle_data[ind0]
    theta_1 = angle_data[ind1]
    
    if(isnan(theta_0) or isnan(theta_1)):
        return np.NAN # local interpolation not possible
    if(theta_1 - theta_0 > pi):
        theta_1 -= 2.*pi
    if(theta_1 - theta_0 < -pi):
        theta_1 += 2.*pi

    time_0 = time_data[ind0]
    time_1 = time_data[ind1]
    angle = theta_0 + (time - time_0)*(theta_1 - theta_0)/(time_1 - time_0)

    if(angle < 0.): angle += 2.*pi
    if(angle > 2.*pi): angle -= 2.*pi

    return angle
