'''
    This code miscellaneous functions
'''

from __future__ import division
from multiprocessing import Pool
from scipy import optimize, signal, linalg, spatial
import numpy as np


def sensibly_divide(num, denom, masked=False):
    # Get broadcasted views
    num_bc, denom_bc = np.broadcast_arrays(num, denom)
    
    # Get float versions, for exact comparison to 0.0 and nan
    if isinstance(num, np.ma.MaskedArray):
        # Manually broadcast mask
        num_bc_mask, __ = np.broadcast_arrays(np.ma.getmaskarray(num), denom)
        num_bc = np.ma.array(num_bc, mask=num_bc_mask)
        num_bc_float = np.ma.array(num_bc, dtype=np.float_, keep_mask=True)
    else:
        num_bc_float = np.array(num_bc, dtype=np.float_)
    
    if isinstance(denom, np.ma.MaskedArray):
        __, denom_bc_mask = np.broadcast_arrays(num, np.ma.getmaskarray(denom))
        denom_bc = np.ma.array(denom_bc, mask=denom_bc_mask)
        denom_bc_float = np.ma.array(denom_bc, dtype=np.float_, copy=True, keep_mask=True)
    else:
        denom_bc_float = np.array(denom_bc, dtype=np.float_, copy=True)
    
    # Identify potentially problematic locations
    denom_zero = (denom_bc_float == 0.0)
    if np.any(denom_zero):
        num_zero_or_nan = np.logical_or(num_bc_float == 0.0, np.isnan(num_bc_float))
        problems = np.logical_and(denom_zero, num_zero_or_nan)
        if np.any(problems):
            # Either mask the problematic locations, or set them to nan
            if masked:
                denom_bc = np.ma.masked_where(problems, denom_bc)
            else:
                # denom_bc_float is a copy (safe to modify), and float (accepts
                # nan)
                denom_bc = denom_bc_float
                denom_bc[problems] = np.nan
    
    return num_bc / denom_bc
