#! /usr/bin/env python

"""Sets the 'TARGNAME' header keyword for all FITS files in a directory.

Authors
-------
    Meredith Durbin, March 2018

Use
---
    This script is intended to be executed from the command line as
    such:
    ::
        python update_targname.py ['fitsdir'] ['targname']
    
    Parameters:
    (Required) [fitsdir] - Directory containing FITS files to update.
    (Required) [targname] - Desired target name.
"""

from __future__ import print_function, division

import glob
import os
import sys

from astropy.io import fits
from functools import partial
from multiprocessing import Pool, cpu_count

def update_targ(fitsfile, targname):
    """Insert or update TARGNAME keyword in a single FITS file.

    Inputs
    ------
    fitsfile : str
        Path to FITS file to modify
    targname : str
        Desired target name

    Returns
    -------
    Nothing
    """
    with fits.open(fitsfile, mode='update') as f:
        f[0].header.set('TARGNAME', targname)
        f.flush()

def update_all(fitsdir, targname):
    """Parallelizes update_targ over all FITS files in a directory

    Inputs
    ------
    fitsdir : str
        Directory of FITS files
    targname : str
        Desired target name

    Returns
    -------
    Nothing
    """
    n_cpu = int(cpu_count()/2)
    fitspath = os.path.join(fitsdir, '*.fits')
    fitslist = glob.glob(os.path.join(fitsdir, '*.fits'))
    print('Fits files to update:', fitslist)
    update_with_targname = partial(update_targ, targname=targname)
    p = Pool(n_cpu)
    p.map(update_with_targname, fitslist)
    p.close()
    p.join()
    print('Finished updating headers')

if __name__ == '__main__':
    fitsdir = sys.argv[1]
    targname = sys.argv[2]
    update_all(fitsdir, targname)
