#! /usr/bin/env python

from __future__ import print_function, division

import glob
import os
import sys

from astropy.io import fits
from functools import partial
from multiprocessing import Pool, cpu_count

def update_targ(fitsfile, targname):
    with fits.open(fitsfile, mode='update') as f:
        f[0].header.set('TARGNAME', targname)
        f.flush()

def update_all(fitsdir, targname):
    n_cpu = int(cpu_count()/2)
    fitspath = os.path.join(fitsdir, '*fl?.fits')
    fitslist = glob.glob(os.path.join(fitsdir, '*fl?.fits'))
    print('Fits files to update:', fitslist)
    update_with_targname = partial(update_targ, targname=targname)
    with Pool(n_cpu) as p:
        p.map(update_with_targname, fitslist)
    print('Finished updating headers')

if __name__ == '__main__':
    fitsdir = sys.argv[1]
    targname = sys.argv[2]
    update_all(fitsdir, targname)
