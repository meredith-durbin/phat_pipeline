#! /usr/bin/env python

from __future__ import print_function, division

import glob
import os
import sys

# Drizzle related packages we'll need
from drizzlepac import tweakreg
from stsci.tools import teal
from stwcs import updatewcs

# Other handy parts
from multiprocessing import Pool

if __name__ == '__main__':
    detector = sys.argv[1]
    cw = 3.5
    if detector == 'acs':
        input_images = sorted(glob.glob('j*flc.fits'))
        threshold = 200.
    elif detector == 'uvis':
        input_images = sorted(glob.glob('i*flc.fits'))
        threshold = 100.
    elif detector == 'ir':
        input_images = sorted(glob.glob('i*flt.fits'))
        threshold = 20.
        cw = 2.5

    # Parallelized option
    p = Pool(8)
    derp = p.map(updatewcs.updatewcs, input_images)
    p.close()
    p.join()

    cat = 'gaia.cat'
    wcsname ='GAIA'
    teal.unlearn('tweakreg')
    teal.unlearn('imagefindpars')

    tweakreg.TweakReg(input_images, # Pass input images
        updatehdr=True, # update header with new WCS solution
        imagefindcfg={'threshold':threshold,'conv_width':cw},# Detection parameters, threshold varies for different data
        separation=0., # Allow for very small shifts
        refcat=cat, # Use user supplied catalog (Gaia)
        clean=True, # Get rid of intermediate files
        interactive=False,
        see2dplot=True,
        shiftfile=True, # Save out shift file (so we can look at shifts later)
        wcsname=wcsname, # Give our WCS a new name
        reusename=True,
        fitgeometry='general') # Use the 6 parameter fit
    os.rename('shifts.txt','shifts_{}.txt'.format(detector))
    os.rename('shifts_wcs.fits','shifts_{}_wcs.fits'.format(detector))
