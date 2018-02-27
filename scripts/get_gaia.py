#! /usr/bin/env python

import glob
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import numpy as np
import os
import sys

# Astropy packages we'll need
from astropy import units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.visualization import wcsaxes
from astropy.wcs import WCS

# Astroquery packages used for queries
from astroquery.gaia import Gaia

# Drizzle related packages we'll need
# from drizzlepac import tweakreg
# from stsci.tools import teal
# from stwcs import updatewcs

# Other handy parts
from multiprocessing import Pool

#import seaborn as sns
#sns.set(style='white', font_scale=1.2)

# ----------------------------------------------------------------------------------------------------------

def get_footprints(im_name):
    """Calculates positions of the corners of the science extensions of some image 'im_name' in sky space"""
    footprints = []
    hdu = fits.open(im_name)
    
    flt_flag = 'flt.fits' in im_name or 'flc.fits' in im_name
    
    # Loop ensures that each science extension in a file is accounted for.  This is important for 
    # multichip imagers like WFC3/UVIS and ACS/WFC
    for ext in hdu:
        if 'SCI' in ext.name:
            hdr = ext.header
            wcs = WCS(hdr, hdu)
            footprint = wcs.calc_footprint(hdr, undistort=flt_flag)
            footprints.append(footprint)
    
    hdu.close()
    return footprints

# ----------------------------------------------------------------------------------------------------------
def bounds(footprint_list):
    """Calculate RA/Dec bounding box properties from multiple RA/Dec points"""
    
    # flatten list of extensions into numpy array of all corner positions
    merged = [ext for image in footprint_list for ext in image]
    merged = np.vstack(merged)
    ras, decs = merged.T
    
    # Compute width/height
    delta_ra = (max(ras)-min(ras))
    delta_dec = max(decs)-min(decs)

    # Compute midpoints
    ra_midpt = (max(ras)+min(ras))/2.
    dec_midpt = (max(decs)+min(decs))/2.
    

    return ra_midpt, dec_midpt, delta_ra, delta_dec
# ----------------------------------------------------------------------------------------------------------

def plot_footprints(footprint_list, folder, axes_obj=None, fill=True):
    """Plots the footprints of the images on sky space on axes specified by axes_obj """
    
    if axes_obj != None: 
        ax = axes_obj
    
    else: # If no axes passed in, initialize them now
        merged = [ext for image in footprint_list for ext in image] # flatten list of RA/Dec
        merged = np.vstack(merged)
        ras, decs = merged.T
        
        # Calculate aspect ratio
        delta_ra = (max(ras)-min(ras))*np.cos(math.radians(min(np.abs(decs))))
        delta_dec = max(decs)-min(decs)
        aspect_ratio = delta_dec/delta_ra
    
        # Initialize axes
        fig = plt.figure(figsize=[8,8*aspect_ratio])
        ax = fig.add_subplot(111)
        ax.set_xlim([max(ras),min(ras)])
        ax.set_ylim([min(decs),max(decs)])
       
        # Labels
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')
        ax.set_title('Footprint sky projection ({} images)'.format(len(footprint_list)))
        
        ax.grid(ls = ':')
    
        
    colors = cm.rainbow(np.linspace(0, 1, len(footprint_list)))
    alpha = 1./float(len(footprint_list)+1.)+.2
    
    if not fill:
        alpha =.8

    for i, image in enumerate(footprint_list): # Loop over images
        for ext in image: # Loop over extensions in images
            if isinstance(ax, wcsaxes.WCSAxes): # Check axes type
                rect = Polygon(ext, alpha=alpha, closed=True, fill=fill, 
                               color=colors[i], transform=ax.get_transform('icrs'))
            else:
                rect = Polygon(ext, alpha=alpha, closed=True, fill=fill, color=colors[i])

            ax.add_patch(rect)
    
    plt.savefig('{}/footprints.png'.format(folder))

# ----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    folder = sys.argv[1]
    imgpath = '/astro/store/gradscratch/tmp/mdurbin/m33data/fits_tweak/{}/*fl?.fits'.format(folder)
    images = glob.glob(imgpath)
    # footprint_list = list(map(get_footprints, images))

    # If that's slow, here's a version that runs it in parallel:
    # from multiprocessing import Pool
    p = Pool(8)
    footprint_list = list(p.map(get_footprints, images))
    p.close()
    p.join()

    ra_midpt, dec_midpt, delta_ra, delta_dec = bounds(footprint_list)

    coord = SkyCoord(ra=ra_midpt, dec=dec_midpt, unit=u.deg)
    print(coord)

    plot_footprints(footprint_list, folder)

    width = Quantity(delta_ra, u.deg)
    height = Quantity(delta_dec, u.deg)

    # Perform the query!
    r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    ras = r['ra']
    decs = r['dec']
    mags = r['phot_g_mean_mag']
    ra_error = r['ra_error']
    dec_error = r['dec_error']

    fig = plt.figure(figsize=[15,15])

    # Plot RA and Dec positions, color points by G magnitude
    ax1 = fig.add_subplot(221)
    plt.scatter(ras,decs,c=mags,alpha=.5,s=6,vmin=14,vmax=20)
    ax1.set_xlim(max(ras),min(ras))
    ax1.set_ylim(min(decs),max(decs))
    ax1.grid(ls = ':')
    ax1.set_xlabel('RA [deg]')
    ax1.set_ylabel('Dec [deg]')
    ax1.set_title('Source location')
    cb = plt.colorbar()
    cb.set_label('G Magnitude')

    # Plot photometric histogram
    ax2 = fig.add_subplot(222)
    hist, bins, patches = ax2.hist(mags,bins='auto',rwidth=.925)
    ax2.grid(ls = ':')
    ax2.set_xlabel('G Magnitude')
    ax2.set_ylabel('N')
    ax2.set_title('Photometry Histogram')
    ax2.set_yscale("log")


    ax3a = fig.add_subplot(425)
    hist, bins, patches = ax3a.hist(ra_error,bins='auto',rwidth=.9)
    ax3a.grid(ls = ':')
    ax3a.set_title('RA Error Histogram')
    ax3a.set_xlabel('RA Error [mas]')
    ax3a.set_ylabel('N')
    ax3a.set_yscale("log")

    ax3b = fig.add_subplot(427)
    hist, bins, patches = ax3b.hist(dec_error,bins='auto',rwidth=.9)
    ax3b.grid(ls = ':')
    ax3b.set_title('Dec Error Histogram')
    ax3b.set_xlabel('Dec Error [mas]')
    ax3b.set_ylabel('N')
    ax3b.set_yscale("log")


    ax4 = fig.add_subplot(224)
    plt.scatter(ra_error,dec_error,alpha=.2,c=mags,s=1)
    ax4.grid(ls = ':')
    ax4.set_xlabel('RA error [mas]')
    ax4.set_ylabel('Dec error [mas]')
    ax4.set_title('Gaia Error comparison')
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    cb = plt.colorbar()
    cb.set_label('G Magnitude')
    plt.tight_layout()
    plt.savefig('{}/Gaia_errors.png'.format(folder))

    # Filtering the data is also quite easy:

    tbl = Table([ras, decs]) # Make a temporary table of just the positions
    tbl.write('{}/gaia.cat'.format(folder), format='ascii.fast_commented_header')