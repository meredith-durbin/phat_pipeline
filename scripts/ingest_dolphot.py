#! /usr/bin/env python

"""Ingests raw DOLPHOT output (an unzipped .phot file) and converts it
to a dataframe, which is then optionally written to an HDF5 file.
Column names are read in from the accompanying .phot.columns file.

Authors
-------
    Meredith Durbin, February 2018

Use
---
    This script is inteneded to be executed from the command line as
    such:
    ::
        python ingest_dolphot.py ['filebase'] ['--to_hdf'] ['--full']
    Parameters:
    (Required) [filebase] - Path to .phot file, with or without .phot extension.
    (Optional) [--to_hdf] - Whether to write the dataframe to an HDF5 
    file. Default is True.
    (Optional) [--full] - Whether to use the full set of columns (photometry of 
    individual exposures). Default is False.
"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import traceback

from astropy.io import fits
from pathlib import Path

# global photometry values
# these are the first 11 columns in the raw dolphot output
global_columns = ['ext','chip','x','y','chi_gl','snr_gl','sharp_gl', 
                  'round_gl','majax_gl','crowd_gl','objtype_gl']

# dictionary mapping text in .columns file to column suffix
colname_mappings = {
    'counts,'                            : 'count',
    'sky level,'                         : 'sky',
    'Normalized count rate,'             : 'rate',
    'Normalized count rate uncertainty,' : 'raterr',
    'Instrumental VEGAMAG magnitude,'    : 'vega',
    'Transformed UBVRI magnitude,'       : 'trans',
    'Magnitude uncertainty,'             : 'err',
    'Chi,'                               : 'chi',
    'Signal-to-noise,'                   : 'snr',
    'Sharpness,'                         : 'sharp',
    'Roundness,'                         : 'round',
    'Crowding,'                          : 'crowd',
    'Photometry quality flag,'           : 'flag',
}

def cull_photometry(df, filter_detectors, snrcut=4.,
                    cut_params={'irsharp':0.15,'uvissharp':0.15,'wfcsharp':0.20,
                                'ircrowd':2.25,'uviscrowd':1.30,'wfccrowd':2.25}):
    """Add 'ST' and 'GST' flag columns based on stellar parameters.

    Inputs
    ------
    df : DataFrame 
        table read in by read_dolphot
    filter_detectors : list of strings
        list of detectors + filters in 'detector-filter' format
    snrcut : scalar, optional
        minimum signal-to-noise ratio for a star to be flagged as 'ST'
        default: 4.0
    cut_params : dict
        dictionary of parameters to cut on, with keys in '<detector><quantity>'
        format and scalar values.

    Returns
    -------
    df : DataFrame
        table read in by read_dolphot, with ST and GST columns added
        (name format: '<filter>_(g)st')
    """
    for filt in filter_detectors:
        try:
            d, f = filt.lower().split('-')
            print('Making ST and GST cuts for {}'.format(f))
            snr_condition = df.loc[:,'{}_snr'.format(f)] > snrcut
            sharp_condition = df.loc[:,'{}_sharp'.format(f)]**2 < cut_params['{}sharp'.format(d)]
            crowd_condition = df.loc[:,'{}_crowd'.format(f)] < cut_params['{}crowd'.format(d)]
            df.loc[:,'{}_st'.format(f)] = (snr_condition & sharp_condition).astype(bool)
            df.loc[:,'{}_gst'.format(f)] = (snr_condition & sharp_condition & crowd_condition).astype(bool)
            print('Found {} out of {} stars meeting ST criteria for {}'.format(df.loc[:,'{}_st'.format(f)].sum(), df.shape[0], f))
            print('Found {} out of {} stars meeting GST criteria for {}'.format(df.loc[:,'{}_gst'.format(f)].sum(), df.shape[0], f))
        except Exception:
            print('Could not perform culling for {}.\n{}'.format(f, traceback.format_exc()))
    return df

def make_header_table(fitsdir, search_string='*fl?.chip?.fit*'):
    """Construct a table of key-value pairs from FITS headers of images
    used in dolphot run. Columns are the set of all keywords that appear
    in any header, and rows are per image.

    Inputs
    ------
    fitsdir : Path 
        directory of FITS files
    search_string : string or regex patter, optional
        string to search for FITS images with. Default is
        '*fl?.chip?.fit*'

    Returns
    -------
    df : DataFrame
        A table of header key-value pairs indexed by image name.
    """
    keys = []
    headers = {}
    fitslist = list(fitsdir.glob(search_string))
    if len(fitslist) == 0:
        print('No fits files found in {}!'.format(fitsdir))
        return pd.DataFrame()
    for fitsfile in fitslist:
        fitsname = fitsfile.name
        head = fits.getheader(fitsfile)
        headers.update({fitsname:head})
        keys += [k for k in head.keys()]
    unique_keys = np.unique(keys).tolist()
    remove_keys = ['COMMENT', 'HISTORY', '']
    for key in remove_keys:
        if key in unique_keys:
            unique_keys.remove(key)
    df = pd.DataFrame(columns=unique_keys)
    for fitsname, head in headers.items():
        row = pd.Series(dict(head.items()))
        df.loc[fitsname.split('.fits')[0]] = row.T
    df_obj = df.select_dtypes('object')
    for c in df_obj:
        dtype = pd.api.types.infer_dtype(df[c], skipna=True)
        if dtype == 'string':
            df.loc[:,c] = df.loc[:,c].astype(str)
        elif dtype in ['float','mixed-integer-float']:
            df.loc[:,c] = df.loc[:,c].astype(float)
        elif dtype == 'integer':
            df.loc[:,c] = df.loc[:,c].astype(int)
        elif dtype == 'boolean':
            df.loc[:,c] = df.loc[:,c].astype(bool)
        else:
            print('Unrecognized datatype for column {}: {}; converting to string'.format(c, dtype))
            df.loc[:,c] = df.loc[:,c].astype(str)
    return df

def name_columns(colfile):
    """Construct a table of column names for dolphot output, with indices
    corresponding to the column number in dolphot output file.

    Inputs
    ------
    colfile : path
        path to file containing dolphot column descriptions

    Returns
    -------
    df : DataFrame
        A table of column descriptions and their corresponding names.
    filters : list
        List of filters included in output
    """
    df = pd.DataFrame(data=np.loadtxt(colfile, delimiter='. ', dtype=str),
                          columns=['index','desc']).drop('index', axis=1)
    df = df.assign(colnames='')
    df.loc[:10,'colnames'] = global_columns
    for k, v in colname_mappings.items():
        indices = df[df.desc.str.find(k) != -1].index
        desc_split = df.loc[indices,'desc'].str.split(", ")
        indices_total = indices[desc_split.str.len() == 2]
        indices_indiv = indices[desc_split.str.len() > 2]
        filters = desc_split.loc[indices_total].str[-1]
        imgnames = desc_split.loc[indices_indiv].str[1].str.split(' ').str[0]
        df.loc[indices_total,'colnames'] = filters.str.lower() + '_' + v.lower()
        df.loc[indices_indiv,'colnames'] = imgnames + '_' + v.lower()
    filters = df.desc[df.desc.str.endswith('sec)')].str.split('\ \(').str[1].str.split(', ').str[0].unique()
    print('Filters found: {}'.format(filters))
    df = df[df.colnames != '']
    return df, filters

def read_dolphot(photfile, columns_df, filters, to_hdf=False, full=False):
    """Reads in raw dolphot output (.phot file) to a DataFrame with named
    columns, and optionally writes it to a HDF5 file.

    Inputs
    ------
    photile : path
        path to raw dolphot output
    columns_df : DataFrame
        table of column names and indices, created by `name_columns`
    filters : list
        List of filters included in output, also from `name_columns`
    to_hdf : bool, optional
        Whether to write photometry table to HDF5 file. Defaults to False
        in the function definition, but defaults to True when this script
        is called from the command line.
    full : bool, optional
        Whether to include full photometry output in DataFrame. Defaults 
        to False.

    Returns
    -------
    df : DataFrame
        A table of column descriptions and their corresponding names.

    Outputs
    -------
        HDF5 file containing photometry table
    """
    if not full:
        columns_df = columns_df[columns_df.colnames.str.find('.chip') == -1]
    colnames = columns_df.colnames
    usecols = columns_df.index
    df = dd.read_csv(photfile, delim_whitespace=True, header=None,
                     usecols=usecols, names=colnames,
                     na_values=99.999).compute()
    if to_hdf:
        outfile = photfile + '.hdf5'
        print('Reading in header information from individual images')
        fitsdir = Path(photfile).parent
        header_df = make_header_table(fitsdir)
        header_df.to_hdf(outfile, key='fitsinfo', mode='w', format='table',
                         complevel=9, complib='zlib')
        lamfunc = lambda x: '-'.join(x[~(x.str.startswith('CLEAR')|x.str.startswith('nan'))])
        filter_detectors = header_df.filter(regex='(DETECTOR)|(FILTER)').astype(str).apply(lamfunc, axis=1).unique()
        print('Writing photometry to {}'.format(outfile))
        df0 = df[colnames[colnames.str.find(r'.chip') == -1]]
        df0 = cull_photometry(df0, filter_detectors)
        df0.to_hdf(outfile, key='data', mode='a', format='table', 
                   complevel=9, complib='zlib')
        if full:
            for f in filters:
                print('Writing individual exposure photometry table for filter {}'.format(f))
                df.filter(regex='_{}_'.format(f)).to_hdf(outfile, key=f, 
                          mode='a', format='table', complevel=9, complib='zlib')
        print('Finished writing HDF5 file')
    else:
        return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', action='store')
    parser.add_argument('--to_hdf', dest='to_hdf', action='store_false', required=False)
    parser.add_argument('--full', dest='full', action='store_true', required=False)
    args = parser.parse_args()
    
    photfile = args.filebase if args.filebase.endswith('.phot') else args.filebase + '.phot'
    colfile = photfile + '.columns'
    print('Photometry file: {}'.format(photfile))
    print('Columns file: {}'.format(colfile))
    columns_df, filters = name_columns(colfile)
    
    import time
    t0 = time.time()
    df = read_dolphot(photfile, columns_df, filters, args.to_hdf, args.full)
    t1 = time.time()
    timedelta = t1 - t0
    ts = (timedelta % 60)
    tm = (timedelta - ts)/60
    print('Finished in {:.0f} m {:.2f} s'.format(tm, ts))
