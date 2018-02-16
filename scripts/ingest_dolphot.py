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
        python ingest_dolphot.py ['filebase'] ['--to_hdf']
    Parameters:
    (Required) [filebase] - Path to .phot file, with or without .phot extension.
    (Optional) [--to_hdf] - Whether to write the dataframe to an HDF5 
    file. Default is True.
"""

# Command line usage: python ingest_dolphot.py <path/to/raw/photometry/file>

import numpy as np
import pandas as pd
import dask.dataframe as dd

# global photometry values
# these are the first 11 columns in the raw dolphot output
global_columns = ['ext','chip','x','y','chi_gl','snr_gl','sharp_gl', 
                  'round_gl','majax_gl','crowd_gl','objtype_gl']

# dictionary mapping description in .columns file to column suffix
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

def name_columns(colfile):
    """Construct a table of column names for dolphot output, with indices
    corresponding to the column number in dolphot output file.

    Returns
    -------
    df : DataFrame
        A table of column descriptions and their corresponding names.
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
    df = df[df.colnames != '']
    return df, np.unique(filters)

def read_dolphot(photfile, columns_df, filters, to_hdf=False, full=False):
    """Construct a table of column names for dolphot output, with indices
    corresponding to the column number in dolphot output file.

    Returns
    -------
    df : DataFrame
        A table of column descriptions and their corresponding names.
    """
    if not full:
        columns_df = columns_df[columns_df.colnames.str.find('.chip') == -1]
    colnames = columns_df.colnames
    usecols = columns_df.index
    df = dd.read_csv(photfile, delim_whitespace=True, header=None,
                     usecols=usecols, names=colnames).compute()
    if to_hdf:
        print('Writing HDF5 photometry file')
        df0 = df[colnames[colnames.str.find(r'.chip') == -1]]
        df0.to_hdf(photfile + '.hdf5', key='data', mode='w',
                   format='table', complevel=9, complib='zlib')
        if full:
            for f in filters:
                print('Writing individual exposure photometry table for filter {}'.format(f))
                df.filter(regex='_{}_'.format(f)).to_hdf(photfile + '.hdf5', key=f, 
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
    
    photfile = args.filebase + '.phot' if not args.filebase.endswith('.phot') else args.filebase
    colfile = photfile + '.columns'
    print('Photometry file: {}'.format(photfile))
    print('Columns file: {}'.format(colfile))
    columns_df, filters = name_columns(colfile)
    import time
    t0 = time.time()
    df = read_dolphot(photfile, columns_df, filters, args.to_hdf, args.full)
    t1 = time.time()
    print('Finished in {} s'.format(t1 - t0))
