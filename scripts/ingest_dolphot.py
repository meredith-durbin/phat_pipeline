#! /usr/bin/env python

# Command line usage: python ingest_dolphot.py <path/to/raw/photometry/file>

import numpy as np
import pandas as pd
import dask.dataframe as dd

# global photometry values
# these are the first 11 columns in the raw dolphot output
global_columns = ['EXT','CHIP','X','Y','CHI_GL','SNR_GL','SHARP_GL', 
                  'ROUND_GL','MAJAX_GL','CROWD_GL','OBJTYPE_GL']

# dictionary mapping column description to column suffix
colname_mappings = {
    'Total counts,'                      : 'count',
    'Total sky level,'                   : 'sky',
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

# construct dataframe mapping column descriptions and indices to names in output
def name_columns(colfile):
    df_all = pd.DataFrame(data=np.loadtxt(colfile, delimiter='. ', dtype=str),
                          columns=['index','desc']).drop('index', axis=1)
    df = df_all[df_all.desc.str.find(r'.chip') == -1]
    df = df.assign(colnames='')
    df.loc[:10,'colnames'] = global_columns
    for k, v in colname_mappings.items():
        indices = df[df.desc.str.find(k) != -1].index
        filters = df.loc[indices,'desc'].str.split(", ").str[-1]
        df.loc[indices,'colnames'] = filters + '_' + v.upper()
    df = df[df.colnames != '']
    return df

# ingest raw dolphot output (and optionally write to hdf5 file) 
def read_dolphot(photfile, columns_df, to_hdf=False):
    colnames = columns_df.colnames
    usecols = columns_df.index
    df = dd.read_csv(photfile, delim_whitespace=True, header=None,
                     usecols=usecols, names=colnames).compute()
    if to_hdf:
        df.to_hdf(photfile + '.hdf5', key='data', mode='w',
                  format='table', complevel=9, complib='zlib')
        return False
    else:
        return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', action='store')
    parser.add_argument('--to_hdf', dest='to_hdf', action='store_false', required=False)
    args = parser.parse_args()
    
    photfile = args.filebase + '.phot' if not args.filebase.endswith('.phot') else args.filebase
    colfile = photfile + '.columns'
    print('Photometry file: {}'.format(photfile))
    print('Columns file: {}'.format(colfile))
    columns_df = name_columns(colfile)
    import time
    t0 = time.time()
    df = read_dolphot(photfile, columns_df, args.to_hdf)
    t1 = time.time()
    print('Finished in {} s'.format(t1 - t0))
