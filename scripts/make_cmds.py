#! /usr/bin/env python

"""Makes CMDs of ST and GST measurements.

Authors
-------
    Meredith Durbin, February 2018

Use
---
    This script is intended to be executed from the command line as
    such:
    ::
        python make_cmds.py ['filebase']
    
    Parameters:
    (Required) [filebase] - Path to .phot file, with or without .phot extension.
    (Optional) [--to_hdf] - Whether to write the dataframe to an HDF5 
    file. Default is True.
    (Optional) [--full] - Whether to use the full set of columns (photometry of 
    individual exposures). Default is False.
"""

import matplotlib.pyplot as plt
import numpy as np
import vaex

try:
    import seaborn as sns; sns.set(style='white', font_scale=1.3)
except ImportError:
    print('install seaborn you monster')

# need better way to do this
try:
    ds = vaex.open('testing/14610_M33-B01-F01.phot.hdf5', key='data')
except:
    import pandas as pd
    df = pd.read_hdf('testing/14610_M33-B01-F01.phot.hdf5', key='data')
    df['gst_count'] = df.filter(regex='gst').sum(axis=1).astype(int)
    ds = vaex.from_pandas(df)

def make_cmd(ds, red_filter, blue_filter, y_filter, n_err=12,
             density_kwargs={'f':'log10', 'colormap':'viridis'},
             scatter_kwargs={'c':'k', 'alpha':0.5, 's':1}):
    color = '{}_{}_vega'.format(blue_filter, red_filter)
    blue_vega = '{}_vega'.format(blue_filter)
    red_vega = '{}_vega'.format(red_filter)
    y_vega = '{}_vega'.format(y_filter)
    ds[color] = ds[blue_vega]-ds[red_vega]
    gst_criteria = ds['{}_gst'.format(red_filter)] & ds['{}_gst'.format(blue_filter)]
    if (y_filter != blue_filter) & (y_filter != red_filter):
        gst_criteria = gst_criteria & ds['{}_gst'.format(y_filter)]
    ds_gst = ds[gst_criteria]
    xmin = np.nanmin(ds_gst[color].tolist())
    xmax = np.nanmax(ds_gst[color].tolist())
    ymin = np.nanmin(ds_gst[y_vega].tolist())
    ymax = np.nanmax(ds_gst[y_vega].tolist())
    if ds_gst.length() > 50000:
        fig, ax = plt.subplots(1, figsize=(6.,4.))
        data_shape = int(np.sqrt(ds_gst.length()))
        ds_gst.plot(color, y_vega, shape=data_shape,
                    limits=[[xmin,xmax],[ymax,ymin]],
                    **density_kwargs)
    else:
        fig, ax = plt.subplots(1, figsize=(5.,4.))
        ds_gst.scatter(color, y_vega, **scatter_kwargs)
        ax.invert_yaxis()
    y_binned = ds_gst.mean(y_vega, binby=ds_gst[y_vega], shape=n_err)
    xerr = ds_gst.median_approx('({}_err**2 + {}_err**2)**0.5'.format(blue_filter, red_filter),
                                 binby=ds_gst[y_vega], shape=n_err)
    yerr = ds_gst.median_approx('{}_err'.format(y_filter),
                                 binby=ds_gst[y_vega], shape=n_err)
    x_binned = [xmax*0.9]*n_err
    ax.errorbar(x_binned, y_binned, yerr=yerr, xerr=xerr,
                fmt=',', color='k', lw=1.5)
