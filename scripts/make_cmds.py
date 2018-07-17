#! /usr/bin/env python

"""Makes CMDs of GST measurements.

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
    (Required) [filebase] - Path to .phot.hdf5 file
"""
import matplotlib
matplotlib.use('Agg') # check this
import matplotlib.pyplot as plt
import numpy as np
import vaex

try:
    import seaborn as sns; sns.set(style='white', font_scale=1.3)
except ImportError:
    print("It looks like you don't have seaborn installed.")
    print("This isn't critical, but the plots will look better if you do!")

# need better way to do this

def make_cmd(ds, red_filter, blue_filter, y_filter, n_err=12,
             density_kwargs={'f':'log10', 'colormap':'viridis'},
             scatter_kwargs={'c':'k', 'alpha':0.5, 's':1}):
    """Plot a CMD with (blue_filter - red_filter) on the x-axis and 
    y_filter on the y-axis.

    Inputs
    ------
    ds : Dataset
        Vaex dataset
    red_filter : string
        filter name for "red" filter
    blue_filter : string
        filter name for "blue" filter
    y_filter : string
        filter name for filter on y-axis (usually same as red_filter)
    n_err : int, optional
        number of bins to calculate median photometric errors for
        default: 12
    density_kwargs : dict, optional
        parameters to pass to ds.plot; see vaex documentation
    scatter_kwargs : dict, optional
        parameters to pass to ds.scatter; see vaex documentation

    Returns
    -------
    Nothing

    Outputs
    -------
    some plots dude
    """
    color = '{}_{}_vega'.format(blue_filter, red_filter)
    blue_vega = '{}_vega'.format(blue_filter)
    red_vega = '{}_vega'.format(red_filter)
    y_vega = '{}_vega'.format(y_filter)
    ds[color] = ds[blue_vega]-ds[red_vega]
    gst_criteria = ds['{}_gst'.format(red_filter)] & ds['{}_gst'.format(blue_filter)]
    if y_filter not in [blue_filter, red_filter]:
        # idk why you would do this but it's an option
        gst_criteria = gst_criteria & ds['{}_gst'.format(y_filter)]
    # cut dataset down to gst stars
    # could use ds.select() but i don't like it that much
    ds_gst = ds[gst_criteria]
    # haxx
    xmin = np.nanmin(ds_gst[color].tolist())
    xmax = np.nanmax(ds_gst[color].tolist())
    ymin = np.nanmin(ds_gst[y_vega].tolist())
    ymax = np.nanmax(ds_gst[y_vega].tolist())
    dx = xmax - xmin
    dy = ymax - ymin
    if ds_gst.length() >= 50000:
        fig, ax = plt.subplots(1, figsize=(6.,4.))
        yshape = int(dy/0.04)
        data_shape = (int(yshape/(dx/dy)),int(yshape))
        print(data_shape)
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
    fig.savefig('{}_{}.jpg'.format(blue_filter, red_filter), dpi=144)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', action='store')
    args = parser.parse_args()

    photfile = args.filebase

    # try:
    #     # I have never gotten vaex to read an hdf5 file successfully
    #     ds = vaex.open(photfile)
    # except:
    import pandas as pd
    df = pd.read_hdf(photfile, key='data')
    ds = vaex.from_pandas(df)

    filter_sets = [('f336w','f275w','f336w'),
                   ('f475w','f336w','f475w'),
                   ('f814w','f475w','f814w'),
                   ('f160w','f475w','f160w'),
                   ('f160w','f814w','f160w'),
                   ('f160w','f110w','f160w')]
    for f in filter_sets:
        make_cmd(ds, *f)



