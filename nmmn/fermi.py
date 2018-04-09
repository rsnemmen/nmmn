"""
Methods to handle Fermi LAT data
=====================================

Handle output of Fermi analysis.
"""

import os, sys
import pylab, numpy






def getconfig(configfile="analysis.conf"):
    """
Reads config file and gets parameters.
    """
    import configobj
    config = configobj.ConfigObj(configfile)

    # gets useful things 
    folder = config['out']
    cmapfile=folder+"/"+config['target']['name']+'_'+config['file']['tag']+'_CountMap.fits'
    modelmap=folder+"/"+config['target']['name']+'_'+config['file']['tag']+'_ModelMap.fits'
    residuals=folder+"/"+config['target']['name']+'_Residual_Model_cmap.fits'
    images = [residuals, modelmap, cmapfile]
    tsdir=folder+"/TSMap/"
    tsfits=folder+"/"+config['target']['name']+'_'+config['file']['tag']+'_TSMap.npz'  # tsmap filename

    # create dictionary with useful values
    useful = {'folder': folder, 'cmapfile': cmapfile, 'modelmap':modelmap, 'residuals':residuals, 'images':images, 'tsdir':tsdir, 'tsfits':tsfits}

    return config, useful





def plotMaps(configfile="analysis.conf",cmap=None):
    """
Given the directory where the results of an Enrico analysis are 
located (enrico_sed), this method plots the different count maps
(observed, model and residuals).

This is the fixed version of the script plotMaps.py from the Enrico
distribution. 

:param configdir: directory with "/" at the end that points to the place
where Enrico output files are located.
:returns: image file SOURCE_Maps.png in the source folder.
    """
    import astropy, astropy.io.fits, astropy.wcs
    import aplpy

    config,c=getconfig(configfile)

    def set_hgps_style(f):
        """Set HGPS style for a f = aplpy.FITSFigure"""
        f.ticks.set_xspacing(2)
        f.ticks.set_yspacing(2)
        f.ticks.set_linewidth(1.5)
        f.tick_labels.set_xformat('dd')
        f.tick_labels.set_yformat('dd')
        f.tick_labels.set_style('colons')
        f.axis_labels.set_xtext('Right Ascension (deg)')
        f.axis_labels.set_ytext('Declination (deg)')

    # Determine image center and width / height
    dpi = 2000
    header = astropy.io.fits.getheader(c['cmapfile'])
    wcs = astropy.wcs.WCS(header)
    header['NAXIS1'] / dpi
    header['NAXIS2'] / dpi
    lon, lat = header['NAXIS1'] / 2., header['NAXIS2'] / 2.
    x_center, y_center = wcs.wcs_pix2world(lon, lat, 0)
    radius = header['CDELT2'] * header['NAXIS2'] / 2.

    # Computing the sub-figure sizes is surprisingly hard
    figsize=(5, 15)
    figure = pylab.figure(figsize=figsize)
    axis_ratio = figsize[0] / float(figsize[1])
    edge_margin_x = 0.12
    edge_margin_y = edge_margin_x * axis_ratio
    edge_margin_x_up = 0.01
    edge_margin_y_up = edge_margin_x_up * axis_ratio
    inner_margin_x = 0.1
    inner_margin_y = inner_margin_x * axis_ratio
    size_x = (1 - edge_margin_x - edge_margin_x_up)
    size_y = (1 - edge_margin_y - edge_margin_y_up - 2 * inner_margin_y) / 3

    for i, image in enumerate(c['images']):
        subplot = [edge_margin_x, edge_margin_y + i * (size_y + inner_margin_y), size_x, size_y]
        f = aplpy.FITSFigure(image, figure=figure, subplot=subplot)
        f.recenter(x_center, y_center, 0.95 * radius)
        set_hgps_style(f)
        f.show_colorscale(stretch='power', exponent=1, cmap=cmap)
        #f.show_colorbar()
        if i==0: f.show_markers(x_center, y_center, edgecolor='Black', s=400.)
        if i==1: 
            f.show_markers(x_center, y_center, edgecolor='White', s=400.)
            f.show_regions(c['folder']+'/Roi_model.reg')
        if i==2: f.show_markers(x_center, y_center, edgecolor='Black', s=400.)

    plotfile =c['folder']+"/"+config['target']['name']+"_Maps.png"
    print('Writing {}'.format(plotfile))
    figure.savefig(plotfile)


def plotMap(counts,out='tmp.png',cmap=None,roi=None):
    """
This method plots some count map. Inherited from the method plotMaps
above.  

:param counts: FITS file with counts to be plotted
:param cmap: desired colormap for image
:param roi: DS9 region file if present, otherwise leave None
:param out: output filename and format
:returns: image file 'tmp.png' in the source folder.
    """
    import astropy, astropy.io.fits, astropy.wcs
    import aplpy

    def set_hgps_style(f):
        """Set HGPS style for a f = aplpy.FITSFigure"""
        f.ticks.set_xspacing(2)
        f.ticks.set_yspacing(2)
        f.ticks.set_linewidth(1.5)
        f.tick_labels.set_xformat('dd')
        f.tick_labels.set_yformat('dd')
        f.tick_labels.set_style('colons')
        f.axis_labels.set_xtext('Right Ascension (deg)')
        f.axis_labels.set_ytext('Declination (deg)')

    # Determine image center and width / height
    dpi = 2000
    header = astropy.io.fits.getheader(counts)
    wcs = astropy.wcs.WCS(header)
    header['NAXIS1'] / dpi
    header['NAXIS2'] / dpi
    lon, lat = header['NAXIS1'] / 2., header['NAXIS2'] / 2.
    x_center, y_center = wcs.wcs_pix2world(lon, lat, 0)
    radius = header['CDELT2'] * header['NAXIS2'] / 2.

    f = aplpy.FITSFigure(counts)
    f.recenter(x_center, y_center, 0.95 * radius)
    set_hgps_style(f)
    f.show_colorscale(stretch='power', exponent=1, cmap=cmap)
    f.show_colorbar()
    f.show_markers(x_center, y_center, edgecolor='White', s=400.)
    if roi!=None: f.show_regions(roi)
   
    print('Writing {}'.format(out))
    pylab.savefig(out)
    
    




def plotTSmap(configfile="analysis.conf") :
    """ From Enrico/tsmap.py.

    Gather the results of the evaluation of 
    each pixel and fill a fits file"""

    # gets parameters from the Enrico configuration file
    config,c=getconfig(configfile)

    # Read the cmap produced before to get the grid for the TS map
    npix = int(config['TSMap']['npix'])
    ts=numpy.zeros((npix,npix))
    ra=ts.copy()
    dec=ts.copy()

    import string # read the results
    for i in range(npix):
        for j in range(npix):
            try : 
                lines = open(c['tsdir']+PixelFile(i,j),"r").readlines()
                tsval = float(string.split(lines[0])[2])    # TS value
                raval = float(string.split(lines[0])[0])    # RA
                decval = float(string.split(lines[0])[1])   # dec
            except :
                print("Cannot find, open or read "+c['tsdir']+PixelFile(i,j))
                Value = 0.
            ts[i,j] = tsval
            ra[i,j] = raval
            dec[i,j] = decval


    # save file
    numpy.savez(c['tsfits'], TS=ts,ra=ra,dec=dec)
    print("TS Map saved in "+c['tsfits'])   

    return ra,dec,ts 
    
    


def PixelFile(i,j):
    """ return the name of a file where the result of 1 pixel 
    evaluation will be stored

    From enrico/tsmap.py.
    """
    return 'Pixel_'+str(i)+'_'+str(j)





def converttime(y):
    """
Converts from MET Fermi time to a python datetime tuple.

>>> convertyear(386817828)

returns datetime.datetime(2013, 4, 5, 1, 23, 48).

:param y: a float or array of floats
:returns: a datetime structure (or list) with the date corresponding to the input float or array

y is the amount of seconds since 2001.0 UTC. Beware that my method
may assume that all years have 365 days.

References: 

- http://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
- http://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl
    """
    import datetime

    t0=2001

    if numpy.size(y)==1:    # if input is float
        day_one = datetime.datetime(t0,1,1) # initial time MET
        d = datetime.timedelta(seconds=y) # dt since t0
        date = d + day_one
    else:   # if input is list/array
        date=[]

        for i, yi in enumerate(y): 
            day_one = datetime.datetime(t0,1,1)
            d = datetime.timedelta(seconds=yi)
            date.append(d + day_one)

    return date





def starttime(t):
    """
Fixes the start time of observations taken with LAT. This assumes that the
time array in your light curve is in days already, and the beginning of
your analysis coincides with t0 for LAT.

:param t: array with input times given in days, with t0 coinciding with the beginning of LAT
:returns: t (days), t (years)
    """
    import pendulum

    # start of observations by Fermi
    tp=pendulum.create(2008,8,4,15,43,37)-pendulum.create(2008,1,1)
    t0=tp.days # how many days since 2008 for the beginning of Fermi observations

    t=t-t[0]+t0 # fixes t array 
    ty=2008.+t/365. # time in years

    return t,ty


