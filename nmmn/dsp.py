"""
Signal processing
===================

Mostly time series.
"""


import numpy
import pylab
import scipy.signal




def peaks(y,x=None,what=0,**args):
	"""
Detects the peaks in the time series given by Y (and X if provided).

:param x,y: time series input arrays
:param what: select what you want -- max/0 or min/1 peaks returned
:returns: xmax,ymax -- arrays with max peaks in the data.
	"""
	from . import peakdetect
	
	peaks=peakdetect.peakdetect(y,x,**args)

	if what==0 or what=="max":
		return zip(*peaks[0])
	else:
		return zip(*peaks[1])





def smooth(x,window_len=11,window='hanning'):
	"""
	Smooth the data using a window with requested size.
	Copied from http://wiki.scipy.org/Cookbook/SignalSmooth
	
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	
	:param x: the input signal 
	:param window_len: the dimension of the smoothing window; should be an odd integer
	:param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'	flat window will produce a moving average smoothing.

	:returns: the smoothed signal
	    
	Example

	>>> t=linspace(-2,2,0.1)
	>>> x=sin(t)+randn(len(t))*0.1
	>>> y=smooth(x)
	
	.. seealso:: numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve, scipy.signal.lfilter
	
	.. note:: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

	.. todo:: the window parameter could be the window itself if an array instead of a string
	""" 
	 
	if x.ndim != 1:
	    raise ValueError("smooth only accepts 1 dimension arrays.")

	if x.size < window_len:
	    raise ValueError("Input vector needs to be bigger than window size.")
	    
	if window_len<3:
	    return x
		
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
	    raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
	
	s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
	#print(len(s))
	if window == 'flat': #moving average
	    w=numpy.ones(window_len,'d')
	else:
	    w=eval('numpy.'+window+'(window_len)')
	
	y=numpy.convolve(w/w.sum(),s,mode='valid')
	return y    




def smoothxy(x,y,*arg,**args):
	"""
:param x: "time" in the time series
:param y: y variable
:returns: smoothed y and corresponding x arrays
	"""
	return smooth(x,*arg,**args), smooth(y,*arg,**args)




def varsig(f,df):
	"""
Quantifies significance of variability of time series, taking into 
account the uncertainties.

Given the time series signal *y* and the corresponding uncertainties
:math:`\sigma y`, the test statistics *T* is defined as 

.. math:: T \equiv \\frac{y_{i+1}-y_i}{\sigma y_i}.

*T* will give the significance of variability at each point in time,
in standard deviations of the preceding noise point.

**Usage:**

Computes TS and plots time series, highlighting the 1sigma variability
interval in gray:

>>> ts=varsig(flux,errflux)
>>> step(t,q,'o-')
>>> fill_between(t, -2*ones_like(tdec), 2*ones_like(tdec), alpha=0.3, facecolor='gray')

.. todo:: generalize to arbitrary windows of time
.. todo:: propagate uncertainty 
	"""
	q=numpy.zeros_like(f)
	for i,fi in enumerate(f):
		if i>0:
			dy=fi-f[i-1]
			q[i]=dy/df[i-1]

	return q





def ls(t,z,plot=False):
    """
Computes and plot Lomb-Scargle periodogram for a given timeseries. 
Returns arrays with periods and LS spectral power.

>>> p,power=l.ls(time,values,plot=False)

:returns: periods, spectral power
    """
    import scipy.signal

    tbin = t[1] - t[0] # time bin
    f = numpy.linspace(0.001, 3./tbin, 10000)
    z = scipy.signal.detrend(z)

    # periodogram
    pgram = scipy.signal.lombscargle(t,z,f)

    # Lomb-Scargle spectral power
    period=1./(f/(2*numpy.pi))
    spower=(numpy.sqrt(4.*(pgram/len(t)))/numpy.mean(numpy.sqrt(4.*(pgram/len(t)))))**2 
    
    # most significant periodicity
    #i=spower.argmax() # finds index of maximum power
    #pbest=period[i]
    #spmax=spower[i]
    #print("Spectral peak found at t="+str(pbest)+", "+str(spmax))

    # plot
    if plot==True:
        #pylab.figure(figsize=(16,8))
        pylab.title("Lomb-Scargle ")
        pylab.ylabel('Spectral Power')
        pylab.xlabel('Period [days]')
        pylab.xscale('log')
        pylab.plot(period, spower)

    return period,spower


def ls_spectra(t,var,n=200,thres=0.1,smooth=0):
    """
Computes Lomb-Scargle power spectrum, find peaks, produces arrays for
plotting images showing the spectral lines.

:param t: array of times
:param var: array of signal
:param n: number of vertical elements in image that will be created showing spectral lines
:param thres: threshold parameter for finding peaks in time series
:param smooth: number of points in the smoothing window. If 0, no smoothing

Usage: ``N,P,l,pp,power=ls_spectra(t,y,thres=0.3)``

Returns the following variables:

- ``N``: 2d number of vertical elements for plotting the spectra
- ``P``: 2d periods
- ``l``: 2d power spectra for plotting images
- ``pp``: periods corresponding to peaks (peak period) in power spectrum
- ``power``: peaks in power spectrum
    """
    import peakutils 

    p,power=ls(t,var)

    # smooth periodogram if requested
    if (smooth!=0): 
        p,power=smoothxy(p,power,smooth)

    T, P = numpy.meshgrid(range(n), p)
    
    # spectral lines
    lines = numpy.ones([1, n]) * power[:, None]
    
    # peaks
    ipeak = peakutils.indexes(power,thres=thres)
        
    return T.T, P.T, lines.T, p[ipeak], power[ipeak]






def error_resampler(errors):
    """
For use with ``pandas``.

Method for performing the proper ``mean`` resampling of the *uncertainties* (error bars) 
in the time series with ``pandas``. Note that doing a simple resampling 
will fail to propagate uncertainties, since error in the mean goes as 

.. math:: \sigma=\sqrt{\Sigma_n \sigma_n^2}

Example: Resamples the errors with 30 day averages:
::

    # df['errflux'] has the 1sigma uncertainties
    err=df['errflux'].resample('30d').apply(nmmn.dsp.error_resampler) 

    # plot y-values (df['flux']) with errors (err)
    df['flux'].resample('30d').mean().plot(yerr=err)
    """
    err=errors**2

    return numpy.sqrt(err.sum())/err.size



def sumts(t1,t2,y1,y2,n=1):
    """
Given two time series, this method returns their sum.

n = number of points in the final TS, as a multiple of the sum of points
in both TS.

Steps:

1. defines a uniform t-array with the number of total elements of the two arrays
2. interpolates both TS in this new t-array
3. sums their values
    """
    # new number of elements
    points=int(n*(t1.size+t2.size)) 
    # lower and upper ranges
    tmin=numpy.min(numpy.hstack((t1,t2)))
    tmax=numpy.max(numpy.hstack((t1,t2)))

    # Defines the new array of interpolated times    
    tnew=numpy.linspace(tmin,tmax,points)
    # interpolates the two TS
    y1new=numpy.interp(tnew,t1,y1)
    y2new=numpy.interp(tnew,t2,y2)
    # sums the two TS
    ynew=y1new+y2new

    return tnew,ynew



def dt(t):
    """
Computes difference between consecutive points in the time series.

:param t: input times
:param y: input y-values
    """
    dtt=[]
    for i in range(t.size):
        if (i>0): 
            dtt.append(t[i]-t[i-1])

    return dtt


def uneven2even(t,y):
    """
Given an uneven timeseries (TS) with multiple values defined at the same 
time, this method will convert it into an evenly sampled
timeseries with a dt as close as possible to the actual dt, removing
duplicate-t values.

Example: suppose you want to compute the CWT for an unevenly sampled
time series. Since the CWT does not support uneven TS, you can first
call this method to regularize your TS and then perform the TS.

Algorithm:

- REMOVE DUPLICATE times
- CREATE REGULAR GRID USING BEST DT
- INTERPOLATE NEW TS ON PREVIOUS ONE 

:param t: input times
:param y: input value
    """
    import scipy.interpolate

    # remove items with same t
    # ==========================
    # gets successive dt for all points
    dtarr=[] 
    idel=[] # list of indexes for elements that will be removed
    for i in range(t.size):
        if (i>0):
            dt=t[i]-t[i-1]
            if (dt==0): # if they have dt==0, store their index and later remove them
                idel.append(i)
            else:
                dtarr.append(dt) # stores only dt!=0

    # Find out optimal value of dt for the new TS
    dt=numpy.mean(dtarr)

    # Removes elements with same t
    tuniq=numpy.delete(t,idel)
    yuniq=numpy.delete(y,idel)

    # Does linear interpolation on new TS
    # ======================================
    # new regular grid, as close as possible to original one
    tnew=numpy.arange(t[0],t[-1],dt)

    # interpolation
    f = scipy.interpolate.interp1d(tuniq, yuniq)
    ynew = f(tnew)

    return tnew,ynew





# Wavelet methods
# ==================


def cwt(t,sig):

    """
Given the time and flux (sig), this method computes a continuous 
wavelet transform (CWT).

.. warning:: Deprecated.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import obspy.signal.tf_misfit
    import scipy.signal


    tbin = t[1] - t[0]
    freqLC = numpy.fft.fftfreq(len(t),d=tbin)
    fluxo = sig
    
    sig = scipy.signal.detrend(sig)
    fig = plt.figure()
    f_min = 5./t.max()
    f_max = 1000*freqLC.max()     #so calcula ate onde vai a FFT. Porque nao faz sentido tentar encontrar frequencias alem da resolucao temporal
    gs0 = matplotlib.gridspec.GridSpec(2, 2,hspace=0,wspace=0)   #GridSpec specifies the geometry of the grid that a subplot will be placed. 
                                                      #The number of rows and number of columns of the grid need to be set (here: 2 x 2).
                                                      #The horizontal and vertical space between graphics can be adjusted in "hspace" and "wspace"
    ax = fig.add_subplot(gs0[0])
    dt = 0.001*tbin
    CWT = obspy.signal.tf_misfit.cwt(sig,dt,6,f_min,f_max,nf=1000)
    x, y = numpy.meshgrid(t, numpy.logspace(numpy.log10(f_min), numpy.log10(f_max), CWT.shape[0]))
    NormCWT = (numpy.abs(CWT)**2)/numpy.var(fluxo)


    return x, 1000./y, NormCWT




def reconstruct(wa,i=None):
    """
Method to reconstruct a time series (TS) signal based on its CWT. 

:param wa: wavelet object generated with `wavelets` module
:param i: index array containing only the elements that will be excluded from the signal reconstruction. i.e. the elements indicated by ``i`` will zeroed in the CWT complex array.
:returns: full reconstructed TS, reconstructed CWT power array, reconstructed CWT complex array, detrended reconstructed TS

Examples: 

1. Compute the CWT for TS in ``var``:
::

    import wavelets

    # compute CWT
    wa = wavelets.WaveletAnalysis(var, dt=dt)

    # time and period 2D arrays
    T, S = numpy.meshgrid(wa.time, wa.scales)
  
2. Reconstruct the signal, throwing out all periods with values between 1000 and 2000:
::

    j=where((S<1000) | (S>2000))
    recdet,rec=nmmn.dsp.reconstruct(wa,j)

3. Plots the reconstructed signal along with the data:
::

    subplot(2,1,1)
    plot(t,flux,label='original')
    plot(t,rec,label='reconstructed signal',lw=5)

    subplot(2,1,2)
    plot(t,recdet,'r')
    title('Pure reconstructed signal')

    """
    wavecut=wa.wavelet_transform
    if i is not None: wavecut[i]=0
    
    # reconstructed signal
    rec=wa.reconstruction(wave=wavecut)
    
    # find the linear trend on the signal
    import scipy.stats
    a, b, r, p, err = scipy.stats.linregress(wa.time,wa.data)

    # detrend the reconstructed signal
    import scipy.signal
    recdet=scipy.signal.detrend(rec)
    
    return recdet+a*wa.time+b,numpy.abs(wavecut)**2,wavecut,recdet



def reconstruct_period(wa,P1,P2):
    """
Returns reconstructed detrended signal from CWT, considering only the interval
between the two given periods.
    """
    T, P = numpy.meshgrid(wa.time, wa.scales)

    i=numpy.where((P<P1) | (P>P2))
    xrec,powerrec,cwtrec,xrecdet=reconstruct(wa,i)
    return xrecdet




def cwt_spectra(t,var,dj=0.01,n=200,thres=0.1):
    """
Computes CWT power spectrum, find peaks, produces arrays for
plotting images showing the spectral lines. Note that the CWT power spectrum
is an average from the CWT power array. Therefore, they are a smoothed out 
version of a Fourier spectrum.

:param t: array of times
:param var: array of signal
:param n: number of vertical elements in image that will be created showing spectral lines
:param thres: threshold parameter for finding peaks in time series

Usage: ``N,P,l,pp,power=cwt_spectra(t,y,thres=0.3)``

Returns the following variables:

- ``N``: 2d number of vertical elements for plotting the spectra
- ``P``: 2d periods
- ``l``: 2d power spectra for plotting images
- ``pp``: periods corresponding to peaks (peak period) in power spectrum
- ``power``: peaks in CWT power spectrum
    """    
    import peakutils, wavelets

    dt=t[1]-t[0] # bin size
    wa = wavelets.WaveletAnalysis(var, dt=dt,dj=dj)
    T, P = numpy.meshgrid(range(n), wa.scales)
    cwtpower=wa.wavelet_power
    
    # spectral lines
    z=cwtpower.mean(axis=1)
    lines = numpy.ones([1, n]) * z[:, None]
    
    # peaks
    ipeak = peakutils.indexes(z,thres=thres)
        
    return T.T, P.T, lines.T, wa.scales[ipeak], z[ipeak]


