"""
Signal processing
===================

Mostly time series.
"""


import numpy




def peaks(y,x=None,what=0,**args):
	"""
Detects the peaks in the time series given by Y (and X if provided).

:param x,y: time series input arrays
:param what: select what you want -- max/0 or min/1 peaks returned
:returns: xmax,ymax -- arrays with max peaks in the data.
	"""
	import peakdetect
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
	    raise ValueError, "smooth only accepts 1 dimension arrays."

	if x.size < window_len:
	    raise ValueError, "Input vector needs to be bigger than window size."
	    
	if window_len<3:
	    return x
		
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
	    raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
	
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

