"""
Fancy plots
==============
"""

import numpy
from matplotlib import pylab
from nmmn import sed

def plot(spec):
	"""
Returns the plot of a grmonty spectrum as a pyplot object or plot it on 
the screen
	
:param param: grmonty spectrum file
	"""

	s = sed.SED()
	s.grmonty(spec)
	pylab.plot(s.lognu, s.ll)
	pylab.show()
	

	

def onehist(x,xlabel='',fontsize=12):
	""" 
Script that plots the histogram of x with the corresponding xlabel. 
	"""

	pylab.clf()
	pylab.rcParams.update({'font.size': fontsize})
	pylab.hist(x,histtype='stepfilled')
	pylab.legend()
	#### Change the X-axis appropriately ####
	pylab.xlabel(xlabel)
	pylab.ylabel('Number')
	pylab.draw()
	pylab.show()





def twohists(x1,x2,xmin,xmax,range=None,x1leg='$x_1$',x2leg='$x_2$',xlabel='',fig=1,sharey=False,fontsize=12,bins1=10,bins2=10):
	"""
Script that plots two histograms of quantities x1 and x2
sharing the same X-axis.

:param x1,x2: arrays with data to be plotted
:param xmin,xmax: lower and upper range of plotted values, will be used to set a consistent x-range
	for both histograms.
:param x1leg, x2leg: legends for each histogram	
:param xlabel: self-explanatory.
:param bins1,bins2: number of bins in each histogram
:param fig: which plot window should I use?
:param range: in the form (xmin,xmax), same as range argument for hist and applied to both
	histograms.

Inspired by `Scipy <http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label>`_.
	"""

	pylab.rcParams.update({'font.size': fontsize})
	fig=pylab.figure(fig)
	pylab.clf()
	
	a=fig.add_subplot(2,1,1)
	if sharey==True:
		b=fig.add_subplot(2,1,2, sharex=a, sharey=a)
	else:
		b=fig.add_subplot(2,1,2, sharex=a)
	
	a.hist(x1,bins1,label=x1leg,color='b',histtype='stepfilled',range=range)
	a.legend(loc='best',frameon=False)
	a.set_xlim(xmin,xmax)
	
	b.hist(x2,bins2,label=x2leg,color='r',histtype='stepfilled',range=range)
	b.legend(loc='best',frameon=False)
	
	pylab.setp(a.get_xticklabels(), visible=False)

	b.set_xlabel(xlabel)
	b.set_ylabel('Number',verticalalignment='bottom')
	pylab.minorticks_on()
	pylab.subplots_adjust(hspace=0.15)
	pylab.draw()
	pylab.show()





def threehists(x1,x2,x3,xmin,xmax,x1leg='$x_1$',x2leg='$x_2$',x3leg='$x_3$',xlabel='',fig=1,sharey=False,fontsize=12):
	"""
Script that plots three histograms of quantities x1, x2 and x3 
sharing the same X-axis.

Arguments:
- x1,x2,x3: arrays with data to be plotted
- xmin,xmax: lower and upper range of plotted values, will be used to set a consistent x-range for both histograms.
- x1leg, x2leg, x3leg: legends for each histogram	
- xlabel: self-explanatory.
- sharey: sharing the Y-axis among the histograms?
- fig: which plot window should I use?

Example:
x1=Lbol(AD), x2=Lbol(JD), x3=Lbol(EHF10)

>>> threehists(x1,x2,x3,38,44,'AD','JD','EHF10','$\log L_{\\rm bol}$ (erg s$^{-1}$)',sharey=True)

Inspired by `Scipy <http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label>`_.
	"""
	pylab.rcParams.update({'font.size': fontsize})
	fig=pylab.figure(fig)
	pylab.clf()
	
	a=fig.add_subplot(3,1,1)
	if sharey==True:
		b=fig.add_subplot(3,1,2, sharex=a, sharey=a)
		c=fig.add_subplot(3,1,3, sharex=a, sharey=a)
	else:
		b=fig.add_subplot(3,1,2, sharex=a)
		c=fig.add_subplot(3,1,3, sharex=a)		
	
	a.hist(x1,label=x1leg,color='b',histtype='stepfilled')
	a.legend(loc='best',frameon=False)
	a.set_xlim(xmin,xmax)
	
	b.hist(x2,label=x2leg,color='r',histtype='stepfilled')
	b.legend(loc='best',frameon=False)

	c.hist(x3,label=x3leg,color='y',histtype='stepfilled')
	c.legend(loc='best',frameon=False)
	
	pylab.setp(a.get_xticklabels(), visible=False)
	pylab.setp(b.get_xticklabels(), visible=False)

	c.set_xlabel(xlabel)
	b.set_ylabel('Number')
	pylab.minorticks_on()
	pylab.subplots_adjust(hspace=0.15)
	pylab.draw()
	pylab.show()
	
	
	




def fourhists(x1,x2,x3,x4,xmin,xmax,x1leg='$x_1$',x2leg='$x_2$',x3leg='$x_3$',x4leg='$x_3$',xlabel='',fig=1,sharey=False,fontsize=12,bins1=10,bins2=10,bins3=10,bins4=10,line1=None,line2=None,line3=None,line4=None,line1b=None,line2b=None,line3b=None,line4b=None,loc='best'):
	"""
Script that plots four histograms of quantities x1, x2, x3 and x4
sharing the same X-axis.

Arguments:

- x1,x2,x3,x4: arrays with data to be plotted
- xmin,xmax: lower and upper range of plotted values, will be used to set a consistent x-range
or both histograms.
- x1leg, x2leg, x3leg, x4leg: legends for each histogram	
- xlabel: self-explanatory.
- sharey: sharing the Y-axis among the histograms?
- bins1,bins2,...: number of bins in each histogram
- fig: which plot window should I use?
- line?: draws vertical solid lines at the positions indicated in each panel
- line?b: draws vertical dashed lines at the positions indicated in each panel

.. figure:: ../figures/fourhists.png
	:scale: 100 %
	:alt: Four histograms in the same figure

	Four histograms in the same figure.

Inspired by `Scipy <http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label>`_.
	"""
	pylab.rcParams.update({'font.size': fontsize})
	fig=pylab.figure(fig)
	pylab.clf()
	
	a=fig.add_subplot(4,1,1)
	if sharey==True:
		b=fig.add_subplot(4,1,2, sharex=a, sharey=a)
		c=fig.add_subplot(4,1,3, sharex=a, sharey=a)
		d=fig.add_subplot(4,1,4, sharex=a, sharey=a)
	else:
		b=fig.add_subplot(4,1,2, sharex=a)
		c=fig.add_subplot(4,1,3, sharex=a)		
		d=fig.add_subplot(4,1,4, sharex=a)
	
	def vline(hist,value,linestyle='k'):
		"""Draw vertical line"""
		yax=hist.set_ylim()
		hist.plot([value,value],[yax[0],yax[1]],linestyle,linewidth=2)

	a.hist(x1,bins1,label=x1leg,color='b',histtype='stepfilled')
	a.legend(loc=loc,frameon=False)
	a.set_xlim(xmin,xmax)
	if line1!=None: vline(a,line1)
	if line1b!=None: vline(a,line1b,'k--')
	
	b.hist(x2,bins2,label=x2leg,color='r',histtype='stepfilled')
	b.legend(loc=loc,frameon=False)
	if line2!=None: vline(b,line2)
	if line2b!=None: vline(b,line2b,'k--')

	c.hist(x3,bins3,label=x3leg,color='y',histtype='stepfilled')
	c.legend(loc=loc,frameon=False)
	if line3!=None: vline(c,line3)
	if line3b!=None: vline(c,line3b,'k--')

	d.hist(x4,bins4,label=x4leg,color='g',histtype='stepfilled')
	d.legend(loc=loc,frameon=False)
	if line4!=None: vline(d,line4)
	if line4b!=None: vline(d,line4b,'k--')
	
	pylab.setp(a.get_xticklabels(), visible=False)
	pylab.setp(b.get_xticklabels(), visible=False)
	pylab.setp(c.get_xticklabels(), visible=False)

	d.set_xlabel(xlabel)
	c.set_ylabel('Number')
	pylab.minorticks_on()
	pylab.subplots_adjust(hspace=0.15)
	pylab.draw()
	pylab.show()






def fourcumplot(x1,x2,x3,x4,xmin,xmax,x1leg='$x_1$',x2leg='$x_2$',x3leg='$x_3$',x4leg='$x_3$',xlabel='',ylabel='$N(x>x\')$',fig=1,sharey=False,fontsize=12,bins1=50,bins2=50,bins3=50,bins4=50):
	"""
Script that plots the cumulative histograms of four variables x1, x2, x3 and x4
sharing the same X-axis. For each bin, Y is the fraction of the sample 
with values above X.

Arguments:

- x1,x2,x3,x4: arrays with data to be plotted
- xmin,xmax: lower and upper range of plotted values, will be used to set a consistent x-range
for both histograms.
- x1leg, x2leg, x3leg, x4leg: legends for each histogram	
- xlabel: self-explanatory.
- sharey: sharing the Y-axis among the histograms?
- bins1,bins2,...: number of bins in each histogram
- fig: which plot window should I use?

Inspired by `Scipy <http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label>`_.

v1 Jun. 2012: inherited from fourhists.
	"""
	pylab.rcParams.update({'font.size': fontsize})
	fig=pylab.figure(fig)
	pylab.clf()
	
	a=fig.add_subplot(4,1,1)
	if sharey==True:
		b=fig.add_subplot(4,1,2, sharex=a, sharey=a)
		c=fig.add_subplot(4,1,3, sharex=a, sharey=a)
		d=fig.add_subplot(4,1,4, sharex=a, sharey=a)
	else:
		b=fig.add_subplot(4,1,2, sharex=a)
		c=fig.add_subplot(4,1,3, sharex=a)		
		d=fig.add_subplot(4,1,4, sharex=a)
	
	a.hist(x1,bins1,label=x1leg,color='b',cumulative=-True,normed=True,histtype='stepfilled')
	a.legend(loc='best',frameon=False)
	a.set_xlim(xmin,xmax)
	
	b.hist(x2,bins2,label=x2leg,color='r',cumulative=-True,normed=True,histtype='stepfilled')
	b.legend(loc='best',frameon=False)

	c.hist(x3,bins3,label=x3leg,color='y',cumulative=-True,normed=True,histtype='stepfilled')
	c.legend(loc='best',frameon=False)

	d.hist(x4,bins4,label=x4leg,color='g',cumulative=-True,normed=True,histtype='stepfilled')
	d.legend(loc='best',frameon=False)
	
	pylab.setp(a.get_xticklabels(), visible=False)
	pylab.setp(b.get_xticklabels(), visible=False)
	pylab.setp(c.get_xticklabels(), visible=False)

	d.set_xlabel(xlabel)
	c.set_ylabel(ylabel)
	pylab.minorticks_on()
	pylab.subplots_adjust(hspace=0.15)
	pylab.draw()
	pylab.show()







def threehistsx(x1,x2,x3,x1leg='$x_1$',x2leg='$x_2$',x3leg='$x_3$',fig=1,fontsize=12,bins1=10,bins2=10,bins3=10):
	"""
Script that pretty-plots three histograms of quantities x1, x2 and x3.

Arguments:
:param x1,x2,x3: arrays with data to be plotted
:param x1leg, x2leg, x3leg: legends for each histogram	
:param fig: which plot window should I use?

Example:
x1=Lbol(AD), x2=Lbol(JD), x3=Lbol(EHF10)

>>> threehists(x1,x2,x3,38,44,'AD','JD','EHF10','$\log L_{\\rm bol}$ (erg s$^{-1}$)')

Inspired by http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label.
	"""
	pylab.rcParams.update({'font.size': fontsize})
	pylab.figure(fig)
	pylab.clf()
	
	pylab.subplot(3,1,1)
	pylab.hist(x1,label=x1leg,color='b',bins=bins1)
	pylab.legend(loc='best',frameon=False)

	pylab.subplot(3,1,2)
	pylab.hist(x2,label=x2leg,color='r',bins=bins2)
	pylab.legend(loc='best',frameon=False)

	pylab.subplot(3,1,3)
	pylab.hist(x3,label=x3leg,color='y',bins=bins3)
	pylab.legend(loc='best',frameon=False)

	pylab.minorticks_on()
	pylab.subplots_adjust(hspace=0.15)
	pylab.draw()
	pylab.show()

	
	

	
def fitconf(xdata,ydata,errx,erry,covxy,nboot=1000,bcesMethod='ort',linestyle='',conf=0.683,confcolor='gray',xplot=None,front=False,**args):
	"""
	This is a wrapper that given the input data performs the BCES
	fit, get the orthogonal parameters and plot the best-fit line and
	confidence band (generated using analytical methods). I decided to put together 
	these commands in a method because I have been using them very frequently.
	
	Assumes you initialized the plot window before calling this method.
	
	Usage:

	>>> a1,b1,erra1,errb1,cov1=nemmen.fitconf(x[i],y[i],errx[i],erry[i],covxy[i],nboot,bces,linestyle='k',confcolor='LightGrey')
	
	Explanation of some arguments:

	- xplot: if provided, will compute the confidence band in the X-values provided
	with xplot
	- front: if True, then will plot the confidence band in front of the data
	points; otherwise, will plot it behind the points
	"""	
	import bces.bces
	from . import stats
	from . import misc	

	# Selects the desired BCES method
	i=misc.whichbces(bcesMethod)
		
	# Performs the BCES fit
	a,b,erra,errb,cov=bces.bces.bcesp(xdata,errx,ydata,erry,covxy,nboot)
	
	# Plots best-fit
	if xplot==None:
		x=numpy.linspace(xdata.min(),xdata.max(),100)
	else:
		x=xplot
	pylab.plot(x,a[i]*x+b[i],linestyle,**args)

	fitm=numpy.array([ a[i],b[i] ])	# array with best-fit parameters
	covm=numpy.array([ (erra[i]**2,cov[i]), (cov[i],errb[i]**2) ])	# covariance matrix
	def func(x): return x[1]*x[0]+x[2]

	# Plots confidence band
	lcb,ucb,xcb=stats.confbandnl(xdata,ydata,func,fitm,covm,2,conf,x)
	if front==True:
		zorder=10
	else:
		zorder=None
	pylab.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor=confcolor, zorder=zorder)
	
	return a,b,erra,errb,cov





def fitconfmc(xdata,ydata,errx,erry,covxy,nboot=1000,bcesMethod='ort',linestyle='',conf=1.,confcolor='gray',xplot=None,front=False,**args):
	"""
	This is a wrapper that given the input data performs the BCES
	fit, get the orthogonal parameters and plot the best-fit line and
	confidence band (generated using MC). I decided to put together these 
	commands in a method because I have been using them very frequently.
	
	Assumes you initialized the plot window before calling this method.
	This method is more stable than fitconf, which is plagued with numerical 
	instabilities when computing the gradient.
	
	Usage:

	>>> a1,b1,erra1,errb1,cov1=nemmen.fitconf(x[i],y[i],errx[i],erry[i],covxy[i],nboot,bces,linestyle='k',confcolor='LightGrey')
	
	Explanation of some arguments:
	- xplot: if provided, will compute the confidence band in the X-values provided
	with xplot
	- front: if True, then will plot the confidence band in front of the data
	points; otherwise, will plot it behind the points
	- conf: size of confidence band to be plotted in standard deviations
	"""	
	import bces.bces
	from . import misc	

	# Selects the desired BCES method
	i=misc.whichbces(bcesMethod)
		
	# Performs the BCES fit
	a,b,erra,errb,cov=bces.bces.bcesp(xdata,errx,ydata,erry,covxy,nboot)
	
	# Plots best-fit
	if xplot==None:
		x=numpy.linspace(xdata.min(),xdata.max(),100)
	else:
		x=xplot
	pylab.plot(x,a[i]*x+b[i],linestyle,**args)

	fitm=numpy.array([ a[i],b[i] ])	# array with best-fit parameters
	covm=numpy.array([ (erra[i]**2,cov[i]), (cov[i],errb[i]**2) ])	# covariance matrix

	# Plots confidence band
	lcb,ucb,y=confbandmc(x,fitm,covm,10000,conf)
	if front==True:
		zorder=10
	else:
		zorder=None
	pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor=confcolor, zorder=zorder)
	
	return a,b,erra,errb,cov










def plotlinfit(xdata,ydata,a,b,erra,errb,cov,linestyle='',conf=0.683,confcolor='gray',xplot=None,front=False,**args):
	"""
	This is a wrapper that given the output data from a linear regression
	method (for example, bayeslin.pro, the Bayesian linear regression method 
	of Kelly (2007)), it plots the fits and the confidence bands.
	The input is:
	X, Y, slope (A), errA, intercept (B), errB and cov(A,B)
	
	Assumes you initialized the plot window before calling this method.
	
	Usage:

	>>> nemmen.plotlinfit(x,y,a,b,erra,errb,covab,linestyle='k',confcolor='LightGrey')
	
	Explanation of some arguments:
	- xplot: if provided, will compute the confidence band in the X-values provided
	with xplot
	- front: if True, then will plot the confidence band in front of the data
	points; otherwise, will plot it behind the points
	"""			
	# Plots best-fit
	if xplot==None:
		x=numpy.linspace(xdata.min(),xdata.max(),100)
	else:
		x=xplot
	pylab.plot(x,a*x+b,linestyle,**args)

	fitm=numpy.array([ a,b ])	# array with best-fit parameters
	covm=numpy.array([ (erra**2,cov), (cov,errb**2) ])	# covariance matrix
	def func(x): return x[1]*x[0]+x[2]

	# Plots confidence band
	lcb,ucb,xcb=confbandnl(xdata,ydata,func,fitm,covm,2,conf,x)
	if front==True:
		zorder=10
	else:
		zorder=None
	pylab.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor=confcolor, zorder=zorder)
	




def jh(xdata,ydata,errx,erry,covxy,nboot=1000,bces='ort',linestyle='',conf=0.683,confcolor='gray',xplot=None,front=False,**args):
	"""
	This is a wrapper that given the input data performs the BCES
	fit, get the orthogonal parameters, best-fit line and
	confidence band. Then returns the points corresponding to the line and
	confidence band. 

	I wrote this for the John Hunter plotting contest, in order to simplify
	my AGN-GRB plot. Inherited from method fitconf.
	
	Usage:

	>>> x,y,lcb,ucb=nemmen.fitconf(x[i],y[i],errx[i],erry[i],covxy[i],nboot,bces,linestyle='k',confcolor='LightGrey')

	where y are the line points, lcb and ucb are the lower and upper confidence band
	points.

	:param xplot: if provided, will compute the confidence band in the X-values provided
	  with xplot
	:param front: if True, then will plot the confidence band in front of the data
	  points; otherwise, will plot it behind the points
	"""	
	# Selects the desired BCES method
	i=whichbces(bces)
		
	# Performs the BCES fit
	a,b,erra,errb,cov=bcesp(xdata,errx,ydata,erry,covxy,nboot)
	
	# Plots best-fit
	if xplot==None:
		x=numpy.linspace(xdata.min(),xdata.max(),100)
	else:
		x=xplot
	y=a[i]*x+b[i]

	fitm=numpy.array([ a[i],b[i] ])	# array with best-fit parameters
	covm=numpy.array([ (erra[i]**2,cov[i]), (cov[i],errb[i]**2) ])	# covariance matrix
	def func(x): return x[1]*x[0]+x[2]

	# Plots confidence band
	lcb,ucb,xcb=confbandnl(xdata,ydata,func,fitm,covm,2,conf,x)
	
	return x,y,lcb,ucb










def fitconfpred(xdata,ydata,errx,erry,covxy,nboot=1000,bces='ort',linestyle='',conf=0.68,confcolor='LightGrey',predcolor='Khaki',xplot=None,front=False,**args):
	"""
	This is a wrapper that given the input data performs the BCES
	fit, get the orthogonal parameters and plot (i) the best-fit line,
	(ii) confidence band and (iii) prediction band. 
	
	I decided to put together these commands in a method because I have been 
	using them very frequently.
	
	Assumes you initialized the plot window before calling this method.
	
	Usage:

	>>> a1,b1,erra1,errb1,cov1=nemmen.fitconfpred(x[i],y[i],errx[i],erry[i],covxy[i],nboot,bces,linestyle='k',confcolor='LightGrey')
	"""	
	# Selects the desired BCES method
	i=whichbces(bces)
		
	# Performs the BCES fit
	a,b,erra,errb,cov=bcesp(xdata,errx,ydata,erry,covxy,nboot)
	
	# Plots best-fit
	if xplot==None:
		x=numpy.linspace(xdata.min(),xdata.max(),100)
	else:
		x=xplot
	pylab.plot(x,a[i]*x+b[i],linestyle,**args)

	fitm=numpy.array([ a[i],b[i] ])	# array with best-fit parameters
	covm=numpy.array([ (erra[i]**2,cov[i]), (cov[i],errb[i]**2) ])	# covariance matrix
	def func(x): return x[1]*x[0]+x[2]
	
	if front==True:
		zorder=10
	else:
		zorder=None
	
	# Plots prediction band
	lpb,upb,xpb=predbandnl(xdata,ydata,func,fitm,covm,2,conf,x)
	pylab.fill_between(xpb, lpb, upb, facecolor=predcolor,edgecolor='', zorder=zorder)
	
	# Plots confidence band
	lcb,ucb,xcb=confbandnl(xdata,ydata,func,fitm,covm,2,conf,x)
	pylab.fill_between(xcb, lcb, ucb, facecolor=confcolor,edgecolor='', zorder=zorder)
	
	return a,b,erra,errb,cov




def fitpred(xdata,ydata,errx,erry,covxy,nboot=1000,bces='ort',linestyle='',conf=0.68,predcolor='Khaki',xplot=None,front=False,**args):
	"""
	This is a wrapper that given the input data performs the BCES
	fit, get the orthogonal parameters and plot (i) the best-fit line and
	(ii) prediction band. 
	
	I decided to put together these commands in a method because I have been 
	using them very frequently.
	
	Assumes you initialized the plot window before calling this method.
	
	Usage:

	>>> a1,b1,erra1,errb1,cov1=nemmen.fitpred(x[i],y[i],errx[i],erry[i],covxy[i],nboot,bces,linestyle='k',predcolor='LightGrey')
	"""	
	# Selects the desired BCES method
	i=whichbces(bces)
		
	# Performs the BCES fit
	a,b,erra,errb,cov=bcesp(xdata,errx,ydata,erry,covxy,nboot)
	
	# Plots best-fit
	if xplot==None:
		x=numpy.linspace(xdata.min(),xdata.max(),100)
	else:
		x=xplot
	pylab.plot(x,a[i]*x+b[i],linestyle,**args)

	fitm=numpy.array([ a[i],b[i] ])	# array with best-fit parameters
	covm=numpy.array([ (erra[i]**2,cov[i]), (cov[i],errb[i]**2) ])	# covariance matrix
	def func(x): return x[1]*x[0]+x[2]
	
	if front==True:
		zorder=10
	else:
		zorder=None
	
	# Plots prediction band
	lpb,upb,xpb=predbandnl(xdata,ydata,func,fitm,covm,2,conf,x)
	pylab.fill_between(xpb, lpb, upb, facecolor=predcolor,edgecolor='', zorder=zorder)
	
	return a,b,erra,errb,cov






def uerrorbar(ux,uy,**args):
	"""
Adaptation of pylab.errorbar to work with arrays defined using the
uncertainties package, which include the errorbars built-in.

Usage:

>>> uerrorbar(x,y,fmt='o')

will plot the points and error bars associated with the 'unumpy'
arrays x and y
	"""	
	x=unumpy.nominal_values(ux)
	y=unumpy.nominal_values(uy)
	errx=unumpy.std_devs(ux)
	erry=unumpy.std_devs(uy)
	
	pylab.errorbar(x,y,xerr=errx,yerr=erry,**args)



def text(x,y,s,**args):
	"""
Version of pylab.text that can be applied to arrays.

Usage:

>>> text(x,y,s, fontsize=10)

will plot the strings in array 's' at coordinates given by arrays
'x' and 'y'.
	"""
	for j in range(x.size):
		pylab.text(x[j],y[j],s[j], **args)



def ipyplots():
	"""
Makes sure we have exactly the same matplotlib settings as in the IPython terminal 
version. Call this from IPython notebook.

`Source <http://stackoverflow.com/questions/16905028/why-is-matplotlib-plot-produced-from-ipython-notebook-slightly-different-from-te)>`_.
	"""
	pylab.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
	pylab.rcParams['font.size']=12                #10 
	pylab.rcParams['savefig.dpi']=100             #72 
	pylab.rcParams['figure.subplot.bottom']=.1    #.125




def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.

    source: http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
    Chris Slocum, Colorado State University
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


def image(Z,xnew,ynew,my_cmap=None,aspect='equal'):
	"""
Creates pretty image. You need to specify:
	"""
	imshow(log10(Z),extent=[xnew[0],xnew[-1],ynew[0],ynew[-1]], cmap=my_cmap)
	pylab.axes().set_aspect('equal')
	colorbar()
	circle2=Circle((0,0),1,color='k')
	gca().add_artist(circle2)
	savefig('tmp.png',transparent=True,dpi=150)




def wolframcmap():
	"""
Returns colormap that matches closely the one used by default
for images in Wolfram Mathematica 11 (dark blue to orange).

I spent one hour playing around to reproduce it.

Usage:

>>> mycmap=nmmn.plots.wolframcmap()
>>> imshow(rho, cmap=mycmap)

.. figure:: ../figures/wolframcmap.png
	:scale: 100 %
	:alt: Image plotted using Wolfram's colormap

	Image plotted using Wolfram's colormap.
	"""
	# Create a list of RGB tuples, recreates Mathematica colormap
	colors3=[(51,91,150),(111,116,143),(167,136,110),(233,167,85),(251,212,141),(255,247,190)]

	# Call the function make_cmap which returns your colormap
	return make_cmap(colors3, bit=True)





def parulacmap():
	"""
	Creates the beautiful Parula colormap which is Matlab's default.

	Usage:

	>>> mycmap=nmmn.plots.parulacmap()
	>>> imshow(rho, cmap=mycmap)

	Code taken from `here <https://github.com/BIDS/colormap/blob/master/parula.py>`_
	"""
	from matplotlib.colors import LinearSegmentedColormap

	cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
	 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
	 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
	  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
	 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
	  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
	 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
	  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
	 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
	  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
	 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
	  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
	 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
	  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
	  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
	 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
	  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
	 [0.0589714286, 0.6837571429, 0.7253857143], 
	 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
	 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
	  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
	 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
	  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
	 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
	  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
	 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
	  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
	 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
	 [0.7184095238, 0.7411333333, 0.3904761905], 
	 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
	  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
	 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
	 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
	  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
	 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
	  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
	 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
	 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
	 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
	  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
	 [0.9763, 0.9831, 0.0538]]

	return LinearSegmentedColormap.from_list('parula', cm_data)


def turbocmap():
	"""
	Returns the Turbo colormap: an improved version of the awful jet colormap.

	The look-up table contains 256 entries. Each entry is a floating point sRGB triplet.

	Usage:

	>>> turbo=nmmn.plots.turbocmap()
	>>> imshow(rho, cmap=turbo)

	Copyright 2019 Google LLC.
	SPDX-License-Identifier: Apache-2.0
	Author: Anton Mikhailov

	References:

	- `turbo colormap array <https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f>`_
	- Google AI `blog post <https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html>`_ describing the advantages of the colormap
	"""
	from matplotlib.colors import ListedColormap

	turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]

	return ListedColormap(turbo_colormap_data)











def jointplot(X,Y,xlabel=None,ylabel=None,binsim=40,binsh=20,contour=True):
	"""
Plots the joint distribution of posteriors for X1 and X2, including the 1D
histograms showing the median and standard deviations.

The work that went in creating this nice method is shown, step by step, in 
the ipython notebook "error contours.ipynb". Sources of inspiration:

- http://python4mpia.github.io/intro/quick-tour.html
- http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals

Usage:

>>> jointplot(M.rtr.trace(),M.mdot.trace(),xlabel='$\log \ r_{\\rm tr}$', ylabel='$\log \ \dot{m}$')

gives the following plot.

.. figure:: ../figures/jointplot.png
   :scale: 100 %
   :alt: Two-dimensional kernel density distribution.

   Two-dimensional kernel density distribution, along with one-dimensional histograms of each distribution.
	"""
	import scipy.stats

	# Generates 2D histogram for image
	histt, xt, yt = numpy.histogram2d(X, Y, bins=[binsim,binsim], normed=False)
	histt = numpy.transpose(histt)  # Beware: numpy switches axes, so switch back.

	# assigns correct proportions to subplots
	fig=pylab.figure()
	gs = pylab.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3], wspace=0.001, hspace=0.001)
	con=pylab.subplot(gs[2])
	histx=pylab.subplot(gs[0], sharex=con)
	histy=pylab.subplot(gs[3], sharey=con)
		
	# Image
	con.imshow(histt,extent=[xt[0],xt[-1], yt[0],yt[-1]],origin='lower',cmap=pylab.cm.gray_r,aspect='auto')

	# Overplot with error contours 1,2 sigma
	if contour==True:
		pdf = scipy.stats.gaussian_kde([X, Y])
		x,y = pylab.meshgrid(xt,yt)
		z = numpy.array(pdf.evaluate([x.flatten(),y.flatten()])).reshape(x.shape)
		# the [61,15] values were obtained by trial and error until the joint confidence 
		# contours matched the confidence intervals from the individual X,Y
		s=scipy.stats.scoreatpercentile(pdf(pdf.resample(1000)), [61,15])
		cs=con.contour(x,y,z, levels=s, extent=[x[0],x[-1], y[0],y[-1]], linestyles=['-','-','-'], colors=['black','blue'])
		# use dictionary in order to assign your own labels to the contours.
		#fmtdict = {s[0]:r'$1\sigma$',s[1]:r'$2\sigma$'}
		#con.clabel(cs, fmt=fmtdict, inline=True, fontsize=20)
		if xlabel!=None: con.set_xlabel(xlabel)
		if ylabel!=None: con.set_ylabel(ylabel)

	# X-axis histogram
	histx.hist(X, binsh, histtype='stepfilled',facecolor='lightblue')
	pylab.setp(histx.get_xticklabels(), visible=False)	# no X label
	pylab.setp(histx.get_yticklabels(), visible=False)	# no Y label
	# Vertical lines with median and 1sigma confidence
	yax=histx.set_ylim()
	histx.plot([numpy.median(X),numpy.median(X)],[yax[0],yax[1]],'k-',linewidth=2) # median
	xsd=scipy.stats.scoreatpercentile(X, [15.87,84.13])
	histx.plot([xsd[0],xsd[0]],[yax[0],yax[1]],'k--') # -1sd
	histx.plot([xsd[-1],xsd[-1]],[yax[0],yax[1]],'k--') # +1sd

	# Y-axis histogram
	histy.hist(Y, binsh, histtype='stepfilled', orientation='horizontal',facecolor='lightyellow')
	pylab.setp(histy.get_yticklabels(), visible=False)	# no Y label
	pylab.setp(histy.get_xticklabels(), visible=False)	# no X label
	# Vertical lines with median and 1sigma confidence
	xax=histy.set_xlim()
	histy.plot([xax[0],xax[1]],[numpy.median(Y),numpy.median(Y)],'k-',linewidth=2) # median
	ysd=scipy.stats.scoreatpercentile(Y, [15.87,84.13])
	histy.plot([xax[0],xax[1]],[ysd[0],ysd[0]],'k--') # -1sd
	histy.plot([xax[0],xax[1]],[ysd[-1],ysd[-1]],'k--') # +1sd




def symlog(x, C=1./numpy.log(10.)):
	"""
Applies a modified logarithm function to x that handles negative 
values while maintaining continuity across 
zero. This function solves a very concrete problem: how to handle
data that spans a huge range and has also negative values? log10
will fail. This is the answer. 

The transformation is defined in an article from the journal 
Measurement Science and Technology (Webber, 2012): 

y = sign(x)*(log10(1+abs(x)/(10^C))) 

where the scaling constant C determines the resolution of the data 
around zero. The smallest order of magnitude shown on either side of 
zero will be 10^ceil(C).

Reference: MATHWORKS symlog <https://www.mathworks.com/matlabcentral/fileexchange/57902-symlog>
	"""
	return numpy.sign(x)*(numpy.log10(1+numpy.abs(x)/(10**C)))




def lineWidth(value,max=1,lw_min=0.5,lw_max=10):
    """
Ascribes a linewidth based on the input value. Useful for
plots where the linewidth changes depending on some weight.

:param value: value that will be used to get the line width
:param max: maximum value for normalization
:param lw_min: minimum line width
:param lw_max: max line width
:returns: line width for plotting

Example: get a sequence of stock price time series and plot them according
to their relative importance in your portfolio.

>>> a=[]
>>> for i in tqdm.tqdm(range(size(acoes))): 
>>>    a.append(yf.download(tickers=acoes['col2'][i]+".SA", period=periodo, interval=dt, progress=False))
>>> for i, acao in enumerate(a): 
>>>    ar=nf.returnsTS(acao)
>>>    ar.plot(label=nome[i],lw=lineWidth(fracao[i],max(fracao)), alpha=lineWidth(fracao[i],max(fracao),0.3,1)) 

.. figure:: ../figures/lineWidth-example.png
   :scale: 100 %
   :alt: Illustrating lineWidth method.
    
ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    return value/max*(lw_max - lw_min) + lw_min
