# Fancy plots
# ==============

def onehist(x,xlabel='',fontsize=12):
	""" Script that plots the histogram of x with the corresponding xlabel. """

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

Inspired by http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label.

v1 Dec. 2011
v1.1 Feb. 2012: Added sharey argument.
v1.2 Feb. 2012: added fontsize argument.
v1.3 Apr. 2012: added fig argument.
v1.4 Apr. 2013: added range argument.
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
- xmin,xmax: lower and upper range of plotted values, will be used to set a consistent x-range
	for both histograms.
- x1leg, x2leg, x3leg: legends for each histogram	
- xlabel: self-explanatory.
- sharey: sharing the Y-axis among the histograms?
- fig: which plot window should I use?

Example:
x1=Lbol(AD), x2=Lbol(JD), x3=Lbol(EHF10)
>>> threehists(x1,x2,x3,38,44,'AD','JD','EHF10','$\log L_{\\rm bol}$ (erg s$^{-1}$)',sharey=True)

Inspired by http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label.

v1 Dec. 2011
v1.1 Feb. 2012:	Added sharey keyword.
v1.2 Feb. 2012: added fontsize argument.
v1.3 Apr. 2012: added fig argument.
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
	for both histograms.
- x1leg, x2leg, x3leg, x4leg: legends for each histogram	
- xlabel: self-explanatory.
- sharey: sharing the Y-axis among the histograms?
- bins1,bins2,...: number of bins in each histogram
- fig: which plot window should I use?
- line?: draws vertical solid lines at the positions indicated in each panel
- line?b: draws vertical dashed lines at the positions indicated in each panel

Inspired by http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label.

v1 Apr. 2012: inherited from threehists.
v1.5 Jun 2012: added option for drawing vertical lines in each histogram (line?)
v1.6 Jul 2012: added option for line?b
v1.7 Oct 2012: added option 'loc'
v1.8 Mar 2013: added 'histtype' option in histogram plots
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

Inspired by http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label.

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

v1 Dec. 2011
v1.1 Feb. 2012:	Added sharey keyword.
v1.2 Feb. 2012: added fontsize argument.
v1.3 Apr. 2012: added fig argument.
v1.4 Jun. 2012: no sharing of X-axis, removed some arguments
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

	
	

	
def fitconf(xdata,ydata,errx,erry,covxy,nboot=1000,bces='ort',linestyle='',conf=0.683,confcolor='gray',xplot=None,front=False,**args):
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

	# Plots confidence band
	lcb,ucb,xcb=confbandnl(xdata,ydata,func,fitm,covm,2,conf,x)
	if front==True:
		zorder=10
	else:
		zorder=None
	pylab.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor=confcolor, zorder=zorder)
	
	return a,b,erra,errb,cov





def fitconfmc(xdata,ydata,errx,erry,covxy,nboot=1000,bces='ort',linestyle='',conf=1.,confcolor='gray',xplot=None,front=False,**args):
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

Source: http://stackoverflow.com/questions/16905028/why-is-matplotlib-plot-produced-from-ipython-notebook-slightly-different-from-te).
	"""
	pylab.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
	pylab.rcParams['font.size']=12                #10 
	pylab.rcParams['savefig.dpi']=100             #72 
	pylab.rcParams['figure.subplot.bottom']=.1    #.125




