"""
Methods for dealing with Bayesian statistics
==============================================

e.g. priors, posteriors, joint density plots.

Right now the module is focused around PyMC, but I am migrating
to emcee.

.. todo:: plot confidence/credibility interval of a model
"""

import numpy, pylab, scipy, scipy.stats





def joint_density(X, Y, bounds=None):
	"""
Plots joint distribution of variables.
Inherited from method in src/graphics.py module in project 
git://github.com/aflaxman/pymc-example-tfr-hdi.git
	"""
	if bounds:
		X_min, X_max, Y_min, Y_max = bounds
	else:
		X_min = X.min()
		X_max = X.max()
		Y_min = Y.min()
		Y_max = Y.max()

	pylab.plot(X, Y, linestyle='none', marker='o', color='green', mec='green', alpha=.2, zorder=-99)

	gkde = scipy.stats.gaussian_kde([X, Y])
	x,y = pylab.mgrid[X_min:X_max:(X_max-X_min)/25.,Y_min:Y_max:(Y_max-Y_min)/25.]
	z = pylab.array(gkde.evaluate([x.flatten(), y.flatten()])).reshape(x.shape)
	pylab.contour(x, y, z, linewidths=2)

	pylab.axis([X_min, X_max, Y_min, Y_max])







def allplot(xb,yb,bins=30,fig=1,xlabel='x',ylabel='y'):
	"""
Input:
X,Y : objects referring to the variables produced by PyMC that you want
to analyze. Example: X=M.theta, Y=M.slope.

Inherited from Tommy LE BLANC's code at astroplotlib|STSCI.
	"""
	#X,Y=xb.trace(),yb.trace()
	X,Y=xb,yb

	#pylab.rcParams.update({'font.size': fontsize})
	fig=pylab.figure(fig)
	pylab.clf()
	
	gs = pylab.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3], wspace=0.07, hspace=0.07)
	scat=pylab.subplot(gs[2])
	histx=pylab.subplot(gs[0], sharex=scat)
	histy=pylab.subplot(gs[3], sharey=scat)
	#scat=fig.add_subplot(2,2,3)
	#histx=fig.add_subplot(2,2,1, sharex=scat)
	#histy=fig.add_subplot(2,2,4, sharey=scat)
	
	# Scatter plot
	scat.plot(X, Y,linestyle='none', marker='o', color='green', mec='green',alpha=.2, zorder=-99)

	gkde = scipy.stats.gaussian_kde([X, Y])
	x,y = numpy.mgrid[X.min():X.max():(X.max()-X.min())/25.,Y.min():Y.max():(Y.max()-Y.min())/25.]
	z = numpy.array(gkde.evaluate([x.flatten(), y.flatten()])).reshape(x.shape)
	scat.contour(x, y, z, linewidths=2)
	scat.set_xlabel(xlabel)
	scat.set_ylabel(ylabel)

	# X-axis histogram
	histx.hist(X, bins, histtype='stepfilled')
	pylab.setp(histx.get_xticklabels(), visible=False)	# no X label
	#histx.xaxis.set_major_formatter(pylab.NullFormatter())	# no X label

	# Y-axis histogram
	histy.hist(Y, bins, histtype='stepfilled', orientation='horizontal')
	pylab.setp(histy.get_yticklabels(), visible=False)	# no Y label
	#histy.yaxis.set_major_formatter(pylab.NullFormatter())	# no Y label

	#pylab.minorticks_on()
	#pylab.subplots_adjust(hspace=0.1)
	#pylab.subplots_adjust(wspace=0.1)
	pylab.draw()
	pylab.show()









def jointplotx(X,Y,xlabel=None,ylabel=None,binsim=40,binsh=20,binscon=15):
	"""
Plots the joint distribution of posteriors for X1 and X2, including the 1D
histograms showing the median and standard deviations. Uses simple method
for drawing the confidence contours compared to jointplot (which is wrong).

The work that went in creating this method is shown, step by step, in 
the ipython notebook "error contours.ipynb". Sources of inspiration:
- http://python4mpia.github.io/intro/quick-tour.html

Usage:
>>> jointplot(M.rtr.trace(),M.mdot.trace(),xlabel='$\log \ r_{\\rm tr}$', ylabel='$\log \ \dot{m}$')
	"""
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
	# Contour plot
	histdata, x, y = numpy.histogram2d(X, Y, bins=[binscon,binscon], normed=False)
	histdata = numpy.transpose(histdata)  # Beware: numpy switches axes, so switch back.
	pmax  = histdata.max()
	cs=con.contour(histdata, levels=[0.68*pmax,0.05*pmax], extent=[x[0],x[-1], y[0],y[-1]], colors=['black','blue'])
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

