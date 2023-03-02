"""
Statistical methods
=====================

- fit residuals
- Computing prediction and confidence bands
- Comparing goodness-of-fit of different models
- operations on statistical distributions
- custom statistical distributions
- p-values and significance
"""

import numpy as np
import scipy
import scipy.stats
from . import misc





# Residuals of a fit
# ===================
# Residual sum of squares, raw scatter about best-fit and 
# intrinsic scatter.

def scatterfit(x,y,a=None,b=None):
	"""
Compute the mean deviation of the data about the linear model given if A,B
(*y=ax+b*) provided as arguments. Otherwise, compute the mean deviation about 
the best-fit line.

:param x,y: assumed to be Numpy arrays. 
:param a,b: scalars.
:rtype: float sd with the mean deviation.
	"""

	if a==None:	
		# Performs linear regression
		a, b, r, p, err = scipy.stats.linregress(x,y)
	
	# Std. deviation of an individual measurement (Bevington, eq. 6.15)
	N=np.size(x)
	sd=1./(N-2.)* np.sum((y-a*x-b)**2)
	sd=np.sqrt(sd)
	
	return sd
	

def scatterfitg(ydata,ymod,deg):
	"""
Compute the mean deviation of the data about the model given in
the array ymod, tabulated exactly like ydata.

Usage:

>>> sd=scatterfitg(ydata,ymod,n)

:param ydata: data
:param ymod: model evaluated at xdata
:param n: number of free parameters in the model
  
:type ydata,ymod: Numpy arrays
:rtype: float sd with the mean deviation.
	"""	
	# Std. deviation of an individual measurement (Bevington, eq. 6.15)
	N=np.size(ydata)
	sd=1./(N-deg)* np.sum((ydata-ymod)**2)
	sd=np.sqrt(sd)
	
	return sd



def scatpratt(x,y,errx,erry,a,b):
	"""
This is an alternative way of computing the "raw" scatter about the best-fit 
linear relation in the presence of measurement errors, proposed by 
Pratt et al. 2009, A&A, 498, 361. 

In the words of Cavagnolo et al. (2010): 
*We have quantified the total scatter about the best-fit relation using a 
weighted estimate of the orthogonal distances to the best-fit line (see 
Pratt et al. 2009).*
	
Usage:

.. function:: sd=scatpratt(x,y,errx,erry,a,b)

  :param x,y: X and Y data arrays
  :param errx, erry: standard deviations in X and Y
  :param a,b: slope and intercept of best-fit linear relation
  :rtype: float sd with the scatter about the best-fit.

v1 Mar 20 2012
	"""
	N=np.size(x)
	
	# Equation 4 from Pratt et al. 2009
	sdsq=erry**2+a**2*errx**2
	wden=1./N*np.sum(1./sdsq)	# Denominator of w
	w=1./sdsq/wden
	
	# Equation 3 from Pratt et al. 2009
	sdrawsq=1./(N-2.)*np.sum(w*(y-a*x-b)**2)
	sdraw=np.sqrt(sdrawsq)
	
	return sdraw



def residual(ydata,ymod):
	"""
Compute the residual sum of squares (RSS) also known as the sum of 
squared residuals (see http://en.wikipedia.org/wiki/Residual_sum_of_squares).

Usage:

>>> rss=residual(ydata,ymod)

where
  ydata : data
  ymod : model evaluated at xdata
  
ydata,ymod assumed to be Numpy arrays. 
Returns the float rss with the mean deviation.

Dec. 2011
	"""	
	# Std. deviation of an individual measurement (Bevington, eq. 6.15)
	N=np.size(ydata)
	rss=np.sum((ydata-ymod)**2)
	
	return rss



def chisq(x,y,a=None,b=None,sd=None):
	"""
Returns the chi-square error statistic as the sum of squared errors between
Y(i) and AX(i) + B. If individual standard deviations (array sd) are supplied, 
then the chi-square error statistic is computed as the sum of squared errors
divided by the standard deviations.	Inspired on the IDL procedure linfit.pro.
See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.
	
If a linear model is not provided via A,B (y=ax+b) then the method computes
the chi-square using the best-fit line.

x,y,sd assumed to be Numpy arrays. a,b scalars.
Returns the float chisq with the chi-square statistic.
	"""

	if a==None:	
		# Performs linear regression
		a, b, r, p, err = scipy.stats.linregress(x,y)
	
	# Chi-square statistic (Bevington, eq. 6.9)
	if sd==None:
		chisq=np.sum((y-a*x-b)**2)
	else:
		chisq=np.sum( ((y-a*x-b)/sd)**2 )
	
	return chisq




def chisqg(ydata,ymod,sd=None):
	"""
Returns the chi-square error statistic as the sum of squared errors between
Ydata(i) and Ymodel(i). If individual standard deviations (array sd) are supplied, 
then the chi-square error statistic is computed as the sum of squared errors
divided by the standard deviations.	Inspired on the IDL procedure linfit.pro.
See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

x,y,sd assumed to be Numpy arrays. a,b scalars.
Returns the float chisq with the chi-square statistic.
	"""
	# Chi-square statistic (Bevington, eq. 6.9)
	if sd==None:
		chisq=np.sum((ydata-ymod)**2)
	else:
		chisq=np.sum( ((ydata-ymod)/sd)**2 )
	
	return chisq






def chisqxy(x,y,errx,erry,a,b):
	"""
Returns the chi-square error statistic for a linear fit, 
computed taking into account the errors in both X and Y (i.e. the
effective variance). See equation 3 in Ascenso et al. 2012, A&A or
Yee & Ellingson 2003 ApJ.

Usage:

>>> chisq=chisqxy(xdata,ydata,errx,erry,a,b)

where

- xdata,ydata : data
- errx,erry : measurement uncertainties in the data
- a,b : slope and intercept of the best-fit linear regression model
	"""
	sdsq=erry**2+a**2*errx**2
	chisq=np.sum( (y-a*x-b)**2/sdsq )
	
	return chisq





def intscat(x,y,errx,erry,a,b):
	"""
Estimates the intrinsic scatter about the best-fit, taking into account
the errors in X and Y and the "raw" scatter. Inspired by Pratt et al. 2009,
A&A, 498, 361. 
	
Usage:

>>> sd=intscat(x,y,errx,erry,a,b)

where
  
- x,y : X and Y data arrays
- errx, erry : standard deviations in X and Y
- a,b : slope and intercept of best-fit linear relation
  
Returns the float sd with the scatter about the best-fit.

v1 Apr 9th 2012
	"""
	# Raw scatter
	sdraw=scatpratt(x,y,errx,erry,a,b)
	sdrawsq=sdraw**2
	
	# Statistical variance, eq. 4 from Pratt et al. 2009
	sdsq=erry**2+a**2*errx**2
	
	# Intrinsic scatter for each data point
	intscati=np.sqrt( np.abs(sdsq-sdrawsq) )
	
	return np.mean(intscati)
	
	


def linregress_error(x,y):
	"""
Compute the uncertainties in the parameters A,B of the least-squares
linear regression fit y=ax+b to the data (scipy.stats.linregress).

x,y assumed to be Numpy arrays.
Returns the sequence sigma_a, sigma_b with the standard deviations in A and B.
	"""
	
	# Performs linear regression
	a, b, r, p, err = scipy.stats.linregress(x,y)
	
	# Std. deviation of an individual measurement (Bevington, eq. 6.15)
	N=np.size(x)
	sd=1./(N-2.)* np.sum((y-a*x-b)**2);	sd=np.sqrt(sd)
	
	# Std. deviation in parameters
	Delta=N*np.sum(x**2)-(np.sum(x))**2	# Bevington, eq. 6.13
	sdb=sd**2/Delta*np.sum(x**2);	sdb=np.sqrt(sdb)	# Bevington, eq. 6.23
	sda=N*sd**2/Delta;	sda=np.sqrt(sda)
	
	return sda,sdb
		
	










	


# Computing prediction and confidence bands
# ===========================================
#
#


def confband(xd,yd,a,b,conf=0.95,x=None):
	"""
Calculates the confidence band of the linear regression model at the desired confidence
level, using analytical methods. The 2sigma confidence interval is 95% sure to contain 
the best-fit regression line. This is not the same as saying it will contain 95% of 
the data points.

Arguments:

- conf: desired confidence level, by default 0.95 (2 sigma)
- xd,yd: data arrays
- a,b: linear fit parameters as in y=ax+b
- x: (optional) array with x values to calculate the confidence band. If none is provided, will by default generate 100 points in the original x-range of the data.
  
Returns:
Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands 
corresponding to the [input] x array.

Usage:

>>> lcb,ucb,x=nemmen.confband(all.kp,all.lg,a,b,conf=0.95)
calculates the confidence bands for the given input arrays

>>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
plots a shaded area containing the confidence band

References:
1. http://en.wikipedia.org/wiki/Simple_linear_regression, see Section Confidence intervals
2. http://www.weibull.com/DOEWeb/confidence_intervals_in_simple_linear_regression.htm

v1 Dec. 2011
v2 Jun. 2012: corrected bug in computing dy
	"""
	alpha=1.-conf	# significance
	n=xd.size	# data sample size

	if x==None: x=np.linspace(xd.min(),xd.max(),100)

	# Predicted values (best-fit model)
	y=a*x+b

	# Auxiliary definitions
	sd=scatterfit(xd,yd,a,b)	# Scatter of data about the model
	sxd=np.sum((xd-xd.mean())**2)
	sx=(x-xd.mean())**2	# array

	# Quantile of Student's t distribution for p=1-alpha/2
	q=scipy.stats.t.ppf(1.-alpha/2.,n-2)

	# Confidence band
	dy=q*sd*np.sqrt( 1./n + sx/sxd )
	ucb=y+dy	# Upper confidence band
	lcb=y-dy	# Lower confidence band

	return lcb,ucb,x
	
	
	
	
	
def predband(xd,yd,a,b,conf=0.95,x=None):
	"""
Calculates the prediction band of the linear regression model at the desired confidence
level, using analytical methods. 

Clarification of the difference between confidence and prediction bands:
"The 2sigma confidence interval is 95% sure to contain the best-fit regression line. 
This is not the same as saying it will contain 95% of the data points. The prediction bands are
further from the best-fit line than the confidence bands, a lot further if you have many data 
points. The 95% prediction interval is the area in which you expect 95% of all data points to fall."
(from http://graphpad.com/curvefit/linear_regression.htm)

Arguments:

- conf: desired confidence level, by default 0.95 (2 sigma)
- xd,yd: data arrays
- a,b: linear fit parameters as in y=ax+b
- x: (optional) array with x values to calculate the confidence band. If none is provided, will
  by default generate 100 points in the original x-range of the data.
  
Usage:

>>> lpb,upb,x=nemmen.predband(all.kp,all.lg,a,b,conf=0.95)
calculates the prediction bands for the given input arrays

>>> pylab.fill_between(x, lpb, upb, alpha=0.3, facecolor='gray')
plots a shaded area containing the prediction band  

:returns: Sequence (lpb,upb,x) with the arrays holding the lower and upper confidence bands 
corresponding to the [input] x array.

References:

1. `Introduction to Simple Linear Regression, Gerard 
E. Dallal, Ph.D. <http://www.JerryDallal.com/LHSP/slr.htm>`_
	"""
	alpha=1.-conf	# significance
	n=xd.size	# data sample size

	if x is None: x=np.linspace(xd.min(),xd.max(),100)

	# Predicted values (best-fit model)
	y=a*x+b

	# Auxiliary definitions
	sd=scatterfit(xd,yd,a,b)	# Scatter of data about the model
	sxd=np.sum((xd-xd.mean())**2)
	sx=(x-xd.mean())**2	# array

	# Quantile of Student's t distribution for p=1-alpha/2
	q=scipy.stats.t.ppf(1.-alpha/2.,n-2)

	# Prediction band
	dy=q*sd*np.sqrt( 1.+1./n + sx/sxd )
	upb=y+dy	# Upper prediction band
	lpb=y-dy	# Lower prediction band

	return lpb,upb,x
	



def confbandnl(xd,yd,fun,par,varcov,deg,conf=0.95,x=None):
	"""
Calculates the confidence band of a nonlinear model at the desired confidence
level, using analytical methods. 

Arguments:

- xd,yd: data arrays
- fun : function f(v) - the model - which returns a scalar. v is an array
  such that v[0]=x (scalar), v[i>0] = parameters of the model
- par : array or list with structure [par0, par1, par2, ...] with the best-fit
  parameters that will be fed into fun
- varcov : variance-covariance matrix obtained from the nonlinear fit
- deg : number of free parameters in the model
- conf: desired confidence level, by default 0.95 (2 sigma)
- x: (optional) array with x values to calculate the confidence band. If none is provided, will
  by default generate 100 points in the original x-range of the data.
  
Usage:

>>> lcb,ucb,x=nemmen.confbandnl(all.kp,all.lg,broken,bfit,bcov,4,conf=0.95)
calculates the confidence bands for the given input arrays

>>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
plots a shaded area containing the prediction band  

:returns: Sequence (lcb,ucb,x) with the arrays holding the lower and upper confidence bands 
corresponding to the [input] x array.

References:

1. `How does Prism compute confidence and prediction bands for nonlinear regression? <http://www.graphpad.com/faq/viewfaq.cfm?faq=1099>`_
2. http://stats.stackexchange.com/questions/15423/how-to-compute-prediction-bands-for-non-linear-regression
3. see also my notebook
	"""
	import numdifftools
	
	alpha=1.-conf	# significance
	n=xd.size	# data sample size

	if x is None: x=np.linspace(xd.min(),xd.max(),100)

	# Gradient (needs to be evaluated)
	dfun=numdifftools.Gradient(fun)
	
	# Quantile of Student's t distribution for p=1-alpha/2
	q=scipy.stats.t.ppf(1.-alpha/2.,n-2)

	# Residual sum of squares		
	rss=residual(yd, misc.evalfun(fun,xd,par) )
	
	grad,p=[],[]
	i=0
	y=misc.evalfun(fun,x,par)
	v=np.zeros_like(x)
		
	for i in range(x.size):
		# List: arrays consisting of [x[i], par1, par2, ...]
		p.append( np.concatenate(([x[i]],par)) )
	
		# List: each element -> gradient evaluated at each xi
		grad.append(dfun(p[i]))
				
		# Before processing the grad array, eliminates the first element
		temp=np.dot(np.transpose(grad[i][1:]), varcov)
		v[i]=np.dot(temp, grad[i][1:])
		
	# Confidence band
	dy=q*np.sqrt( v*rss/(n-deg) )
	ucb=y+dy	# Upper confidence band
	lcb=y-dy	# Lower confidence band

	return lcb,ucb,x





def predbandnl(xd,yd,fun,par,varcov,deg,conf=0.95,x=None):
	"""
Calculates the prediction band of a nonlinear model at the desired confidence
level, using analytical methods. 

Arguments:

- xd,yd: data arrays
- fun : function f(v) - the model - which returns a scalar. v is an array
  such that v[0]=x, v[i>0] = parameters of the model
- par : array or list with structure [par0, par1, par2, ...] with the best-fit
  parameters that will be fed into fun
- varcov : variance-covariance matrix obtained from the nonlinear fit
- deg : number of free parameters in the model
- conf: desired confidence level, by default 0.95 (2 sigma)
- x: (optional) array with x values to calculate the confidence band. If none is provided, will
  by default generate 100 points in the original x-range of the data.
  
Usage:

>>> lpb,upb,x=nemmen.predbandnl(all.kp,all.lg,broken,bfit,bcov,4,conf=0.95)
calculates the prediction bands for the given input arrays

>>> pylab.fill_between(x, lpb, upb, alpha=0.3, facecolor='gray')
plots a shaded area containing the prediction band  

:returns: Sequence (lpb,upb,x) with the arrays holding the lower and upper confidence bands 
corresponding to the [input] x array.

References:
1. http://www.graphpad.com/faq/viewfaq.cfm?faq=1099, "How does Prism compute confidence and prediction bands for nonlinear regression?"
2. http://stats.stackexchange.com/questions/15423/how-to-compute-prediction-bands-for-non-linear-regression
3. see also my notebook)
	"""
	import numdifftools
	
	alpha=1.-conf	# significance
	n=xd.size	# data sample size

	if x is None: x=np.linspace(xd.min(),xd.max(),100)

	# Gradient (needs to be evaluated)
	dfun=numdifftools.Gradient(fun)
	
	# Quantile of Student's t distribution for p=1-alpha/2
	q=scipy.stats.t.ppf(1.-alpha/2.,n-2)

	# Residual sum of squares		
	rss=residual(yd, misc.evalfun(fun,xd,par) )
	
	grad,p=[],[]
	i=0
	y=misc.evalfun(fun,x,par)
	v=np.zeros_like(x)
		
	for i in range(x.size):
		# List: arrays consisting of [x[i], par1, par2, ...]
		p.append( np.concatenate(([x[i]],par)) )
	
		# List: each element -> gradient evaluated at each xi
		grad.append(dfun(p[i]))
				
		# Before processing the grad array, eliminates the first element
		temp=np.dot(np.transpose(grad[i][1:]), varcov)
		v[i]=np.dot(temp, grad[i][1:])
		
	# Confidence band
	dy=q*np.sqrt( (1.+v)*rss/(n-deg) )
	upb=y+dy	# Upper prediction band
	lpb=y-dy	# Lower prediction band

	return lpb,upb,x
		
	


def confbandmc(x,par,varcov,n=100000,sigmas=1.):
	"""
Calculates the 1sigma confidence band of a linear model without needing
to specify the data points, by doing a Monte Carlo simulation using the
Multivariate normal distribution. Assumes the parameters follow a 2D
normal distribution.

Arguments:

- x: array with x values to calculate the confidence band. 
- par : array or list with structure [par0, par1, par2, ...] with the best-fit
  parameters
- varcov : variance-covariance matrix of the parameters
- n : number of (a,b) points generated from the multivariate gaussian
- sigmas : number of standard deviations contained within the prediction
  band

Output:

- lcb, ucb : lower and upper prediction bands
- y : values of the model tabulated at x

Usage:

>>> lcb,ucb,y=nemmen.confbandmc(x,bfit,bcov)
calculates the prediction bands for the given input arrays

>>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
plots a shaded area containing the prediction band
	"""
	# Generates many realizations of a and b from the multinormal distribution
	ar,br = np.random.multivariate_normal(par,varcov,n).T
	
	erry=[]	# will contain the std deviation in y
	y=[]	# values of y for each x
	
	for xi in x:
		yr=ar*xi+br
		erry.append( yr.std() )
		y.append( yr.mean() )	
	
	erry=np.array(erry)
	y=np.array(y)

	ucb=y+sigmas*erry	# Upper confidence band
	lcb=y-sigmas*erry	# Lower confidence band

	return lcb,ucb,y
	
	
	
	
def credbandmc(x,slope,inter,sigmas=1.):
	"""
Calculates the confidence (or credibility in case of Bayesian analysis) 
band of a linear regression model, given the posterior probability distributions
of the slope and intercept (presumably computed via Bayesian regression,
see bayeslin.pro). These distributions do not need to be normal.

Arguments:

- x: array with x values to calculate the confidence band. 
- slope, inter: posterior distributions of slope and intercept for linear
  regression
- sigmas : number of standard deviations contained within the confidence
  band

Output:

- lcb, ucb : lower and upper confidence bands
- y : best-fit values of the model tabulated at x

Usage:

>>> lcb,ucb,y=nemmen.predbandmc(x,a,b)
calculates the prediction bands for the given input arrays

>>> pylab.fill_between(x, lcb, ucb, alpha=0.3, facecolor='gray')
plots a shaded area containing the prediction band

v1 Jun. 2012: inspired by private communication with B. Kelly.
	"""
	a,b = slope,inter
	
	# Define the confidence/credibility interval
	conf=1.-scipy.special.erf(sigmas/np.sqrt(2.))
	
	lcb,ucb,ym=np.zeros_like(x),np.zeros_like(x),np.zeros_like(x)
	
	for i, xi in enumerate(x): 
		# Distribution of y for each x
		yp=a*xi+b	# 'p' as in posterior
		
		# Lower confidence band
		lcb[i]=scipy.stats.scoreatpercentile(yp,conf*100./2.)
		# Upper confidence band
		ucb[i]=scipy.stats.scoreatpercentile(yp,100.*(1-conf/2.))
		
		ym[i]=np.median(yp)	
		

	return lcb,ucb,ym
	
	

def confband_linboot(x,a,b,n):
	"""
	Computes information for plotting confidence band ("bow tie" plot) around
	best-fit. Run this after running `linboot` below.

	Usage:

	Plot confidence band and best-fit given data x,y

	>>> n=1000
	>>> a,b=nmmn.stats.linboot(x,y,n)
	>>> xMock,yBest,yStd=nmmn.stats.confband_linboot(x,a,b,n)
	>>> plt.plot(x,y,'o')
	>>> plt.plot(xMock,amed*xMock+bmed)
	>>> plt.fill_between(xMock,yBest-yStd,yBest+yStd,alpha=0.3, facecolor='gray')		

:param x: array of x data values
:param a: array of slopes from bootstrapped fits
:param b: array of intercepts from bootstrapped fits
:param n: number of bootstrapping resamples that will be generated
:returns: arrays x (mock x-values with same interval as data),yStd (confidence band)
	"""
	xsim=np.linspace(x.min(),x.max(),50)

	for i in range(n):
	    if i==0: 
	        yarr=a[i]*xsim+b[i]
	    else:
	        yarr2=a[i]*xsim+b[i]
	        yarr=np.vstack((yarr,yarr2))

	# best-fit values
	amed=np.median(a)
	bmed=np.median(b)
	yBest=amed*xsim+bmed
	# confidence band
	yStd=np.std(yarr,axis=0)

	# confidence band
	return xsim,yBest,yStd















# Monte Carlo simulations, generate random numbers
# ===================================================	
	

def gauss2d(par,varcov,n=10000):
	"""
Calculates random numbers drawn from the multinormal distribution in
two dimensions. Computes also the probability associated with each
number that can be used to computed confidence levels.

Arguments:

- par : array or list with structure [par0, par1, par2, ...] with the best-fit
  parameters
- varcov : variance-covariance matrix of the parameters
- n : number of (a,b) points generated from the multivariate gaussian

Output:

- x,y : arrays of random values generated from the multinormal distribution
- prob : probability associated with each value

Usage:
>>> x,y,prob=gauss2d(par,varcov,100000)

v1 Apr. 2012
	"""
	# Generates many realizations of a and b from the multinormal distribution
	x,y = np.random.multivariate_normal(par,varcov,n).T
	
	# Best-fit values
	x0,y0 = par[0], par[1]
	
	dx,dy = x-x0,y-y0
	sigx,sigy=np.sqrt(varcov[0,0]),np.sqrt(varcov[1,1])
	cov=varcov[0,1]
	rho=cov/sigx*sigy
	
	numchisq=(dx/sigx)**2+(dy/sigy)**2-2.*rho*(dx/sigx)*(dy/sigy)
	chisq=numchisq/(1.-rho**2)
	prob=np.exp(-chisq/2.)
	
	return x,y,prob




def bootcorr(x,y,nboot):
	"""
Given (X,Y) data points with intrinsic scatter, computes the 
bootstrapped Pearson and Spearman correlation coefficients. This
will give you an idea of how much outliers affect your correlation.

Usage:

>>> r,rho=bootcorr(x,y,100000)

performs 100000 bootstrapping realizations on the arrays x and y.

:returns: *r* - array with bootstrapped Pearson statistics
:returns: *rho* - bootstrapped array with Spearman statistics

	"""
	r,rho=[],[]
	for i in range(nboot):
		[xsim,ysim]=bootstrap([x,y])
		
		# Pearson r
		rsim=scipy.stats.pearsonr(xsim,ysim)	
		r.append(rsim[0])
		
		# Spearman rho
		rhosim=scipy.stats.spearmanr(xsim,ysim)
		rho.append(rhosim[0])

	r,rho=np.array(r),np.array(rho)

	results=np.array([ np.median(r), r.std(), np.median(rho), rho.std() ])
	print("<r>    err_r <rho> errrho")
	print(np.round(results, 2))

	results=np.array([ r2p(np.median(r)-np.abs(r.std()),x.size), r2p(np.median(r),x.size), r2p(np.median(r)+np.abs(r.std()),x.size) ])
	print("Prob. <- <r>-std,  <r>,    <r>+std")
	print(results)

	print("Rejection of H0 respectively at")
	for p in results:
		print(round(p2sig(p),2)	)
		
	return r,rho


def gen_ts(y,erry,n,zeropad=True):
    """
Given a time series (TS) with uncertainties on the signal, this will generate 
*n* new TS with y-values distributed according to the error bars. 

Output will be a :math:`n \\times {\rm{size}(t)}` array. Each row of this
array contains a simulated TS.

:param y: array of y-values for time series (do not need to be in order)
:param erry: array of 1-sigma errors on y-values
:param n: number of Mock TS to generate
:param zeropad: are y-values<0 not allowed? `True` will make any values<0 into 0
:returns: `n x size(t)` array. Each row of this array contains a simulated TS
    """
    ysim=np.empty((n,y.size))
    
    for i in range(y.size):
        # generate new points given normal distribution
        ysim[:,i]=np.random.normal(y[i],erry[i],n)   
        
    # makes sure no value is smaller than zero
    ysim[ysim<0]=0.
    
    return ysim



def random(a,b,n=None):
	"""
Generates an array of random uniformly distributed floats in the 
interval *[x0,x1)*.

>>> random(0.3,0.4,1000)
	"""
	return (b - a) * scipy.random.random_sample(n) + a




def linboot(x,y,n):
	"""
Performs the usual linear regression *with bootstrapping*. 

Usage:

>>> a,b=lincorr(x,y,10000)

performs 10000 bootstrapped fits on the arrays x and y. 

Plot the best-fit with

>>> amed=np.median(a)
>>> bmed=np.median(b)
>>> plt.plot(x,y,'o')
>>> plt.plot(x,amed*x+bmed)

:param x: array of x data values
:param y: array of y data values
:param n: number of bootstrapping resamples that will be generated
:returns: arrays `a` and `b`, each element is a different bootstrapped fit
	"""
	from . import lsd

	# y = A*x + B
	a, b=[],[]

	for i in range(n):
	    [xsim,ysim]=lsd.bootstrap([x,y])
	    
	    # Linear fit
	    asim, bsim, rsim, p, err = scipy.stats.linregress(xsim,ysim)
	    a.append(asim)
	    b.append(bsim)

	a,b=np.array(a),np.array(b)

	return a,b














	
	
	
# Comparing goodness-of-fit of different models
# ================================================
# F-test, Akaike information criterion and reduced chi-squared	

def ftest(rss1,rss2,n,p1,p2):
	"""
Carries out the F-test to decide which model fits the data better.
Computes the F statistic, associated p-value and the significance
in sigmas with which we can reject the null hypothesis. You can also
give the :math:`\chi^2` values instead of RSS if you have y-errors.

Note that p2>p1 must be obeyed, i.e. model 2 is "nested" within model 1.

Usage:

>>> fstat,pvalue,conf = ftest(rss1,rss2,n,p1,p2)

Arguments:
- rss1, rss2 : residual sum of squares for models 1 and 2
- n : sample size, i.e. number of data points
- p1, p2 : number of free parameters in the models

Returns:
- fstat : the F statistic
- pvalue : p-value associated with the F statistic
- conf : significance in sigmas with which we can reject the null hypothesis

References:
1. http://en.wikipedia.org/wiki/F-test
2. http://graphpad.com/curvefit/2_models__1_dataset.htm, Graphpad.com, Comparing the fits of two models (CurveFit.com)

v1 Dec. 2011
v2 Jan 16 2012: added comment regarding y-errors
"""
	fstat=scipy.stats.f_value(rss1,rss2,n-p1,n-p2)
	pvalue=1.-scipy.stats.f.cdf(fstat,p2-p1,n-p2)
	conf=np.sqrt(2.)*scipy.special.erfinv(1.-pvalue)
	
	return fstat, pvalue, conf
	
	



def aic(k,n,rss,errors=False):
	"""
Computes the Akaike information criterion which is "a measure of the relative goodness of
fit of a statistical model". Unlike the F-test, it does not assume that one model is 
a particular case of the more complex one. If errors==False then assume the
errors are the same for every data point. Otherwise, assumes you are providing
the chi-squared values instead of RSS.

Usage:

>>> aic = aic(k,n,rss)

:param rss: residual sum of squares for the model in case errors=False, otherwise assumes you are giving the chi-square values
:param n: sample size, i.e. number of data points
:param k: number of free parameters in the model
:returns: AIC statistic

References:

1. Documentation for Origin software on fit comparison: http://www.originlab.com/index.aspx?go=Products/Origin/DataAnalysis/CurveFitting/NonlinearFitting&pid=1195 (first heard about this test there)
2. http://en.wikipedia.org/wiki/Akaike_information_criterion

v1 Dec 2011
	"""
	if errors==False:
		# AIC assuming the errors are identical and given the residual sum of squares,
		# see http://en.wikipedia.org/wiki/Akaike_information_criterion#Relevance_to_chi-squared_fitting
		aicstat=n*np.log(rss/n)+2.*k
	else:
		# If you have different errors for the data points, it will assume rss=chisq
		aicstat=rss+2.*k
			
	# AICc = AIC with a correction for finite sample size
	aicc=aicstat+2.*k*(k+1.)/(n-k-1.)
	
	return aicc
	
	
	

def bic(k,n,rss,errors=False):
	"""
Computes the Bayesian information criterion which is "a criterion for model selection
among a finite set of models". From wikipedia: "The BIC [...] introduces a 
penalty term for the number of parameters in the model. The penalty term is larger 
in BIC than in AIC."

In order to use BIC to quantify the evidence against a specific model, check out
the section "use of BIC" in the presentation "171:290 Model Selection, Lecture VI: 
The Bayesian Information Criterion" by Joseph Cavanaugh 
(http://myweb.uiowa.edu/cavaaugh/ms_lec_6_ho.pdf).

If errors==False then assume the errors are the same for every data point. 
Otherwise, assumes you are providing the chi-squared values instead of RSS.

Usage:
>>> bic = bic(k,n,rss)

Arguments:
- rss : residual sum of squares for the model in case errors=False, otherwise assumes you are giving the chi-square values
- n : sample size, i.e. number of data points
- k : number of free parameters in the model

Returns: BIC statistic

References:
1. http://en.wikipedia.org/wiki/Bayesian_information_criterion
2. Model Selection, Lecture VI: The Bayesian Information Criterion" by Joseph Cavanaugh (http://myweb.uiowa.edu/cavaaugh/ms_lec_6_ho.pdf)

v1 Apr 2012
	"""
	if errors==False:
		# BIC assuming the errors are identical and given the residual sum of squares,
		# using the unbiased variance
		bicstat=n*np.log(rss/(n-1.))+k*np.log(n)
	else:
		# If you have different errors for the data points, it will assume rss=chisq
		bicstat=rss+k*np.log(n)
	
	return bicstat
	
	



def redchisq(x,y,a=None,b=None,sd=None):
	"""
Returns the reduced chi-square error statistic for a linear fit:
  chisq/nu
where nu is the number of degrees of freedom. If individual standard deviations 
(array sd) are supplied, then the chi-square error statistic is computed as the 
sum of squared errors divided by the standard deviations. 
See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.
	
If a linear model is not provided via A,B (y=ax+b) then the method computes
the chi-square using the best-fit line.

x,y,sd assumed to be Numpy arrays. a,b scalars. deg integer.
Returns the float chisq/nu with the reduced chi-square statistic.
	"""

	if a is None:	
		# Performs linear regression
		a, b, r, p, err = scipy.stats.linregress(x,y)
	
	# Chi-square statistic
	if sd is None:
		chisq=np.sum((y-a*x-b)**2)
	else:
		chisq=np.sum( ((y-a*x-b)/sd)**2 )
		
	# Number of degrees of freedom assuming 2 free parameters
	nu=x.size-3
	
	return chisq/nu
	



def redchisqxy(x,y,errx,erry,a,b):
	"""
Returns the reduced chi-square error statistic for a linear fit :math:`\chi^2/\nu`
where nu is the number of degrees of freedom. The chi-square statistic is
computed taking into account the errors in both X and Y (i.e. the
effective variance). See equation 3 in Ascenso et al. 2012, A&A or
Yee & Ellingson 2003 ApJ.

Usage:

>>> chisq=redchisqxy(xdata,ydata,errx,erry,a,b)

where

- xdata,ydata : data
- errx,erry : measurement uncertainties in the data
- a,b : slope and intercept of the best-fit linear regression model
	"""
	sdsq=erry**2+a**2*errx**2
	chisq=np.sum( (y-a*x-b)**2/sdsq )
		
	# Number of degrees of freedom assuming 2 free parameters
	nu=x.size-3
	
	return chisq/nu





def redchisqg(ydata,ymod,deg=2,sd=None):
	"""
Returns the reduced chi-square error statistic for an arbitrary model, 
chisq/nu, where nu is the number of degrees of freedom. If individual 
standard deviations (array sd) are supplied, then the chi-square error 
statistic is computed as the sum of squared errors divided by the standard 
deviations. See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

ydata,ymod,sd assumed to be Numpy arrays. deg integer.

Usage:

>>> chisq=redchisqg(ydata,ymod,n,sd)

where

- ydata : data
- ymod : model evaluated at the same x points as ydata
- n : number of free parameters in the model
- sd : uncertainties in ydata
  	"""
	# Chi-square statistic
	if sd is None:
		chisq=np.sum((ydata-ymod)**2)
	else:
		chisq=np.sum( ((ydata-ymod)/sd)**2 )
		
	# Number of degrees of freedom assuming 2 free parameters
	nu=ydata.size-1-deg
	
	return chisq/nu	


def r2(ydata,ymod):
	"""
Computes the "coefficient of determination" R^2. According to wikipedia,
"It is the proportion of variability in a data set that is accounted for 
by the statistical model. It provides a measure of how well future outcomes 
are likely to be predicted by the model."

This method was inspired by http://stackoverflow.com/questions/3460357/calculating-the-coefficient-of-determination-in-python

References:
1. http://en.wikipedia.org/wiki/Coefficient_of_determination
2. http://stackoverflow.com/questions/3460357/calculating-the-coefficient-of-determination-in-python

v1 Apr 18th 2012
	"""
	ss_tot = np.sum( (ydata-ydata.mean())**2 )
	ss_err = np.sum( (ydata-ymod)**2 )
	
	return 1.-ss_err/ss_tot



def fitstats(xdata,ydata,errx,erry,a,b):
	"""
Computes different useful statistics for the given (X +- errX, Y +- errY)
arrays of data. Also quantifies the goodness-of-fit of the provided
linear fit with slope 'a' and intercept 'b'.

Usage:

>>> r,rho,rchisq,scat,iscat,r2=fitstats(xdata,ydata,errx,erry,a,b)

  :param xdata,ydata: data
  :param errx,erry: measurement uncertainties in the data
  :param a,b: slope and intercept of the best-fit linear regression model
  
Returns:

	- Pearson 'r'
	- Spearman 'rho'
	- Reduced chi-squared 'chi^2_nu'
	- raw scatter about the best-fit
	- intrinsic scatter
	- Coefficient of determination 'R^2'
	"""
	# Pearson r
	r=scipy.stats.pearsonr(xdata,ydata)	
	# Spearman rho
	rho=scipy.stats.spearmanr(xdata,ydata)	
	# Reduced chi-squared
	rchisq=redchisqxy(xdata,ydata,errx,erry,a,b)
	# raw scatter
	scat=scatpratt(xdata,ydata,errx,erry,a,b)
	# intrinsic scatter	
	iscat=intscat(xdata,ydata,errx,erry,a,b)	
	# Coefficient of determination R^2
	rtwo=r2(ydata,a*xdata+b)
	
	return r,rho,rchisq,scat,iscat,rtwo


















# Operations on statistical distributions
# =======================================
#

def splitstd(x):
	"""
Given the input distribution, this method computes two standard deviations:
the left and right side spreads. This is especially useful if you are dealing
with a split normal distribution from your data and you want to quantify the
lower and upper error bars without having to fit a split normal distribution.

This is a quick nonparametric method, as opposed to the more computationally
intensive minimization involved when fitting a function.

Algorithm:

- divide the original distribution in two distributions at the mode
- create two new distributions, which are the symmetrical versions of the left and right side of the original one
- compute the standard deviations for the new mirror distributions

:param x: array or list containing the distribution
:returns: left.stddev, right.stddev

.. note:: do not forget to inspect *x*.
	"""
	#med=median(x)
	med=mode(x)

	x1=x[ np.where(x<med) ] # left side
	x1mirror=x1+2.*np.abs(x1-med) # mirror side
	x1new=np.concatenate((x1,x1mirror))

	x2=x[ np.where(x>=med) ] # right side
	x2mirror=x2-2.*np.abs(x2-med) # mirror side
	x2new=np.concatenate((x2,x2mirror))

	return x1new.std(), x2new.std()




def mode(x,**kwargs):
	"""
Finds the mode of a distribution, i.e. the value where the PDF peaks.

:param x: input list/array with the distribution
	"""
	from . import lsd

	yh,xh=np.histogram(x,50,normed=True,**kwargs)
	dxh=(xh[1]-xh[0])/2.
	xh=xh+dxh

	return xh[ lsd.search(yh.max(),yh) ]















# Custom statistical distributions
# ==================================
#

def randomvariate(pdf,n=1000,xmin=0,xmax=1):
	"""
Rejection method for random number generation:
Uses the rejection method for generating random numbers derived from an arbitrary 
probability distribution. For reference, see Bevington's book, page 84. Based on
rejection*.py.

Usage:

>>> randomvariate(P,N,xmin,xmax)

  where

:param P: probability distribution function from which you want to generate random numbers
:param N: desired number of random values
:param xmin,xmax: range of random numbers desired
  
:returns: the sequence (ran,ntrials) where
  	ran : array of shape N with the random variates that follow the input P
  	ntrials : number of trials the code needed to achieve N

Here is the algorithm:

- generate x' in the desired range
- generate y' between Pmin and Pmax (Pmax is the maximal value of your pdf)
- if y'<P(x') accept x', otherwise reject
- repeat until desired number is achieved

v1 Nov. 2011
	"""
	# Calculates the minimal and maximum values of the PDF in the desired
	# interval. The rejection method needs these values in order to work
	# properly.
	x=np.linspace(xmin,xmax,1000)
	y=pdf(x)
	pmin=0.
	pmax=y.max()

	# Counters
	naccept=0
	ntrial=0

	# Keeps generating numbers until we achieve the desired n
	ran=[]	# output list of random numbers
	while naccept<n:
		x=np.random.uniform(xmin,xmax)	# x'
		y=np.random.uniform(pmin,pmax)	# y'

		if y<pdf(x):
			ran.append(x)
			naccept=naccept+1
		ntrial=ntrial+1
	
	ran=np.asarray(ran)
	
	return ran,ntrial





def splitnorm(x,sig1=1.,sig2=1.,mu=0.):
	"""
Split normal distribution (or asymmetric gaussian) PDF. Useful when dealing
with asymmetric error bars.

See http://en.wikipedia.org/wiki/Split_normal_distribution.

:param x: input array where the PDF will be computed
:param sig1: left standard deviation
:param sig2: right std. dev.
:param mu: mode
:returns: probability distribution function for the array x
	"""
	const=np.sqrt(2./np.pi)/(sig1 + sig2)
    
	p=np.where(x<mu, 
		np.exp(-((x-mu)**2/(2.*sig1**2)))*const, 
		np.exp(-((x-mu)**2/(2.*sig2**2)))*const)
    
	return p



class splitnorm_gen(scipy.stats.rv_continuous):
    """
Split normal distribution defined using scipy.stats. Can be called as 
any scipy.stats distribution. I took the effort of defining the extra
CDF and PPF methods below in order to speedup calls to this class 
(I have particularly in mind the package *mcerp*).

:param x: input array where the PDF will be computed
:param sig1: left standard deviation
:param sig2: right std. dev.
:param mu: mode

Examples: 

Defines distribution with sig1=1, sig2=3, mu=0:

>>> split = splitnorm_gen(name='splitnorm', shapes='sig1, sig2, mu')
>>> s=split(1,3,0.0001)
>>> x=np.linspace(-10,10,100)

Computes PDF:

>>> s.pdf(x)

Computes CDF:

>>> s.cdf(x)

Generates 100 random numbers:

>>> s.rvs(100)

.. warning:: for some reason, this fails if mu=0.
    """
    def _pdf(self, x, sig1, sig2, mu):
        const=np.sqrt(2./np.pi)/(sig1 + sig2)
            
        p=np.where(x<=mu, 
                np.exp(-((x-mu)**2/(2.*sig1**2)))*const, 
                np.exp(-((x-mu)**2/(2.*sig2**2)))*const)
                      
        return p
    
    def _cdf(self, x, sig1, sig2, mu):
        const=1./(sig1+sig2)
        
        c=np.where(x<=mu, 
                sig1*const*(1.+scipy.special.erf((x-mu)/(np.sqrt(2.)*sig1))), 
                const*(sig1+sig2*scipy.special.erf((x-mu)/(np.sqrt(2.)*sig2)))
                )
        
        return c
    
    def _ppf(self,q,sig1,sig2,mu):
        nu=sig1/(sig1 + sig2)
        
        pf=np.where(q<=nu, 
                np.sqrt(2.)*sig1*scipy.special.erfinv((sig1 + sig2)/sig1*q - 1.) + mu, 
                np.sqrt(2)*sig2*scipy.special.erfinv(((sig1 + sig2)*q - sig1)/sig2) + mu
                )
        
        return pf     
    
    def _stats(self, sig1, sig2, mu):
        mean=mu+np.sqrt(2./np.pi)*(sig2 - sig1)
        var=(1.-2./np.pi)*(sig2-sig1)**2 + sig1*sig2
        skew=np.sqrt(2./np.pi)*(sig2-sig1)*( (4./np.pi-1.)*(sig2-sig1)**2+sig1*sig2 )
        kurt=None
        
        return mean, var, skew, kurt






















# p-values and significance
# ===========================
#


def r2p(r,n):
	"""
Given a value of the Pearson r coefficient, this method gives the
corresponding p-value of the null hypothesis (i.e. no correlation).

Code copied from scipy.stats.pearsonr source.

Usage:

>>> p=r2p(r,n)

where 'r' is the Pearson coefficient and n is the number of data
points.

:returns: p-value of the null hypothesis.
	"""
	import scipy.special
	df = n-2
	if abs(r) == 1.0:
		prob = 0.0
	else:
		t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
		prob = scipy.special.betainc(0.5*df, 0.5, df / (df + t_squared))
        
	return prob



def p2sig(p):
	"""
Given the p-value Pnull (i.e. probability of the null hypothesis) below, 
evaluates the confidence level with which we can reject the hypothesis in standard 
deviations.

Inspired on prob2sig.nb.

Usage:

>>> s=p2sig(0.001)
	"""
	sig=np.sqrt(2.)*scipy.special.erfinv(1.-p)
	
	return sig



def conf2sig(p):
	"""
Given a confidence p-value, translates it into standard deviations.
E.g., p=0.683 -> 1sigma, p=0.954 -> 2sigma, etc.

Inspired on p2sig method.

Usage:

>>> s=conf2sig(0.683)
	"""
	sig=np.sqrt(2.)*scipy.special.erfinv(p)
	
	return sig



def sig2conf(sig):
	"""
Given a number of std. deviations, translates it into a p-value.
E.g., 1sigma -> p=0.683, 2sigma -> p=0.954, etc.

Inspired on p2sig method.

Usage:

>>> s=sig2conf(5.)
	"""
	return scipy.special.erf(sig/np.sqrt(2.))
	

	