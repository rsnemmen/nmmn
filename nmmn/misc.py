"""
Miscelaneous methods
======================
"""

import numpy



# COORDINATE TRANSFORMATIONS
# ===========================
#
#

def pol2cart(r, th):
	"""
Converts from polar to cartesian coordinates.

>>> x,y=pol2cart(r,phi)
	"""
	x = r * numpy.cos(th)
	y = r * numpy.sin(th)
	return x, y


def sph2cart(r, th):
	"""
Converts from spherical polar to cartesian coordinates.

>>> x,y=pol2cart(r,phi)
	"""
	# spherical polar angle to polar
	th=-(th-numpy.pi/2.) 

	x = r * numpy.cos(th)
	y = r * numpy.sin(th)
	return x, y



def cart2pol(x, y):
	"""
Converts from cartesian to polar coordinates.

>>> r,t=cart2pol(x,y)
	"""
	r = numpy.sqrt(x**2 + y**2)
	t = numpy.arctan2(y, x)
	return r, t


def cart2sph(x, y):
	"""
Converts from cartesian to spherical polar coordinates,
poles are at theta=0, equator at theta=90deg

>>> r,t=cart2pol(x,y)
	"""
	r = numpy.sqrt(x**2 + y**2)
	t = numpy.pi/2.-numpy.arctan2(y, x)
	return r, t




def vel_p2c(th,vr,vth):
	"""
Computes the cartesian components of a velocity vector which
is expressed in polar coordinates. i.e. apply a change of 
basis. See for example discussion after eq. 4 in 
https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-07-dynamics-fall-2009/lecture-notes/MIT16_07F09_Lec05.pdf

Returns: vx, vy
	"""
	vx=vr*numpy.cos(th)-vth*numpy.sin(th)
	vy=vr*numpy.sin(th)+vth*numpy.cos(th)

	return vx, vy




def vel_c2p(th,vx,vy):
	"""
Computes the polar components of a velocity vector which
is expressed in cartesian coordinates. 

Returns: vr, vth
	"""
	vr=vx*numpy.cos(th)+vy*numpy.sin(th)
	vth=-vx*numpy.sin(th)+vy*numpy.cos(th)

	return vr, vth

    











    
	

def evalfun(fun,x,par):
	"""
Evaluates the function fun at each element of the array x, for the parameters
provided in the array/list par. fun is assumed to return a scalar. Returns an
array with fun evaluated at x. See example below.

Usage:

>>> p=array([1,2,3])

  x=1, par0=2, par1=3 
  
>>> fun(p)

  returns a scalar
  
>>> x=linspace(0,10,50)

>>> evalfun(fun,x,[2,3])

  evaluates fun at the array x and returns an array.

v1 Dec. 2011
	"""	
	y=numpy.zeros_like(x)
	
	for i in range(x.size):
		# Array consisting of [x[i], par1, par2, ...]
		p=numpy.concatenate(([x[i]],par))
		
		# Function evaluated
		y[i]=fun(p)
		
	return y







def savepartial(x,y,z,obsx,obsy,obsz,outfile):
	"""
Exports data file for partial correlation analysis with cens_tau.f.
cens_tau.f quantifies the correlation between X and Y eliminating the
effect of a third variable Z. I patched the Fortran code available from
http://astrostatistics.psu.edu/statcodes/sc_regression.html.

Method arguments:
x,y,z = arrays with data for partial correlation
obs? = arrays of integers. 1 if there is a genuine measurement
available and 0 if there is only an upper limit i.e. censored data.

In the case of this study, X=Pjet, Y=Lgamma, Z=distance.

The structure of the resulting datafile is:
	logPjet detected? logLgamma detected? logDist detected?
where the distance is in Mpc. 

Example:

>>> agngrb.exportpartial(all.kp,all.lg,log10(all.d),ones_like(all.kp),ones_like(all.kp),ones_like(all.kp),'par
tialdata.dat')

v1 Sep. 2011
	"""	
	numpy.savetxt(outfile,numpy.transpose((x,obsx,y,obsy,z,obsz)),fmt='%10.4f %i %10.4f %i %10.4f %i')






	

def log2lin(dlogyl,dlogyu,logy):
	"""
From a given uncertainty in log-space (dex) and the value of y, calculates the
error in linear space.

Returns a sequence with the lower and upper arrays with errors.
	"""
	# Upper error bar
	dyu=10**logy*(10.**dlogyu-1.)

	# Lower error bar
	dyl=-10**logy*(10.**-dlogyl-1.)

	return dyl, dyu





def lin2log(dyl,dyu,logy):
	"""
From a given uncertainty in linear space and the value of y, calculates the
error in log space.

Returns a sequence with the lower and upper arrays with errors.
	"""
	# Upper error bar
	dlogyu=-logy+numpy.log10(10.**logy+dyu)

	# Lower error bar
	dlogyl=+logy-numpy.log10(10.**logy-dyl)

	return dlogyl, dlogyu
	
	
	


def whichbces(bces):
	"""
Given the 'bces' string selector, returns an integer which tells the 
location of the BCES fitting results in the arrays returned by
the bces* methods.
	"""
	# Selects the appropriate BCES fitting method
	import sys
	
	if bces=='ort':
		i=3
	elif bces=='y|x':
		i=0
	elif bces=='x|y':
		i=1
	elif bces=='bis':
		i=2	
	else:
		sys.exit("Invalid BCES method selected! Please select bis, ort, y|x or x|y.")	

	return i
	




def mathcontour(x,y,errx,erry,cov):
	"""
Suppose I want to draw the error ellipses for two parameters 'x' and 'y' 
with 1 s.d. uncertainties and the covariance between these uncertainties. 
I have a mathematica notebook that does that: 'chisq contour plot.nb'.

This method takes the values 'x,y,errx,erry,cov' and outputs Mathematica
code that I can just copy and paste in the appropriate notebook in order 
to draw the ellipses.
	"""
	print("x0=",x,";")
	print("y0=",y,";")
	print("\[Sigma]x=",errx,";")
	print("\[Sigma]y=",erry,";")
	print("\[Sigma]xy=",cov,";")






# Methods related to datetime tuples
# ====================================
#
def convertyear(y):
	"""
Converts from decimal year to a python datetime tuple.

>>> convertyear(2012.9)

returns datetime.datetime(2012, 11, 24, 12, 0, 0, 3).

:param y: a float or array of floats
:returns: a datetime structure (or list) with the date corresponding to the input float or array

Reference: http://stackoverflow.com/questions/19305991/convert-fractional-years-to-a-real-date-in-python
	"""
	import datetime

	if numpy.size(y)==1:	# if input is float
		year = int(y)
		d = datetime.timedelta(days=(y - year)*365)
		day_one = datetime.datetime(year,1,1)
		date = d + day_one
	else:	# if input is list/array
		date=[]

		for i, yi in enumerate(y): 
			year = int(yi)
			d = datetime.timedelta(days=(yi - year)*365)
			day_one = datetime.datetime(year,1,1)
			date.append(d + day_one)

	return date


def string2year(s):
	"""
Converts from a string in the format DDMMYYYY to a python datetime tuple.

>>> string2year('28122014')

returns datetime.datetime(2014, 12, 22, 0, 0).

:param y: a string or list of strings
:returns: a datetime structure (or list) with the date corresponding to the input float or array

Reference: https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
	"""
	import datetime

	if numpy.size(s)==1:	# if input is a string
		date=datetime.datetime.strptime(str(s),'%d%m%Y')
	else:	# if input is list/array
		date=[]

		for i, si in enumerate(s): 
			date.append(datetime.datetime.strptime(str(si),'%d%m%Y'))

	return date



def date2dec(date):
	"""
Convert a python datetime tuple to decimal year.

Inspired on http://stackoverflow.com/a/6451892/793218.
	"""
	import datetime
	import time

	def sinceEpoch(date): # returns seconds since epoch
		return time.mktime(date.timetuple())

	def getyear(date): # returns decimal year for a datetime tuple
		year = date.year
		startOfThisYear = datetime.datetime(year=year, month=1, day=1)
		startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)
		yearElapsed = sinceEpoch(date) - sinceEpoch(startOfThisYear)
		yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
		fraction = yearElapsed/yearDuration
		return date.year + fraction
	
	if numpy.size(date)==1:
		year=getyear(date)
	else:
		year=[]
		for datei in date:
			year.append(getyear(datei))

	return numpy.array(year)











def runsave(cmd,log):
	"""
Executes command cmd and saves its standard output as log
	"""
	import subprocess
	
	# executes command
	p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out,err=p.communicate()

	# saves output in a diagnostic file
	text=open(log,"w")
	text.write(str(out))
	text.close()



def scinotation(x,n=2):
	"""
Displays a number in scientific notation.

:param x: number
:param n: number of significant digits to display
	"""
	import decimal
	fmt='%.'+str(n)+'E'
	s= fmt % decimal.Decimal(str(x))

	return s






def readmodel(file,m=1e8):
    """
Auxiliary method to rescale the BHBH simulation time series from Gold+14 and 
compare with the data.
::

    tphys12, t12, y12 = readmodel('Gold paper/1,2nc mdot bin.csv')

:returns: time in years, time in code units, signal
    """
    from . import astro, dsp
    import astropy.io.ascii as ascii
    import scipy.signal

    data = ascii.read(file)
    tmod,ymod=data['col1'],data['col2']

    # cleans TS (remove duplicate times, regular dt for CWT later)
    tmod,ymod=dsp.uneven2even(tmod,ymod)
    
    # detrends data
    ymoddet=scipy.signal.detrend(ymod)
    
    # physical times in years
    const=astro.Constants()
    tphys=tmod*1000*const.G*m*const.solarmass/const.c**3/const.year
    
    return tphys, tmod, ymoddet




def adjustmodel(file,stretch,translate,obs):
    """
Auxiliary method to read and normalize the BHBH simulation TS from Gold+14 
to the observations. 
::

    tmod,smod=adjustmodel('/Users/nemmen/work/projects/fermi/ngc1275/Gold paper/1,2nc mdot bin.csv',5.6,2008.6,y)

obs is the data you want to normalize the model to.

:returns: time in years, time in code units, signal
    """
    from . import lsd
    import astropy.io.ascii as ascii
    import scipy.signal

    data = ascii.read(file)
    tmod,ymod=data['col1'],data['col2']

    # detrends and normalize data
    ymoddet=scipy.signal.detrend(ymod)
    ymod=lsd.norm(ymoddet,scipy.signal.detrend(obs))
    
    # physical times (scaled to data)
    tphys=(tmod-tmod[0])*stretch+translate
    
    # BH mass implied by the scaling, in solar masses
    # Careful with units: CGS then converted to Msun
    mass=1e-3*4.04e38*(tphys[-1]-tphys[0])/(tmod[-1]-tmod[0])*31556926./1.99e33
    print('M = ',mass,' Msun')
    
    return tphys, ymod







def paperimpact(citations,dt,impactfactor):
	"""
Given a journal `impactfactor` and a given time interval `dt`, this
computes the expected number of citations for a given paper 
published in that journal over `dt`. It then gives the ratio of your
paper's citation over the predicted value, to give you an idea
whether your paper is above or below average.

:param citations: number of citations for your paper
:param dt: time since your paper was published in years
:param impactfactor: impact factor for the journal where the paper was published
	"""
	print("Journal expected number of citations =",dt*impactfactor)

	print("Actual citations/expected citations",citations/(dt*impactfactor))




def findPATH(filename,envVar="PYTHONPATH"):
	"""
Given a PATH or PYTHONPATH environment variable, find the full path of a file 
among different options. From https://stackoverflow.com/a/1124851/793218

:param filename: file for which full path is desired
:param envVar: environment variable with path to be searched
:returns: string with full path to file

Example:

>>> fullpath=findPATH("fastregrid.cl")
	"""
	import os

	for p in os.environ[envVar].split(":"):
	    for r,d,f in os.walk(p):
	        for files in f:
	             if files == filename:
	                 return os.path.join(r,files)



def mario():
	"""
Displays a nice Super Mario. :)

Analogous to \mario in LaTeX.
	"""
	from IPython.display import Image
	from IPython.core.display import HTML 
	Image(url= "https://banner2.kisspng.com/20180410/kye/kisspng-new-super-mario-bros-u-super-mario-64-8-bit-5acd5c8ba05651.6908995015234080116568.jpg")
