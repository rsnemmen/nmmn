"""
Miscelaneous methods
======================
"""

import numpy



def pol2cart(r, phi):
	"""
Converts from polar to cartesian coordinates.

>>> x,y=pol2cart(r,phi)
	"""
	x = r * numpy.cos(phi)
	y = r * numpy.sin(phi)
	return x, y
    

def cart2pol(x, y):
	"""
Converts from cartesian to polar coordinates.

>>> r,t=cart2pol(x,y)
	"""
	r = numpy.sqrt(x**2 + y**2)
	t = numpy.arctan2(y, x)
	return r, t



    
	

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




