"""
LSD operations = lists, sets, dictionaries (and arrays)
=========================================================
"""

import numpy
import scipy


def cmset_and(x,y):
	"""
Usage:

>>> cmset_and(x,y)

returns the index of the elements of array x which are also present in the
array y. 

This is equivalent to using the IDL command

>>> botha=cmset_op(namea, 'AND', nameb, /index)

i.e. performs the same thing as the IDL routine `cmset_op <http://cow.physics.wisc.edu/~craigm/idl/idl.html>`_.
	"""
	
	idel=[]	# list of indexes of x elements which are also in y
	i=0
	for xx in x:
		if xx in y: idel.append(i)
		i=i+1
		
	return idel




def cmsetsort_and(x,y):
	"""
Usage:

>>> cmsetsort_and(x,y)

returning the index of the elements of array x which are also present in the
array y. 

The resulting elements have the same order as the ones in y. For 
instance, if you run

>>> i=cmsetsort_and(x,y)
>>> x[i]==y

will return an array of True, whereas if you used instead cmset_and it is
not guaranteed that all elements would match in x[i] and y.

Inherited from :func:`nemmen.cmset_and`.
	"""
	
	idel=[]	# list of indexes of x elements which are also in y
	i=0
	for yy in y:
		i=numpy.where(x==yy)
		idel.append( i[0].item() )
		
	return idel
	



def cmset_not(x,y):
	"""
Usage:

>>> cmset_not(x,y)

returning the index of the elements of array x which are not present in the
array y. 

This is equivalent to using the IDL command
SET = CMSET_OP(A, 'AND', /NOT2, B, /INDEX)   ; A but not B
i.e. performs the same thing as the IDL routine cmset_op from
http://cow.physics.wisc.edu/~craigm/idl/idl.html.
	"""
	
	idel=[]	# list of indexes of x elements which NOT in y
	i=0
	for xx in x:
		if xx not in y: idel.append(i)
		i=i+1
		
	return idel


	
def nanzero(x):
	"""
Set nan elements to zero in the array.
	"""
	# Index of nan elements
	i=numpy.where(numpy.isnan(x)==True)

	y=x.copy()
	y[i]=0.
	#y[i]=1e-20
	
	# Removes the nan elements
	return y




def delnan(x):
	"""
Remove nan elements from the array.
	"""
	# Index of nan elements
	i=numpy.where(numpy.isnan(x)==True)
	
	# Removes the nan elements
	return numpy.delete(x,i)

	


def delweird(x):
	"""
Remove nan or inf elements from the array.
	"""
	# Index of nan elements
	i=numpy.where( (numpy.isnan(x)==True) | (numpy.isinf(x)==True) )
	
	# Removes the nan elements
	return numpy.delete(x,i)



def findnan(x):
	"""
Return index of nan elements in the array.
	"""
	# Index of nan elements
	i=numpy.where(numpy.isnan(x)==True)
	
	return i




	
def replacevals(x,minval):
	"""
Replace all values in array x for which abs(x)<=minval with x=sign(x)*minval.
	"""
	i=numpy.where(numpy.abs(x)<=minval)
	y=x.copy()
	y[i]=numpy.sign(y[i])*minval

	return y



	
def search(xref, x):
	"""
Search for the element in an array x with the value nearest xref.
Piece of code based on http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

>>> i=search(xref, x)

:param xref: input number, array or list of reference values
:param x: input array
:returns: index of the x-elements with values nearest to xref:
	"""
	if numpy.size(xref)==1:
		i=(numpy.abs(x-xref)).argmin()
	else:
		i=[]

		for y in xref:
			i.append( (numpy.abs(x-y)).argmin() )
	        
	return i
	


def sortindex(x,**kwargs):
	"""
Returns the list of indexes, ordered according to the numerical value of each 
element of x.

:param x: input array or list.
:returns: list of element indexes.
	"""
	f=lambda i: x[i]

	return sorted( range(numpy.size(x)) , key=f,**kwargs)
	



def norm(x1,x2=None):
	"""
Normalizes x1. If also given as input x2, then normalizes x1 to x2.

:param x1: input array
:param x2: optional
:returns: normalized x1
	"""
	if x2 is None:
		return x1/x1.max()
	else:
		return x1*x2.max()/x1.max()





def uarray(x,errx):
	"""
With the new releases of the uncertainties and astropy.io.ascii (0.2.3, the
replacement of asciitable), if I try to create an uncertainties array with
the column of a table imported with ascii I run into trouble. For instance, 
if I use the sequence of commands below:

>>> import astropy.io.ascii as asciitable
>>> raff= asciitable.read('data/rafferty06.dat')
>>> m,errm=raff['mass'],raff['errm']
>>> mass=unumpy.uarray(m,errm)
>>> x=0.2*mass

I get the error message: 

>>> TypeError: unsupported operand type(s) for *: 'float' and 'Column'

which I can only assume is due to the new way ascii handles tables.

I created this method to use as a replacement for unumpy.uarray that handles
the tables created with astropy.io.ascii.

Usage is the same as uncertainties.unumpy.uarray.

:type x,errx: arrays created with astropy.io.ascii.
:returns: uncertainties array.
	"""
	import uncertainties.unumpy as unumpy

	x=numpy.array(x)
	errx=numpy.array(errx)

	return unumpy.uarray(x,errx)






	
def bootstrap(v):
	"""
Constructs Monte Carlo simulated data set using the
Bootstrap algorithm.                                                                                   

Usage:

>>> bootstrap(x)

where x is either an array or a list of arrays. If it is a
list, the code returns the corresponding list of bootstrapped 
arrays assuming that the same position in these arrays map the 
same "physical" object.

Rodrigo Nemmen, http://goo.gl/8S1Oo
	"""
	if type(v)==list:
		vboot=[]	# list of boostrapped arrays
		n=v[0].size
		iran=scipy.random.randint(0,n,n)	# Array of random indexes
		for x in v:	vboot.append(x[iran])
	else:	# if v is an array, not a list of arrays
		n=v.size
		iran=scipy.random.randint(0,n,n)	# Array of random indexes
		vboot=v[iran]
	
	return vboot
	



def regrid(x,y,z,xnew,ynew,method='cubic'):
	"""
Regrid 1D arrays (x,y,z) -- where z is some scalar field mapped at positions
x,y -- to a 2d array Z defined in the cartesian grids xnew,ynew (1D arrays with 
new grid).

For the interpolation method, choose nearest, linear or cubic.

>>> rho=regrid(d.x,d.y,d.rho,xnew,ynew)

.. todo:: need to create a 3d version of this method, paving the road for the 3d simulations.
	"""
	import scipy.interpolate

	# regrid the data to a nice cartesian grid
	Z = scipy.interpolate.griddata((x, y), z, (xnew[None,:], ynew[:,None]), method=method)

	# get rid of NaNs
	return nanzero(Z)




def crop(z, x,y, xmin, xmax, ymin, ymax, all=False):
	"""
Crops the image or 2D array, leaving only pixels inside the region
you define.

>>> Znew,Xnew,Ynew = crop(Z, X, Y, 0,10,-20,20)

where X,Y are 1D or 2D arrays, and Z is a 2D array.

:param z: 2d array 
:param x,y: 1d or 2d arrays. In the latter case, they should have the same shape as z
:param all: should I return cropped Z,X,Y or only Z?
:returns: Z_cropped, X_cropped, Y_cropped
	"""
	if x.ndim==1: # if x,y are 1D
		# Index tuples with elements that will be selected along each dimension
		i=numpy.where((x>=xmin) & (x<=xmax))	# x
		j=numpy.where((y>=ymin) & (y<=ymax))	# y

		# Defines new x and y arrays
		xnew,ynew=x[i],y[j]
		
		i,j=i[0],j[0]	# tuples -> arrays (for matrix slicing below)
		znew=z[j[0]:j[-1],i[0]:i[-1]]	# CAREFUL with the ordering of the indexes!
	elif x.ndim==2: # if x,y are 2D
		i=numpy.where((x[0,:]>=xmin) & (x[0,:]<=xmax))
		j=numpy.where((y[:,0]>=ymin) & (y[:,0]<=ymax))
		i,j=i[0],j[0]

		xnew=x[j[0]:j[-1],i[0]:i[-1]]
		ynew=y[j[0]:j[-1],i[0]:i[-1]]
		znew=z[j[0]:j[-1],i[0]:i[-1]]
	else: 
		print("Dimensions of the input arrays are inconsistent")
		return
	
	if all==False:
		return znew	
	else:
		return znew,xnew,ynew





def arrAvg(alist):
	"""
	Given a list of 1D or 2D arrays, this method computes their average, 
	returning an array with the same shape as the input.

	:param alist: list of arrays 
	:returns: average, std. dev. -- arrays with the same shape as the input arrays

	Usage:

	>>> avg=arrAvg([x,y,z])
	"""
	if alist[0].ndim==2:
		# join arrays together, creating 3D arrays where the third dimension is e.g. time
		# or whatever index for the different arrays you want to average
		arr=numpy.stack(alist,axis=2)

		# performs the average 
		return numpy.mean(arr,axis=2), numpy.std(arr,axis=2)
	elif alist[0].ndim==1:
		arr=numpy.stack(alist,axis=1)
		return numpy.mean(arr,axis=1), numpy.std(arr,axis=1)
	else:
		print("Dimensionality not supported")




def string2float(s):
	"""
Converts from an array of strings to floats.

>>> string2float('28122014')

returns 28122014.0.

:param s: a string or list/array of strings
:returns: a numpy array of floats
	"""
	if numpy.size(s)==1:	# if input is a single string
		out=numpy.float(s)
	else:	# if input is list/array
		out=[]

		for i, si in enumerate(s): 
			out.append(numpy.float(si))

		out=numpy.array(out)

	return out

