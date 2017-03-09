"""
Astrophysical routines
=========================

"""


import numpy
import scipy




def dist2z(d):
	"""
Converts luminosity distance to redshift by solving the equation
'd-z=0'.

Input is assumed float.
	"""
	import cosmolopy 

	# x here is the unknown redshift
	f = lambda x: d-cosmolopy.distance.luminosity_distance(x,**cosmolopy.fidcosmo)
 
	z = scipy.optimize.fsolve(f, 0.01)  
	return z

 


def mjy(lognu,ll,dist,llerr=None):
    """
Converts log(nu/Hz), log(nu Lnu [erg/s]), error in log(nuLnu) to
log(lambda/micron), log(Fnu/mJy), error in log(Fnu).
The input units are CGS.

Usage:
If you have errors in the flux:

>>> lamb,fnu,ferr=mjy(xdata,ydata,dist,yerr)

If you do not have errors in the flux:

>>> lamb,fnu=mjy(xdata,ydata,dist)

:param dist: distance in Mpc
    """
    import uncertainties.unumpy as unumpy

    c=29979245800.	# speed of light in CGS
    dist=dist*3.085677581e24	# Mpc -> cm

    nu=10**lognu
    lamb=c/nu*1e4 # cm -> micron
    if llerr!=None:
    	lllerr=unumpy.uarray(ll,llerr)
    else:
    	lllerr=ll
    lnuerr=10**lllerr/nu  
    fluxerr=lnuerr/(1e-26*4.*numpy.pi*dist**2) # Lnu (erg/s/Hz) -> Fnu (mJy)
    if llerr!=None:
    	fluxerr=unumpy.log10(fluxerr)
    	return numpy.log10(lamb),unumpy.nominal_values(fluxerr),unumpy.std_devs(fluxerr)
    else:
    	return numpy.log10(lamb),numpy.log10(fluxerr)






def arcsec2pc(d=15.,a=1.):
	"""
Given the input angular size and distance to the object, computes
the corresponding linear size in pc. 

:param d: distance in Mpc
:param a: angular size in arcsec
:returns: linear size in pc
	"""

	# convert arcsec to radians
	a=a*4.848e-6	
	# convert distance to pc instead of Mpc
	d=d*1e6

	return d*numpy.tan(a)



def freq(T):
    """
Convert array of periods in days to frequencies in Hz.
    """
    return 1./T/86400.



def period(freq):
    """
Convert array of frequencies to periods.
    """
    return 1./freq




class Constants:
    """
Defines a set of useful constants in CGS.
::

    const=nmmn.astro.Constants()
    E=mass*const.c**2
    """

    def __init__(self):
        self.G = 6.673E-8
        self.c = 29979245800
        self.solarmass = 1.99e33
        self.year = 31556926
        
