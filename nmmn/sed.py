"""
Spectral energy distributions
===============================

Methods and classes for dealing with spectral energy distributions (SEDs).
"""

import numpy, pylab, os, scipy
from . import lsd # intrapackage reference
import astropy.io.ascii as ascii



class SED:
	"""
	Class that defines objects storing Spectral energy distributions (SEDs). 
	The SEDs can be imported either from datafiles with the observations/models,
	or from arrays with the values of log(nu) and log(nuLnu).
	
	The object's initial attributes are:

	- nu: frequency in Hertz
	- lognu: log10(frequency)
	- nlnu: nu L_nu in erg/s
	- ll: log10(nu Lnu)
	- file : input datafile
	
	To import a SED from an ASCII file:

	>>> s=sed.SED(file=datafile, logfmt=XX)

	where logfmt is 1 if the data is already in logarithm or 0 otherwise.
	NOTE: logfmt for now is by default 0 (Eracleous' SEDs).
	
	To create a SED object from the arrays lognu and lognulnu:

	>>> s=sed.SED(lognu=lognu, ll=lognulnu)
	
	To import a SED in a special format:

	>>> s=sed.SED()
	>>> s.erac('ngc1097.dat')
	
	To plot the SED:

	>>> plot(s.lognu, s.ll)
	
	To calculate the bolometric luminosity:

	>>> s.bol()
	
	after which the value will be stored in s.lumbol	
	"""
		
	def __init__(self, file=None, logfmt=0, lognu=None, ll=None):
		"""
	To import a SED from an ASCII file:
	>>> s=sed.SED(file=XX, logfmt=XX)
	where logfmt is 1 if the data is already in logarithm or 0 otherwise.
	NOTE: logfmt for now is by default 0 (e.g. Eracleous' SEDs).
	
	To create a SED object from the arrays lognu and lognulnu:

	>>> s=sed.SED(lognu=lognu, ll=lognulnu)
	
	>>> s=sed.SED()

	creates an empty SED object. Then you can call specific methods to handle
	SEDs in special formats (e.g. s.erac).
		"""
		self.file=file	# input datafile (string) from which the observed spectrum was imported
		self.lognu=lognu
		self.ll=ll

		# If the SED is imported from a file...
		if file is not None:
			if logfmt==0:
				# Reads SED from datafile
				self.nu,self.nlnu = numpy.loadtxt(file,unpack=True,usecols=(0,1))
		
				# Take logs
				self.lognu, self.ll = numpy.log10(self.nu), numpy.log10(self.nlnu)
			else:
				self.lognu, self.ll = numpy.loadtxt(file,unpack=True,usecols=(0,1))
				self.nu,self.nlnu = 10.**self.lognu, 10.**self.ll

		# If the SED is created from the arrays
		if file is None and lognu is not None:
			if logfmt==1:
				self.lognu, self.ll = lognu, ll
				self.nu,self.nlnu = 10.**self.lognu, 10.**self.ll
			else:
				self.nu, self.nlnu = lognu, ll
				self.lognu,self.ll = numpy.log10(self.lognu), numpy.log10(self.ll)
		
		# Checks if ll has NaN or Inf values
		if file is not None or lognu is not None:
			self.check()



	def erac(self, file):
		"""
		Reads SEDs in the format provided by Eracleous et al. LINER SEDs:
		nu	nuLnu	nuLnu(extinction)	error	upperLimit?(0/1)
		
		Adds the following attributes to the SED object:
		- nlnuex: nuLnu after extinction correction
		- llex: log10(nuLnu) after extinction correction
		- ul: is the datapoint an upper limit (arrow)? 1 if yes, 0 if not
		"""
		self.file=file
		
		self.nu,self.nlnu,self.nlnuex,self.ul = numpy.loadtxt(file,unpack=True,usecols=(0,1,2,4))
		self.lognu, self.ll = numpy.log10(self.nu), numpy.log10(self.nlnu)
		self.llex = numpy.log10(self.nlnuex)
		
		# Checks if ll has NaN or Inf values
		self.check()




	def hayden(self, file, dist):
		"""
		Reads SEDs in the format provided by Hampadarath et al.:
	nu(GHz) flux(microJy) error(microJy) Detection[1]orUpperlimit[0]
		
		Adds the following attributes to the SED object:
		- ul: is the datapoint an upper limit (arrow)? 1 if yes, 0 if not
		- distance: distance to object in Mpc
		"""
		self.file=file
		self.distance=dist
		
		self.nu,flux,x,self.ul = numpy.loadtxt(file,unpack=True,usecols=(0,1,2,3))
		flux=flux*1e-26*1e-6*1e7*1e-4	# microJy to erg/s/cm^2/Hz
		dist=dist*3.086e24	# Mpc to cm
		self.lnu=4.*numpy.pi*dist**2*flux
		self.lognu=numpy.log10(self.nu)
		self.ll=numpy.log10(self.lnu)+self.lognu
		

		# inverts the convention for upper limits, to be consistent with my
		# previous one (Hayden adopts the inverse convention)
		x=self.ul.copy()
		x[self.ul==0]=1
		x[self.ul==1]=0
		self.ul=x

		# Checks if ll has NaN or Inf values
		self.check()



	def prieto(self, file, dist):
		"""
		Reads SEDs in the format provided by Hampadarath et al.:
	nu(Hz) nu*F_nu(Jy Hz) 
		
		Adds the following attributes to the SED object:
		- distance: distance to object in Mpc
		"""
		self.file=file
		self.distance=dist

		self.lognu,nfn = numpy.loadtxt(file,unpack=True,usecols=(0,1),delimiter=',')
		self.nu=10**self.lognu
		nfn=10**nfn*1e-26*1e7*1e-4	# Jy to erg/s/cm^2
		dist=dist*3.086e24	# Mpc to cm
		self.ll=numpy.log10(4.*numpy.pi*dist**2*nfn)
		self.lnu=10**self.ll/self.nu

		# Checks if ll has NaN or Inf values
		self.check()



	def grmonty(self, file):
		"""
		Reads SEDs in the format provided by grmonty. I ported the 
		original SM script `plspec.m` provided by J. Dolence. 
		"""
		# reads grmonty SED file
		self.file=file
		s=ascii.read(file, Reader=ascii.NoHeader)

		# carries out array conversions
		small = 1e-12
		ll0 = numpy.log10(s['col2']+small) + numpy.log10(3.83e33)
		ll1 = numpy.log10(s['col8']+small) + numpy.log10(3.83e33)
		ll2 = numpy.log10(s['col14']+small) + numpy.log10(3.83e33)
		ll3 = numpy.log10(s['col20']+small) + numpy.log10(3.83e33)
		ll4 = numpy.log10(s['col26']+small) + numpy.log10(3.83e33)
		ll5 = numpy.log10(s['col32']+small) + numpy.log10(3.83e33)
		lw = s['col1'] + numpy.log10(9.1e-28*3e10*3e10/6.626e-27)

		# gets log(nu) and log(nuLnu)
		self.lognu, self.ll = lw, ll5
		self.nu,self.nlnu = 10**self.lognu,10**self.ll
		
		# Checks if ll has NaN or Inf values
		self.check()


	def raniere(self, file, dist):
		"""
		Reads SEDs in the format provided by Hampadarath et al.:
		nu(GHz) flux(microJy) error(microJy) Detection[1]orUpperlimit[0]

		:param file: file with SED data
		:param dist: distance in Mpc

		:returns: SED object in units of Hz vs erg/s

		Adds the following attributes to the SED object:
		- ul: is the datapoint an upper limit (arrow)? 1 if yes, 0 if not
		- distance: distance to object in Mpc
		"""
		self.file=file
		self.distance=dist

		f=ascii.read(file)  
		
		nu=f['col3']	# GHz 
		flux=f['col4']	# v*F(v)(flux) [W.m-2]
		fluxerr=f['col5']	# flux error, if 999999 then this is an upper limit with 95 confidence
		self.observatories=f['col2']
		self.references=f['col1']
		
		# preprocessing
		dist=self.distance*3.086e24	# Mpc to cm
		self.nu=nu*1e9 # GHz => Hz		
		self.nlnu=flux*4*numpy.pi*dist**2*1000	# to erg/s
		self.nlnuerr=fluxerr*4*numpy.pi*dist**2*1000	# to erg/s

		self.lognu=numpy.log10(self.nu)
		self.ll=numpy.log10(self.nlnu)
		self.llerr=numpy.abs(self.ll-numpy.log10(self.nlnuerr))
		
		# inverts the convention for upper limits, to be consistent with my
		# previous one (Hayden adopts the inverse convention)
		self.ul=numpy.zeros_like(self.nu)
		self.ul[fluxerr==999999] = 1

		# Checks if ll has NaN or Inf values
		self.check()








	def haydenx(self, file):
		"""
		Reads X-ray data in the format provided by F. Panessa:
		nu	error_nu nuLnu	error	?
		
		Adds the following attributes to the SED object:
		- ul: is the datapoint an upper limit (arrow)? 1 if yes, 0 if not
		"""
		self.file=file
		
		self.lognu,self.lognuerr,self.ll,self.llerr = numpy.loadtxt(file,unpack=True,usecols=(0,1,2,4))
		self.lognuerr=numpy.abs(self.lognu-self.lognuerr)
		self.llerr=numpy.abs(self.ll-self.llerr)
		self.ul=numpy.zeros_like(self.ll)
		
		# Checks if ll has NaN or Inf values
		self.check()










	def check(self):
		"""
	Checks whether there are NaN or Inf values in the array log10(nuLnu).
	If so, displays a warning to the user and the corresponding filename.
	Method called by initialization methods init and erac.
	
	Creates a new attribute, weird, which is 1 if there are NaN or Inf values,
	and 0 otherwise.
		"""
		if False in numpy.isfinite(self.ll):
			print('Warning: NaN or Inf in the Lnu array! '),
			print(self.file)
			self.weird=1
		else:
			self.weird=0








	def sort(self):
		"""
	>>> s.sort()

	Sorts the SED in ascending order of frequencies.
		"""
		# Index of sorted elements
		i=self.lognu.argsort()
		self.nu,self.nlnu = self.nu[i],self.nlnu[i]
		self.lognu,self.ll = self.lognu[i],self.ll[i]
		
		# Precaution in case the user used the interp method
		if hasattr(self, 'nlnui'): 
			self.nui,self.nlnui = self.nui[i],self.nlnui[i]
			self.lognui,self.lli = self.lognui[i],self.lli[i]
		

		
	def unit(self):
		"""Normalizes the spectrum to one."""

		# Precaution in case the user did not use the interp method
		if not hasattr(self, 'nlnui'): self.interp()
	
		self.nlnu=self.nlnu/self.nlnu.max()
		self.ll=numpy.log10(self.nlnu)
		
		# Normalizes the interpolated values
		self.nlnui=self.nlnui/self.nlnui.max()
		self.lli=numpy.log10(self.nlnui)






	def write(self, file):
		"""
		Exports the SED as an ASCII file with the structure:
		col1=log10(nu/Hz), col2=log10(nu Lnu / erg per sec)
		
		>>> s.write(file)

		exports the SED s as the file "file"
		"""
		numpy.savetxt(file,numpy.transpose((self.lognu,self.ll)))






	def interp(self, seds=None, points=1000, xrange=[8,22]):
		"""
	Interpolates the SED and a list of SEDs (optional) in the given range
	of values of log(nu) and for the given number of points.
	
	>>> s.interp([s,s1,s2])
	interpolates s, s1 and s2 with 1000 points in the range
	log(nu)=8-22 (Hz). s, s1 and s2 are also instances of the class SED. The
	interpolated arrays (lognui, lli, nui, nlnui) will be added as new 
	attributes to each object. If given no argument, will interpolate the
	current SED object.
	
	:param seds: list of SED objects to interpolate
	:param points: number of interpolated points
	:param xrange: list in the form [xinitial, xfinal] with the x-range of interpolation. 
	
	If provided with no list of SEDs (e.g., s.interp()), then the method
	interpolates only the current SED s. If provided with no xrange argument,
	it will by default assume the range [8,22] discarding data points outside
	that range in the interpolation, hence adjust accordingly.
		"""
		# If seds==None, then the method operates only on the object itself 
		# (e.g., s.interp()) and proper actions are taken to avoid problems
		if seds==None:
			# Defines the new array of interpolated frequencies (with a 
			# preferentially large number of points)
			xold, yold = self.lognu, self.ll
			xnew=numpy.linspace(xrange[0],xrange[-1],points)
			ynew=numpy.interp(xnew,xold,yold,left=-20,right=-20)

			# Interpolates the current SED (simple linear interpolation).
			# Interpolated y-values outside the original x-range are zeroes (-20
			# in log space of luminosities).			
			self.lognui, self.lli = xnew, ynew	# log scale
			self.nui, self.nlnui = 10.**xnew, 10.**ynew
		else:
			# Interpolates the other SEDs with the same binning. But only does
			# that if a list of SEDs was actually provided!
			for sed in seds:
				xold, yold = sed.lognu, sed.ll
				xnew=numpy.linspace(xrange[0],xrange[-1],points)
				ynew=numpy.interp(xnew,xold,yold,left=-20,right=-20)
			
				sed.lognui, sed.lli = xnew, ynew	# log scale
				sed.nui, sed.nlnui = 10.**xnew, 10.**ynew




		



	
	
	
	def normalize(self, seds=None, nuref=17.684, refnlnu=1e40, xray=None):
		"""
	Normalizes a SED at the given frequency (lognu/Hz) and nuLnu (erg/s). 
	NOTE: this will modify the SED(s). Alternatively, if the user provides 
	the argument xray=True, then the method normalizes the SEDs such that 
	they have the same X-ray luminosity refnlnu in the range 2-10 keV 
	(thus ignoring the arg.	nuref).
	
	>>> s.normalize([s1,s2,s3])

	normalizes the SEDs in the object list at the same frequency and nuLnu
	
	>>> s.normalize([s1,s2,s3],refnlnu=1e40,xray=True)

	normalizes the SEDs such that they have the same X-ray luminosity 1e40 
	erg/s in the range 2-10 keV.
	
	The method is designed to automatically call the interp method (e.g., 
	s.interp([s1,s2])) if needed.
		"""
		# Precaution in case the user did not use the interp method
		if not hasattr(self, 'nlnui'): self.interp(seds)
		
		# If seds==None, then the method operates on the object itself, 
		# otherwise it operates on a list of objects
		if seds==None:
			if xray==None:	# if xray keyword is not provided
				# Finds the nuLnu corresponding to the frequency nearest nuref.
				# Uses the interpolated arrays for this purpose.
				i=lsd.search(nuref, self.lognui)
			
				self.nlnu=self.nlnu*refnlnu/self.nlnui[i]
				self.nlnui=self.nlnui*refnlnu/self.nlnui[i]
				self.lli=numpy.log10(self.nlnui)
				self.ll=numpy.log10(self.nlnu)
			else:	# if xray keyword IS provided
				self.xrays()
				factor=refnlnu/self.lumx	# normalization factor
				
				self.nlnu=self.nlnu*factor
				self.nlnui=self.nlnui*factor
				self.lli=numpy.log10(self.nlnui)
				self.ll=numpy.log10(self.nlnu)
		else:
			for sed in seds:
				if xray==None:	# if xray keyword is not provided
					i=lsd.search(nuref, sed.lognui)
			
					sed.nlnu=sed.nlnu*refnlnu/sed.nlnui[i]
					sed.nlnui=sed.nlnui*refnlnu/sed.nlnui[i]
					sed.lli=numpy.log10(sed.nlnui)
					sed.ll=numpy.log10(sed.nlnu)
				else:	# if xray keyword IS provided
					sed.xrays()
					factor=refnlnu/sed.lumx	# normalization factor
				
					sed.nlnu=sed.nlnu*factor
					sed.nlnui=sed.nlnui*factor
					sed.lli=numpy.log10(sed.nlnui)
					sed.ll=numpy.log10(sed.nlnu)
		

		
		
		
		
	def meanlin(self, seds, nuref=17.684, refnlnu=1e40):
		"""
	Given a list of SEDs and a reference value of log10(nu), normalizes the 
	SEDs (s.normalize) at the given nuref and refnlnu, calculates the average 
	of their luminosities (log10(average nuLnu)) and returns the list [mean,sd] 
	where 'mean' is SED object with the mean and 'sd' is the object with the
	standard deviation.
	
	>>> mean=s.meanlin([s1,s2],17.684,1e40)
	returns mean <- [lognu, <s1,s2>],
	where <s,s1,s2> -> log10[<nuLnu(s),nuLnu(s1),nuLnu(s2)>], lognu being the common 
	units of frequency for the SEDs after interpolation.
		"""	
		self.normalize(seds,nuref,refnlnu)	# normalize them to a common nu
		
		s=self.sum(seds)
		lognu=s.lognu
		m=s.nlnu/len(seds)	# takes the average
		
		# Computes the standard deviation
		sd=numpy.zeros_like(m)	# initializes the sd array		
		for sed in seds:
			sd=sd+(sed.nlnui-m)**2
		sd=numpy.sqrt(sd/len(seds))
		
		meansed=SED(lognu=lognu, ll=numpy.log10(m), logfmt=1)
		sdsed=SED(lognu=lognu, ll=numpy.log10(sd), logfmt=1)
		
		return [meansed,sdsed]		
		#return SED(lognu=lognu, ll=numpy.log10(m))
		





		
	def mean(self, seds, nuref=17.684, refnlnu=1e40, xray=None):
		"""
	Given a list of SEDs and a reference value of log10(nu), normalizes the 
	SEDs (s.normalize) at the given nuref and refnlnu, calculates the average 
	of their luminosities (average(log10(nuLnu)), like Eracleous et al. 2010 does) 
	and returns the list [mean,sd] where 'mean' is SED object with the mean and 
	'sd' is the object with the	standard deviation.
	
	Alternatively, if the user provides the argument xray=True, then
	the method normalizes the SEDs such that they have the same X-ray
	luminosity refnlnu in the range 2-10 keV (thus ignoring the arg.
	nuref).
	
	>>> means=s.mean([s1,s2],17.684,1e40)

	returns mean <- [lognu, <s1,s2>],
	where <s,s1,s2> -> <log10[nuLnu(s),nuLnu(s1),nuLnu(s2)]>, lognu being the common 
	units of frequency for the SEDs after interpolation.
	
	>>> means=s.mean([s1,s2],refnlnu=1e40,xray=True)

	returns the mean after normalizing the SEDs to the X-ray lum.
	1e40 erg/s in the range 2-10 keV.
		"""
		self.normalize(seds,nuref,refnlnu,xray)	# normalize them to a common nu
			
		# Sum the values of log10(nuLnu)
		sums=numpy.zeros_like(seds[0].lli)	# initializes the sum
		for sed in seds:
			sums=sums+sed.lli

		lognu=seds[0].lognui		
		m=sums/len(seds)	# takes the average
		
		# Computes the standard deviation
		sd=numpy.zeros_like(sums)	# initializes the std. dev. array
		for sed in seds:
			sd=sd+(sed.lli-m)**2
		sd=numpy.sqrt(sd/len(seds))
		
		meansed=SED(lognu=lognu, ll=m, logfmt=1)
		sdsed=SED(lognu=lognu, ll=sd, logfmt=1)
		
		return [meansed,sdsed]
		#return SED(lognu=lognu, ll=m)





	def median(self, seds, nuref=17.684, refnlnu=1e40, xray=None):
		"""
	Given a list of SEDs and a reference value of log10(nu), normalizes the 
	SEDs (s.normalize) at the given nuref and refnlnu, calculates the average 
	of their luminosities (average(log10(nuLnu)), like Eracleous et al. 2010 does) 
	and returns the list [mean,sd] where 'mean' is SED object with the median and 
	'sd' is the object with the	standard deviation.
	
	Alternatively, if the user provides the argument xray=True, then
	the method normalizes the SEDs such that they have the same X-ray
	luminosity refnlnu in the range 2-10 keV (thus ignoring the arg.
	nuref).
	
	>>> means=s.median([s1,s2],17.684,1e40)

	returns median <- [lognu, <s1,s2>],
	where <s,s1,s2> -> <log10[nuLnu(s),nuLnu(s1),nuLnu(s2)]>, lognu being the common 
	units of frequency for the SEDs after interpolation.
	
	>>> means=s.median([s1,s2],refnlnu=1e40,xray=True)

	returns the mean after normalizing the SEDs to the X-ray lum.
	1e40 erg/s in the range 2-10 keV.
		"""
		self.normalize(seds,nuref,refnlnu,xray)	# normalize them to a common nu
		
		# Array that stores the median of luminosities
		m=numpy.zeros_like(seds[0].lli)
		
		# Loop that uses the numpy.median method
		for i in numpy.arange(m.size):
			x=numpy.zeros_like(numpy.array(seds))
			for j in numpy.arange(numpy.size(seds)):
				x[j]=seds[j].lli[i]
			m[i]=numpy.median(x)

		lognu=seds[0].lognui

		# Computes the standard deviation
		sd=numpy.zeros_like(m)	# initializes the std. dev. array
		for sed in seds:
			sd=sd+(sed.lli-m)**2
		sd=numpy.sqrt(sd/len(seds))
		
		meansed=SED(lognu=lognu, ll=m, logfmt=1)
		sdsed=SED(lognu=lognu, ll=sd, logfmt=1)
		
		return [meansed,sdsed]






	




	def geomean(self, seds, nuref=17.684, refnlnu=1e40, xray=None):
		"""
	Given a list of SEDs and a reference value of log10(nu), normalizes the 
	SEDs (s.normalize) at the given nuref and refnlnu, calculates the geometric
	mean of their luminosities (average(log10(nuLnu)), like Eracleous et al. 2010 does) 
	and returns the list [mean,sd] where 'mean' is SED object with the geometric 
	mean and 'sd' is the object with the standard deviation.
	
	Alternatively, if the user provides the argument xray=True, then
	the method normalizes the SEDs such that they have the same X-ray
	luminosity refnlnu in the range 2-10 keV (thus ignoring the arg.
	nuref).
	
	>>> means=s.mean([s1,s2],17.684,1e40)

	returns mean <- [lognu, <s1,s2>],
	where <s,s1,s2> -> <log10[nuLnu(s),nuLnu(s1),nuLnu(s2)]>, lognu being the common 
	units of frequency for the SEDs after interpolation.
	
	>>> means=s.mean([s1,s2],refnlnu=1e40,xray=True)
	
	returns the mean after normalizing the SEDs to the X-ray lum.
	1e40 erg/s in the range 2-10 keV.
		"""	
		self.normalize(seds,nuref,refnlnu,xray)	# normalize them to a common nu
		
		# Calculates the product of log(nuLnu) (produtorio)
		sums=seds[0].lli.copy()	# initializes the sum
		for i in numpy.arange(1,len(seds)):
			sums=sums*seds[i].lli

		lognu=seds[0].lognui		
		m=sums**(1./len(seds)) 	# takes the nth root
		
		# Computes the standard deviation
		sd=sums.copy()	# initializes the std. dev. array
		for sed in seds:
			sd=sd+(sed.lli-m)**2
		sd=numpy.sqrt(sd/len(seds))
		
		meansed=SED(lognu=lognu, ll=m, logfmt=1)
		sdsed=SED(lognu=lognu, ll=sd, logfmt=1)
		
		return [meansed,sdsed]







	
	
	
	def bol(self, lognu0=None, lognu1=None, xrange=[8,22]):
		"""
	Calculates the bolometric luminosity of the SED. Performs the integration 
	using the trapezoidal rule. Adds the attribute lumbol to the object with
	the result.
	
	>>> s.bol()

	makes the variable s.lumbol hold the bolometric luminosity.
	
	>>> s.bol(17, 20)

	makes s.lumbol hold the luminosity integrated in the range 1e17-1e20 Hz.

	>>> s.bol(18, 20, [s.lognu[0],s.lognu[-1]])

	gives the integrated luminosity in the range 1e18-1e20 Hz. Before integrating,
	interpolates the SED over its original frequency range with 1000 points.

		"""
		import scipy.integrate

		# Performs interpolation before integrating. This is a precaution in 
		# case the user specifies weird integration limits.	In addition, I 
		# found out that integrating certain sparse SEDs (e.g. qsos) without
		# interpolating induces incorrect Lbol estimates		
		if not hasattr(self, 'nlnui'): self.interp(xrange=xrange)			
		
		# If no arguments are given to the method, integrates the entire SED
		if lognu0==None and lognu1==None:
			lumbol=scipy.integrate.trapz(self.nlnui/self.nui, self.nui)
		else:
			# Gets only the elements in the given range
			i=numpy.where((self.lognui>=lognu0) & (self.lognui<=lognu1))
			
			# Integration
			lumbol=scipy.integrate.trapz(self.nlnui[i]/self.nui[i], self.nui[i])

		self.lumbol=lumbol

		return lumbol





	def edd(self,mass=None):
		"""
	Computes the Eddington ratio :math:`L_{\\rm bol}/L_{\\rm} Edd}`.
	First computes the bolometric luminosity via interpolation and 
	then calculates the Eddington ratio.

	:param mass: log10(BH mass in solar masses)
		"""
		# Computes Lbol before anything else	
		if not hasattr(self, 'lumbol'): 			
			lumbol=self.bol()

		# if no mass argument was given
		if mass==None: 
			mass=self.mass

		# Eddington ratio
		self.lumedd=1.3e38*10**mass
		self.eddratio=lumbol/self.lumedd

		return self.eddratio

	


	def xrays(self):
		"""
	Calculates the photon index of the X-ray spectrum and Lx (2-10 keV). 
	Stores the new attributes gammax and lumx in the object.
	
	>>> Lx,gamma=s.xrays()

	creates the new attributes s.lumx and s.gammax
		"""
		import scipy.integrate
		import scipy.stats
		
		# Performs interpolation before integrating. This is a precaution in 
		# case the user specify weird integration limits.			
		if not hasattr(self, 'nlnui'): self.interp()
		
		# 2-10 keV in log(nu)
		xi, xf = 17.684, 18.384
		
		# Gets only the elements in the range 2-10 keV
		i=numpy.where((self.lognui>=xi) & (self.lognui<=xf))
		x,y = self.lognui[i],self.lli[i]	# easier notation
		
		# Calculates Lx using integration (trapezoidal rule)
		self.lumx=scipy.integrate.trapz(self.nlnui[i]/self.nui[i], self.nui[i])

		# Linear fit to the SED in the interval 2-10 keV, y=ax+b
		a, b, r, p, err = scipy.stats.linregress(x,y)
		
		# Calculates Gamma_X (photon index)
		self.gammax=2-a

		return self.lumx, self.gammax
		
		
		
		
		
	
	def findlum(self,x):
		"""
	Given the input frequency in units of log(nu/Hz), returns the list [log(nu),
	Lnu] with the frequency and luminosity closest to the specified value.
	
	>>> lognu,lum=s.findlum(14)

	will look for the freq. and lum. nearest nu=10^14 Hz
		"""
		# Performs interpolation before the search
		if not hasattr(self, 'nlnui'): self.interp()
		
		# Looks for the frequency
		i=lsd.search(x,self.lognui)	# index
		
		return [self.lognui[i], self.nlnui[i]/self.nui[i]]
	
	
	
	
	
	
	def radioloud(self):
		"""
	Computes the radio-loudness for the SED. Adds the attribute rl to the 
	object with	the result. If Ro>=10. we have a RL AGN, otherwise it is a
	RQ one.
	
	>>> rl=s.radioloud()

	makes the variable s.rl hold the radio-loudness.
		"""
		# Radio luminosity at 6 cm. 6 cm -> 4.997x10^9 Hz -> lognu=9.7
		lumrad=self.findlum(9.7)[1]
		
		# Optical lum. at 4400 A. B band -> 4400 A -> 6.813x10^14 Hz -> lognu=14.833 
		lumopt=self.findlum(14.833)[1]
				
		# Kellermann recipe: Lnu(6 cm)/Lnu(B)
		return lumrad/lumopt





	def alphaox(self):
		"""
	Computes the alpha_ox index.

	>>> alphaox=s.alphaox()
		"""
		# Lnu(2 keV)
		lumx=self.findlum(17.684)[1]
		
		# UV lum. at 2500 AA -> 1.199e15 Hz  -> lognu=15.079 
		lumuv=self.findlum(15.079)[1]
		
		return numpy.log10(lumx/lumuv)/2.605





	def ion(self):
		"""
	Calculates the rate of ionizing photons in the SED.
	
	>>> q=s.ion()
		"""
		import scipy.integrate
		import scipy.stats
		h=6.62607e-27	# Planck constant in CGS
		
		# Performs interpolation before integrating. This is a precaution in 
		# case the user specify weird integration limits.			
		if not hasattr(self, 'nlnui'): self.interp()
		
		# 13.6 eV - "infty"
		xi, xf = 15.52, 22.
		
		# Gets only the elements corresponding to ionizing frequencies
		i=numpy.where((self.lognui>=xi) & (self.lognui<=xf))
		x,y = self.lognui[i],self.lli[i]	# easier notation
		
		# Calculates ionizing rate using integration (trapezoidal rule)
		q=scipy.integrate.trapz(self.nlnui[i]/self.nui[i]/(h*self.nui[i]), self.nui[i])

		return q


		
		
	
	
	
	def chisq(self, model, npars=None):
		"""
	Computes the goodness of fit between the observed SED (assumed to be "self") 
	and the given model. If you don't provide the number of free 
	parameters in the model, it returns a modified chi squared (modified in 
	the sense that it assumes the uncertainties in all data points to be unity).
	If you provide npars it returns the modified reduced chi squared.
	
	>>> s.chisq(m)
	returns the (modified) chi square with s and m being the observed and model
	SEDs, respectively.
	
	>>> s.chisq(m,4)
	returns the reduced chi square.
	
	NOTE: this method was written assuming that the observed SEDs correspond
	to Eracleous' SEDs. Needs to be adapted in order to be applied to other 
	types of SEDs. We discard the 100 keV interpolation done by EHF10.
		"""
		obs=self
		
		# Discards upper limits in the data and the 100 keV interpolated point 
		i=numpy.where((obs.ul==0) & (obs.lognu<19.))	# index of elements corresponding to good elements
		
		# Interpolates the model SED in the same binning as the obs.
		yobs=obs.ll[i]
		ymod=numpy.interp(obs.lognu[i],model.lognu,model.ll)

		# Chi-square 
		if npars==None:
			chisq=numpy.sum((yobs-ymod)**2)
		else:	# or reduced chi-square statistic
			nu=yobs.size-1-npars		# Number of degrees of freedom
			chisq=numpy.sum((yobs-ymod)**2)/nu
			
		return chisq
	
	
	
	
	
	def copy(self):
		"""
	Returns a copy of the current SED object. Use this in the same way you would use
	the numpy copy() method to avoid changing the attributes of a SED
	unintentionally.
		"""
		c=SED(lognu=self.lognu,ll=self.ll,file=self.file)
		
		# Precaution if SED was imported using erac method
		if hasattr(self, 'llex'):
			c.llex=self.llex
			c.nlnuex=self.nlnuex
			c.ul=self.ul
		
		return c




def sum(seds):
	"""
Given a list of SEDs previously interpolated in the same binning, 
sums their luminosities and returns a SED object with the sum.
	
>>> ss=sum([s,s1,s2])

returns the SED ss <- [lognu, s+s1+s2],
where s+s1+s2 -> log10[nuLnu(s)+nuLnu(s1)+nuLnu(s2)], lognu being the common 
units of frequency for the SEDs after interpolation.
	
The method is designed to automatically call the interp method for each SED
if needed.
	"""
	# Precaution in case the user did not use the interp method
	seds[0].interp(seds)
		
	sums=numpy.zeros_like(seds[0].lognui)	# initializes the sum

	for sed in seds:
		sums=sums+sed.nlnui
		
	return SED(lognu=seds[0].lognui, ll=numpy.log10(sums), logfmt=1)

		







def haydensed(source, patho='/Users/nemmen/work/projects/hayden/all/', info='/Users/nemmen/work/projects/hayden/info.dat'):
	"""
Reads an observed SED from the Hayden et al. sample. Computes useful 
quantities.

:param source: source name e.g. 'NGC1097' or 'ngc1097' or 'NGC 1097'. Must have four numbers.
:param patho: path to observed SED data files
:param info: path to information datafile with distances and masses

:returns: SED object 
	"""
	# READS SEDS
	# =============
	# Finds the required parameters to plot the data for the specified source.
	# First converts the source name to uppercase, no spaces
	source=source.upper()
	source=source.replace(' ', '')

	# Reads information table and gets required info
	t=ascii.read(info)
	names=t['name'].tolist()
	# finds location of source in the arrays
	i=names.index(source)
	# gets distance 
	dist=t['distance/Mpc'][i]
	# gets BH mass
	mass=t['logmass'][i]

	# Reads observed SED from radio to X-rays 
	# (which is given in weird units)
	ofile=patho+source+'_sed.txt'
	o=SED()
	o.hayden(ofile,dist)
	o.mass=mass	
		
	# X-ray data
	# Reads Panessa's X-ray SED and creates another SED object
	xfile=patho+source+'_x.txt'
	x=SED()
	x.haydenx(xfile)	

	# Stacks together all the multiwavelength information
	o.lognu=numpy.concatenate([o.lognu,x.lognu])
	o.ll=numpy.concatenate([o.ll,x.ll])	

	return o









def modelsplot(pathm='/Users/nemmen/Dropbox/work/projects/adafjet/',
	jetfile=None,showthin=True,showjet=True,showsum=True,showadaf=[True,True,True]):
	"""
Nice wrapper that plots 3 models computed with the ADAF Fortran code available at
~/work/projects/adafjet/adaf.

Useful when modeling a given SED.

Input:
  - *file : path to ASCII files containing the precomputed models
  - showadaf : should I display the three ADAF models? (list or array of booleans, 
  	e.g., [True,True,True])
  - showthin : Include thin disk?
  - showjet : Display jet?
  - showsum : Show sum of components?
	"""
	if showjet==True:
		if jetfile==None:
			jet=SED(file=pathm+'jet/jet.dat',logfmt=1)
		else:
			jet=SED(file=jetfile,logfmt=1)	
		pylab.plot(jet.lognu,jet.ll,'-.m')

	if showadaf[0]==True:
		adaf=SED(file=pathm+'adaf/perl/run01/spec_01',logfmt=1)
		pylab.plot(adaf.lognu,adaf.ll,'--b',linewidth=1)
		if showthin==True:
			thin=SED(file=pathm+'adaf/perl/run01/spec_01_ssd',logfmt=1) 
			#thin=SED(file=thinfile,logfmt=1) 
			pylab.plot(thin.lognu,thin.ll,':r',linewidth=1)
		if showjet and showthin: sum=sum([adaf,jet,thin])
		if showjet==True and showthin==False: sum=sum([adaf,jet])
		if showjet==False and showthin==True: sum=sum([adaf,thin])
		if showsum==True: pylab.plot(sum.lognu,sum.ll,'k')

	if showadaf[1]==True:
		adaf=SED(file=pathm+'adaf/perl/run02/spec_02',logfmt=1)
		pylab.plot(adaf.lognu,adaf.ll,'--b',linewidth=2)
		if showthin==True:
			thin=SED(file=pathm+'adaf/perl/run02/spec_02_ssd',logfmt=1) 
			#thin=SED(file=thinfile,logfmt=1) 
			pylab.plot(thin.lognu,thin.ll,':r',linewidth=2)
		if showjet and showthin: sum=sum([adaf,jet,thin])
		if showjet==True and showthin==False: sum=sum([adaf,jet])
		if showjet==False and showthin==True: sum=sum([adaf,thin])
		if showsum==True: pylab.plot(sum.lognu,sum.ll,'k')

	if showadaf[2]==True:
		adaf=SED(file=pathm+'adaf/perl/run03/spec_03',logfmt=1)
		pylab.plot(adaf.lognu,adaf.ll,'--b',linewidth=3)
		if showthin==True:
			thin=SED(file=pathm+'adaf/perl/run03/spec_03_ssd',logfmt=1) 
			#thin=SED(file=thinfile,logfmt=1) 	
			pylab.plot(thin.lognu,thin.ll,':r',linewidth=3)
		if showjet and showthin: sum=sum([adaf,jet,thin])
		if showjet==True and showthin==False: sum=sum([adaf,jet])
		if showjet==False and showthin==True: sum=sum([adaf,thin])
		if showsum==True: pylab.plot(sum.lognu,sum.ll,'k')





def modelplot(ssdf,adaff,jetf,sumf,**args):
	"""
Plots the model components.

Arguments: The filenames of each model component.
	"""
	# Reads model SEDs, checking first if the data files exist
	if os.path.isfile(sumf): msum=SED(file=sumf,logfmt=1) 
	else: msum=None
	if os.path.isfile(adaff): adaf=SED(file=adaff,logfmt=1) 
	else: adaf=None
	if os.path.isfile(jetf): jet=SED(file=jetf,logfmt=1) 
	else: jet=None
	if os.path.isfile(ssdf): ssd=SED(file=ssdf,logfmt=1) 
	else: ssd=None
	
	# Groups the model components in a dictionary
	m=groupmodel(sum=msum,ssd=ssd,adaf=adaf,jet=jet)

	if 'sum' in m: pylab.plot(m['sum'].lognu,m['sum'].ll,'k',**args)
	if 'adaf' in m: pylab.plot(m['adaf'].lognu,m['adaf'].ll,'--',**args)
	if 'jet' in m: pylab.plot(m['jet'].lognu,m['jet'].ll,'-.',**args)
	#if 'ssd' in m: pylab.plot(m['ssd'].lognu,m['ssd'].ll,':',**args)
	if 'ssd' in m: pylab.plot(m['ssd'].lognu,m['ssd'].ll,':',linewidth=2,**args)













def obsplot(source,
	  table='/Users/nemmen/Dropbox/work/projects/finished/liners/tables/models.csv',
	  patho='/Users/nemmen/Dropbox/work/projects/finished/liners/seds/obsdata/',
	  ext='0'
	  ):
	"""
Plots the observed SED taken from the Eracleous et al. sample. Includes arrows 
for upper limits and the nice bow tie. 

Extinction corrections represented as either "error bars" or "circle + error bar".

:param source: source name e.g. 'NGC1097' or 'ngc1097' or 'NGC 1097'. Must have four numbers.
:param table: path of ASCII table models.csv with the required SED parameters gammax, error in gammax, X-ray luminosity and distratio
:param patho: path to observed SED data files
:param ext: 0 for showing extinction corrections as "error bars"; 1 for "circle + error bar"
:returns: SED object with distance-corrected luminosities
	"""
	# Reads the ASCII table models.csv
	t=ascii.read(table,Reader=ascii.CommentedHeader,delimiter='\t')
	tname,tmodel,tdistratio,tgammax,tegammal,tegammau,tlumx=t['Object'],t['Model'],t['distratio'],t['gammax'],t['egammal'],t['egammau'],t['lumx']

	# Finds the required parameters to plot the data for the specified source.
	# First converts the source name to uppercase, no spaces
	source=source.upper()
	source=source.replace(' ', '')
	# Finds the first occurrence of the source in the table
	i=numpy.where(tname==source)
	i=i[0][0]
	# Retrieves the required parameters
	distratio=tdistratio[i]
	gammax=tgammax[i]
	egammal=tegammal[i]
	egammau=tegammau[i]
	lumx=tlumx[i]

	# Reads observed SED
	ofile=patho+source+'.dat'
	o=SED()
	o.erac(ofile)	
	
	# Corrects SED data points for new distance
	o.nlnu=o.nlnu*(distratio)**2
	o.nlnuex=o.nlnuex*(distratio)**2
	o.ll=numpy.log10(o.nlnu)
	o.llex=numpy.log10(o.nlnuex)
	lumx=lumx*(distratio)**2
		
	# Separates the observed SED in the different bands and uses a different recipe
	# to plot each region

	## Radio and IR as filled points
	rir=numpy.where(o.lognu<14.46)
	pylab.plot(o.lognu[rir],o.ll[rir],'ko')

	## Optical and UV (extinction uncertainty)
	ouv=numpy.where((o.lognu>=14.46) & (o.lognu<=16.))
	if (numpy.size(ouv)==1) and (o.llex[ouv]-o.ll[ouv]>2.):
		# if the uncertainty in the UV due to extinction is too big (NGC 3169, NGC3226 and NGC4548), replace the "error bar" with an upper limit
		pylab.plot(o.lognu[ouv],o.llex[ouv],'ko')
		pylab.quiver(o.lognu[ouv],o.llex[ouv],0,-1,scale=30,width=3e-3)
	else:
		if ext==0: 
			pylab.errorbar(o.lognu[ouv],(o.ll[ouv]+o.llex[ouv])/2.,yerr=o.llex[ouv]-(o.ll[ouv]+o.llex[ouv])/2.,fmt=None,ecolor='k',label='_nolegend_')
		else:
			pylab.plot(o.lognu[ouv],o.ll[ouv],'ko')
			pylab.errorbar(o.lognu[ouv],(o.ll[ouv]+o.llex[ouv])/2.,yerr=o.llex[ouv]-(o.ll[ouv]+o.llex[ouv])/2.,fmt=None,ecolor='k',label='_nolegend_')
	
	## X-rays as bow tie
	xray(lumx,gammax,egammal,egammau)	
	
	# Upper limits with arrows
	u=numpy.where(o.ul==1)	# index of upper limits
	pylab.quiver(o.lognu[u],o.ll[u],0,-1,scale=30,width=3e-3)	# careful with the arrow parameters	

	# Dummy plot just to include the name of the source with legend later
	pylab.plot(o.lognu+50,o.ll,'w',label=source)

	return o

	






def haydenobs(source, patho='/Users/nemmen/work/projects/hayden/all/', info='/Users/nemmen/work/projects/hayden/info.dat'):
	"""
Plots the observed SED from the Hayden et al. sample. Includes arrows 
for upper limits and the nice bow tie. 

Extinction corrections represented as either "error bars" or "circle + error bar".

:param source: source name e.g. 'NGC1097' or 'ngc1097' or 'NGC 1097'. Must have four numbers.
:param patho: path to observed SED data files
:param info: path to information datafile with distances and masses
:returns: SED object 
	"""
	# Finds the required parameters to plot the data for the specified source.
	# First converts the source name to uppercase, no spaces
	source=source.upper()
	source=source.replace(' ', '')

	# Reads information table and gets required info
	t=ascii.read(info)
	names=t['name'].tolist()
	# finds location of source in the arrays
	i=names.index(source)
	# gets distance 
	dist=t['distance/Mpc'][i]

	# Reads observed SED (which is given in weird units)
	ofile=patho+source+'_sed.txt'
	o=SED()
	o.hayden(ofile,dist)	
	
	#########################################################	
	# Separates the observed SED in the different bands and uses a different recipe
	# to plot each region
	#########################################################	

	pylab.plot(o.lognu,o.ll,'ko')
	
	## X-rays as bow tie
	#xray(lumx,gammax,egammal,egammau)	
	# Reads Panessa's X-ray SED
	xfile=patho+source+'_x.txt'
	x=SED()
	x.haydenx(xfile)	
	pylab.plot(x.lognu,x.ll,'k.')
	pylab.errorbar(x.lognu,x.ll,yerr=x.llerr,fmt="none",label='_nolegend_',alpha=0.5)



	
	# Upper limits with arrows
	u=numpy.where(o.ul==1)	# index of upper limits
	pylab.quiver(o.lognu[u],o.ll[u],0,-1,scale=30,width=3e-3)	# careful with the arrow parameters	

	# Dummy plot just to include the name of the source with legend later
	pylab.plot(o.lognu+50,o.ll,'w',label=source)

	return o








# ====================================
#
# Code below imported from sedplot.py
#
# ====================================

def plotprops(labelfontsize=18, legend=True, option=1, loc='upper right'):
	"""
Define other properties of the SED plot: additional axis, font size etc.
	"""
	# Increase the size of the fonts
	pylab.rcParams.update({'font.size': 15})

	# Defines general properties of the plot
	if option==1:
		pylab.xlim(8,20) # Nemmen et al. 2014 plots
		pylab.ylim(35,44) # Nemmen et al. 2014 plots

	if option==2: 
		pylab.ylim(34,40) # Hayden plots
		pylab.xlim(8,19)

	if option==3:
		pylab.xlim(8,27)	# with Fermi data (large x-axis!)

	pylab.xlabel('log($\\nu$ / Hz)',fontsize=labelfontsize)
	pylab.ylabel('log($\\nu L_\\nu$ / erg s$^{-1}$)',fontsize=labelfontsize)
	pylab.minorticks_on()
	if legend is True:
		pylab.legend(loc=loc, frameon=False)

	# Add second X axis with common units of wavelength
	ax1=pylab.subplot(111)
	ax2=pylab.twiny()
	pylab.minorticks_on()

	if option==1 or option==2:
		ax1.set_xlim(8,20)	
		ax2.set_xlim(8,20)	# set this to match the lower X axis
		ax2.set_xticks([8.477,9.477,10.477,11.477,12.477,13.477,14.477,15.4768,16.383,17.383,18.383,19.383])
		ax2.set_xticklabels(['1m','10cm','1cm','1mm','100$\mu$m','10$\mu$m','1$\mu$m','1000$\AA$','.1keV','1keV','10keV','100keV'],size=10.5)

	if option==3:
		ax1.set_xlim(8,27)	
		ax2.set_xlim(8,27)	# set this to match the lower X axis
		ax2.set_xticks([8.477,10.477,12.477,14.477,16.383,18.383,20.383,22.383,24.383,26.383])
		ax2.set_xticklabels(['1m','1cm','100$\mu$m','1$\mu$m','0.1keV','10keV','1MeV','100MeV','10GeV','1TeV'],size=10.5)






def plot(source,o,m,table):
	"""
DEPRECATED. Split these method in separate units: modelplot, obsplot, plotprops.
Main method to create the SED plot.

Arguments:
- source: source name e.g. 'NGC1097' or 'ngc1097' or 'NGC 1097'. Must have four 
	numbers.
- o: observed SED data points imported using the SED class and the erac method.
- m: dictionary with the different model components created with the method
	groupmodel below.
- table: path of ASCII table models.csv with the required SED parameters gammax, 
	error in gammax, X-ray luminosity and distratio.
	"""
	# Reads the ASCII table models.csv
	t=ascii.read(table,Reader=ascii.CommentedHeader,delimiter='\t')
	tname,tmodel,tdistratio,tgammax,tegammal,tegammau,tlumx=t['Object'],t['Model'],t['distratio'],t['gammax'],t['egammal'],t['egammau'],t['lumx']

	# Finds the required parameters to plot the data for the specified source.
	# First converts the source name to uppercase, no spaces
	source=source.upper()
	source=source.replace(' ', '')
	# Finds the first occurrence of the source in the table
	i=numpy.where(tname==source)
	i=i[0][0]
	# Retrieves the required parameters
	distratio=tdistratio[i]
	gammax=tgammax[i]
	egammal=tegammal[i]
	egammau=tegammau[i]
	lumx=tlumx[i]

	# Precaution to avoid modifying unintentionally the obs SED
	o=o.copy()
	
	


	pylab.clf()
	

	
	
	# Plots OBSERVED SED
	# ====================
	# Corrects SED data points for new distance
	o.nlnu=o.nlnu*(distratio)**2
	o.nlnuex=o.nlnuex*(distratio)**2
	o.ll=numpy.log10(o.nlnu)
	o.llex=numpy.log10(o.nlnuex)
	lumx=lumx*(distratio)**2
		
	# Separates the observed SED in the different bands and uses a different recipe
	# to plot each region
	## Radio and IR as filled points
	rir=numpy.where(o.lognu<14.46)
	pylab.plot(o.lognu[rir],o.ll[rir],'ko')
	## Optical and UV as error bars (extinction uncertainty)
	ouv=numpy.where((o.lognu>=14.46) & (o.lognu<=16.))
	if (numpy.size(ouv)==1) and (o.llex[ouv]-o.ll[ouv]>2.):
		# if the uncertainty in the UV due to extinction is too big (NGC 3169, NGC3226 and NGC4548), replace the "error bar" with an upper limit
		pylab.plot(o.lognu[ouv],o.llex[ouv],'ko')
		pylab.quiver(o.lognu[ouv],o.llex[ouv],0,-1,scale=30,width=3e-3)
	else:
		pylab.errorbar(o.lognu[ouv],(o.ll[ouv]+o.llex[ouv])/2.,yerr=o.llex[ouv]-(o.ll[ouv]+o.llex[ouv])/2.,fmt=None,ecolor='k',label='_nolegend_')
	## X-rays as bow tie
	xray(lumx,gammax,egammal,egammau)	
	
	# Upper limits with arrows
	u=numpy.where(o.ul==1)	# index of upper limits
	pylab.quiver(o.lognu[u],o.ll[u],0,-1,scale=30,width=3e-3)	# careful with the arrow parameters	

	# Dummy plot just to include the name of the source with legend later
	pylab.plot(o.lognu+50,o.ll,'w',label=source)

	
	

	# Plots MODEL SEDs
	# ==================
	# Please be sure that m was created with the groupmodel method
	
	if 'sum' in m: pylab.plot(m['sum'].lognu,m['sum'].ll,'k')
	if 'adaf' in m: pylab.plot(m['adaf'].lognu,m['adaf'].ll,'--')
	if 'jet' in m: pylab.plot(m['jet'].lognu,m['jet'].ll,'-.')
	if 'ssd' in m: pylab.plot(m['ssd'].lognu,m['ssd'].ll,':')



	
	# Defines general properties of the plot
	pylab.xlim(8,20)
	pylab.ylim(35,44)
	pylab.xlabel('log($\\nu$ / Hz)')
	pylab.ylabel('log($\\nu L_\\nu$ / erg s$^{-1}$)')
	pylab.legend(loc='best')
	pylab.minorticks_on()
	pylab.legend(loc='upper right', frameon=False)

	# Add second X axis with common units of wavelength
	ax1=pylab.subplot(111)
	ax2=pylab.twiny()
	ax2.set_xlim(8,20)	# set this to match the lower X axis
	ax2.set_xticks([8.477,9.477,10.477,11.477,12.477,13.477,14.477,15.4768,16.383,17.383,18.383,19.383])
	ax2.set_xticklabels(['1m','10cm','1cm','1mm','100$\mu$m','10$\mu$m','1$\mu$m','1000$\AA$','.1keV','1keV','10keV','100keV'],size=10.5)
	pylab.minorticks_on()

	pylab.draw()
	#pylab.show()




def groupmodel(sum=None,ssd=None,jet=None,adaf=None):
	"""
Groups the different components of a SED model into one dictionary for convenient
access. The model component SEDs are assumed to have been imported using the
SED class. This method is used in the plot method.

Example:

Imports the model SEDs:
>>> s=SED(file='sum.dat', logfmt=1)
>>> disk=SED(file='disk.dat', logfmt=1)
>>> a=SED(file='adaf.dat', logfmt=1)

Groups them together in a dictionary d:
>>> d=groupmodel(sum=s,ssd=disk,adaf=a)
returning {'sum':s,'ssd':disk,'adaf':a}. 
	"""
	m={}	# empty dictionary
	
	if sum!=None: m['sum']=sum
	if ssd!=None: m['ssd']=ssd
	if jet!=None: m['jet']=jet
	if adaf!=None: m['adaf']=adaf
	
	return m




def obsplot1097(source,o,table):
	"""
Modification of the plot method to handle the nice data of NGC 1097.

:param source: source name e.g. 'NGC1097' or 'ngc1097' or 'NGC 1097'. Must have four numbers.
:param o: observed SED data points imported using the SED class and the erac method.
:param table: path of ASCII table models.csv with the required SED parameters gammax, error in gammax, X-ray luminosity and distratio.
:returns: SED object with distance-corrected luminosities
	"""
	# Reads the ASCII table models.csv
	t=ascii.read(table,Reader=ascii.CommentedHeader,delimiter='\t')
	tname,tmodel,tdistratio,tgammax,tegammal,tegammau,tlumx=t['Object'],t['Model'],t['distratio'],t['gammax'],t['egammal'],t['egammau'],t['lumx']

	# Finds the required parameters to plot the data for the specified source.
	# First converts the source name to uppercase, no spaces
	source=source.upper()
	source=source.replace(' ', '')
	# Finds the first occurrence of the source in the table
	i=numpy.where(tname==source)
	i=i[0][0]
	# Retrieves the required parameters
	distratio=tdistratio[i]
	gammax=tgammax[i]
	egammal=tegammal[i]
	egammau=tegammau[i]
	lumx=tlumx[i]

	# Precaution to avoid modifying unintentionally the obs SED
	#o=o.copy()

	pylab.clf()
	
	# Plots OBSERVED SED
	# ====================
	# Corrects SED data points for new distance
	#o.nlnu=o.nlnu*(distratio)**2
	#o.ll=numpy.log10(o.nlnu)
	#lumx=lumx*(distratio)**2
		
	# Separates the observed SED in the different bands and uses a different recipe
	# to plot each region
	# Radio and IR as filled points
	rir=numpy.where(o.lognu<14.46)
	pylab.plot(o.lognu[rir],o.ll[rir],'ko')
	
	# Optical and UV as solid line
	ouv=numpy.where((o.lognu>=14.46) & (o.lognu<=16.))	
	pylab.plot(o.lognu[ouv],o.ll[ouv],'k')
	
	pylab.plot()
	
	# X-rays as bow tie
	xray(lumx,gammax,egammal,egammau)	
	
	# Upper limits with arrows
	u=numpy.where((o.lognu>=10) & (o.lognu<=14.46))	# index of IR upper limits
	pylab.quiver(o.lognu[u],o.ll[u],0,-1,scale=30,width=3e-3)	# careful with the arrow parameters	

	# Dummy plot just to include the name of the source with legend later
	pylab.plot(o.lognu+50,o.ll,'w',label=source)

	return o

	


def xray(L2_10, gammax, gammaerru, gammaerrl, nurange=[2.,10.], **args):
	"""
Plots X-ray "bow tie", provided the X-ray luminosity in the 2-10 keV range,
the best-fit photon index and the error in this index.

Arguments:
- L2_10: X-ray luminosity in the 2-10 keV range, erg/s
- gammax: best-fit photon index
- gammaerru, gammaerrl: upper and lower 1sigma error bars on the value of
  gammax
 - nurange: two-element list or array with range of frequencies to plot, 
  in keV. e.g. [2,10]
	"""
	# Retrieves current axis limits
	xy=pylab.axis()

	# Creates vectors containing the X-ray powerlaws and plots them
	lognu,nuLbest=powerlaw(gammax-1.,L2_10,nurange)	# best-fit
	lognu,nuLlow=powerlaw(gammax-gammaerrl-1.,L2_10,nurange)	# lower
	lognu,nuLup=powerlaw(gammax+gammaerru-1.,L2_10,nurange)	# upper

	pylab.fill_between(lognu, nuLlow, nuLup, alpha=0.5, facecolor='gray')
	pylab.plot(lognu, nuLbest,'k', linewidth=2.5, **args)
	




def powerlaw(alpha,Lx,nurange,npoints=10):
	""" 
Calculates the X-ray powerlaw variables. 

Arguments:
- alpha: spectral power-law index
- Lx: total luminosity in X-rays in the nurange
 - nurange: two-element list or array with range of frequencies to plot, 
  in keV. e.g. [2,10]

Returns the sequence of arrays log10(nu),log10(nu*Lnu) for plotting.
	"""
	h=0.6626e-26 # Planck constant (CGS)
	conv=1.602171e-9/h # conversion factor keV -> Hz
	nu0=nurange[0]*conv # nu_i from keV -> Hz
		
	# Creates a vector of frequencies
	points,nui,nuf = npoints,nu0,nurange[1]*conv	# e.g. 2 keV - 10 keV
	nu=numpy.linspace(nui,nuf,points)
	
	# Takes into account the possibility of alpha=1 which changes the way you
	# calculate (integrate) the total X-ray luminosity from the power-law
	if round(alpha,2)!=1. :
		L0=nu0**(-alpha)*Lx*(1.-alpha)/(nuf**(1.-alpha)-nui**(1.-alpha))
	else:
		L0=nu0**(-alpha)*Lx/numpy.log(nuf/nui)
		
	Lnu=L0*(nu/nu0)**(-alpha)

	return numpy.log10(nu), numpy.log10(nu*Lnu)





def xrayinset(source,ssdf,adaff,jetf,sumf,
	table='/Users/nemmen/Dropbox/work/projects/finished/liners/tables/models.csv',
	**args
	):
	"""
Plots an inset with the zoomed-in X-ray spectra.
	"""
	# This piece of code was quickly and dirty copied from the obsplot method,
	# in order to plot the X-ray bowtie
	# ========================= BEGIN

	# Reads the ASCII table models.csv
	t=ascii.read(table,Reader=ascii.CommentedHeader,delimiter='\t')
	tname,tmodel,tdistratio,tgammax,tegammal,tegammau,tlumx=t['Object'],t['Model'],t['distratio'],t['gammax'],t['egammal'],t['egammau'],t['lumx']

	# Finds the required parameters to plot the data for the specified source.
	# First converts the source name to uppercase, no spaces
	source=source.upper()
	source=source.replace(' ', '')
	# Finds the first occurrence of the source in the table
	i=numpy.where(tname==source)
	i=i[0][0]
	# Retrieves the required parameters
	distratio=tdistratio[i]
	gammax=tgammax[i]
	egammal=tegammal[i]
	egammau=tegammau[i]
	lumx=tlumx[i]

	# Corrects Lx for new distance
	lumx=lumx*(distratio)**2
	# ========================= END

	# Gets central Lx
	lognu,y=powerlaw(gammax-egammal-1.,lumx)

	# Creates the inset
	if numpy.median(y)<38.:
		# precaution for NGC 3379, which has a very low Lx and hence the inset hides the main plot
		pylab.axes([0.7,0.6,0.17,0.17])
	else:
		pylab.axes([0.7,0.15,0.17,0.17])
	pylab.xlim(17.65,18.4)
	pylab.ylim(y.min()-0.1,y.max()+0.1)
	pylab.xticks([])
	pylab.yticks([])

	## X-rays as bow tie
	xray(lumx,gammax,egammal,egammau,**args)


	# Now the piece of code below was copied from the modelplot method.
	# Plots the model components.
	# ========================= BEGIN


	# Reads model SEDs, checking first if the data files exist
	if os.path.isfile(sumf): msum=SED(file=sumf,logfmt=1) 
	else: msum=None
	if os.path.isfile(adaff): adaf=SED(file=adaff,logfmt=1) 
	else: adaf=None
	if os.path.isfile(jetf): jet=SED(file=jetf,logfmt=1) 
	else: jet=None
	if os.path.isfile(ssdf): ssd=SED(file=ssdf,logfmt=1) 
	else: ssd=None
	
	# Groups the model components in a dictionary
	m=groupmodel(sum=msum,ssd=ssd,adaf=adaf,jet=jet)

	if 'sum' in m: pylab.plot(m['sum'].lognu,m['sum'].ll,'k')
	if 'adaf' in m: pylab.plot(m['adaf'].lognu,m['adaf'].ll,'--')
	if 'jet' in m: pylab.plot(m['jet'].lognu,m['jet'].ll,'-.')
	if 'ssd' in m: pylab.plot(m['ssd'].lognu,m['ssd'].ll,':')
	# ========================= END







def popstar(o):
	"""
Fits the stellar population spectrum, finds the required mass and plots its
SED.

:param o: object containing the observed SED
:returns: fitted stellar population mass
	"""
	# Gets the optical-UV points (SELECT THE FREQUENCY RANGE)
	ouv=numpy.where((o.lognu>=14.46) & (o.lognu<=14.87) & (o.ul==0))
	#pylab.plot(o.lognu[ouv],o.ll[ouv],'ro')

	# Reads stellar population sed
	#xx,yy = numpy.loadtxt('/Users/nemmen/Dropbox/work/projects/finished/liners/seds/popstar/bruzual/bc2003_hr_m62_salp_ssp_181.spec',unpack=True,usecols=(0,1))
	xx,yy = numpy.loadtxt('/Users/nemmen/Dropbox/work/projects/finished/liners/seds/popstar/bruzual/Sed_Mar05_Z_0.02_Age_10.00000000.dat',unpack=True,usecols=(0,1))
	xpop=numpy.log10(29979245800./(xx*1e-8)) # AA to cm to Hz
	ypop=numpy.log10(xx*yy*3.826e33)	# Lsun/AA -> erg/s

	# Fits popstar to OUV data
	# only fits if number of OUV points > 0
	ff=lambda x,a: f(xpop,ypop,x)+a # function definition
	if numpy.size(ouv)>0:
		fit,cov = scipy.optimize.curve_fit(ff, o.lognu[ouv], o.ll[ouv], p0=8.)
		m=fit[0]
		print('popstar mass = '+str(m))

		pylab.plot(xpop,ypop+m,'-',alpha=0.7,color='gray')

		return m






def f(xpop,ypop,x):
    """
Given the spectrum of a popstar and an array of frequencies 'x',
this outputs the values of ypop for each x.

:param x: frequencies
:param xpop,ypop: spectrum from stellar population
:returns: returns y=a*L where L is the stellar pop. luminosity
    """
    i=lsd.search(x,xpop)
    
    return ypop[i]





def pionDecay(logfile, parfile, alpha=2.3, E0=None, proton_energy=None, photon_energy=None):
	"""
Gamma-ray spectrum from a RIAF, pion 0 decay
=============================================
Computes the gamma-ray spectrum from pp collisions followed by pion decay for the full 
RIAF computed by my semi-analytical model. 

:param logfile: log file generated by the semi-analytical RIAF model, containing the electron number density and ion temperature. e.g. 'cd ~/science/projects/adafjet/adaf/perl/run01/out_01'
:param parfile: parameter file used as input by the semi-analytical RIAF model, containing the black hole mass and other model parameters. e.g. 'cd ~/science/projects/adafjet/adaf/perl/run01/in.dat'
:param alpha: power-law index
:param E0: reference energy. Default = proton_mass*c^2 ~ 1 GeV
:param proton_energy: array of proton energies, e.g. logspace(0, 6, 50) * astropy.units.MeV
:param photon_energy: array of photon energies for SED. If None, taken as the same as proton_energy. In astropy.units

:returns: SED object in units of Hz vs erg/s

Example:

Computes gamma-ray SED with default particle DF options and power-law index. Issue
the command below in the directory with model datafiles.

.. code-block:: python

    m=nmmn.sed.pionDecay('out_01','in.dat')
    plot(m.lognu,m.ll)


Based on the jupyter notebook "gamma-rays RIAF". 

References about pion decay physics:
• Oka & Manmoto 2003: overview of calculation
• Fatuzzo & Melia 2003: overview of calculation
• Dermer 1986: formulas

naima model:
• Kafexhiu, E., Aharonian, F., Taylor, A.M., & Vila, G.S. 2014, Physical Review D, 90, 123014
• Zabalza 2015
	"""
	import naima
	from naima.models import (ExponentialCutoffPowerLaw, Synchrotron,
							  InverseCompton)
	import astropy.units as u
	from . import misc # intrapackage reference
	import tqdm, re


	# Preliminaries
	# ==============

	# constants in CGS (everything in CGS unless otherwise noted)
	k = 1.38065e-16 # Boltzmann
	mp=1.67e-24 # proton mass
	c = 29979245800
	mev = 1.6e-6 # 1 MeV in erg
	G = 6.673e-8 
	solarmass = 1.99e33
	planck=6.62607e-27	# Planck constant 

	# Input from RIAF model
	# ======================

	# ## reads files
	#
	# The RIAF model gives us the following variables:
	# 
	# - electron number density $n_e(R)$ where $R$ is the cylindrical radius 
	# - proton temperature $T_p(R)$
	# - scale height $H(R)$
	r,ti,h,ne = numpy.loadtxt(logfile,unpack=True,usecols=(0,3,7,14))

	# proton temperature
	Tp=10**ti
	# electron number density
	ne=10**ne
	# number of cylindrical shells in RIAF
	nshells=r.size

	# ## gets BH mass from parameter file
	f = open(parfile,"r")

	for line in f:
		# if m=7.9d0 in the file
		if re.search(r'm=\d+\.d', line):
			m=re.search(r'\d+\.\d+d0', line).group()
			m=float(re.sub(r"d", "e", m)) # replaces d with e in the number for use in python
		else: 
			# if m=7.9
			if re.search(r'm=\d+\.', line):
				m=re.search(r'\d+\.\d+', line).group()
				m=float(m) # replaces d with e in the number for use in python

	f.close()

	m=m*1e6 # in solar masses

	# The quantity below is useful for converting from geometrized to physical units. Example: to convert radius `r` from geometrized units to physical units: 
	# 
	#     rphys=length*r
	length=G*m*solarmass/c**2

	# ## compute shells
	# 
	# We need to convert the RIAF variables to:
	# 
	# - proton number density $n_p$: assume $n_p(R)=n_e(R)$
	# - proton dimensionless temperature $\theta_p$
	# - volume of each shell $\Delta V (R)=4\pi R H \Delta R$ 
	# 
	# Everything in CGS unless otherwise noted.

	# thickness of each shell
	dr=numpy.zeros(nshells-1)

	for i in range(nshells-1):
		dr[i]=r[i]-r[i+1]

	# repeats last element to ensure that r and dr have same number of elements and avoid headaches
	dr=numpy.append(dr,dr[-1])

	# distances in physical units
	rphys=r*length	# cylindrical radius
	drphys=dr*length # thickness of each shell
	hphys=h*length	# scale height

	# ## gets RIAF proton parameters
	np=ne # proton number density
	thetap=k*Tp/(mp*c**2)	# k Tp/(mp c^2)
	#kTp=k*Tp # k Tp
	dV=4*numpy.pi*rphys*hphys*drphys 	# volume of shells

	# Spectrum
	# =============
	# 
	# ## particle distribution function
	# 
	# Spectral parameters for the [Power Law model](https://naima.readthedocs.io/en/latest/api-models.html#naima.models.PowerLaw) 
	# this is a list of amplitudes
	amplitude = 1.5*(alpha-1)*(alpha-2)*np*thetap*dV/(mp*c**2) / u.erg
	#e_cutoff = 80 * u.TeV
	if E0 is None:
		E0=mp*c**2 * u.erg

	if proton_energy is None:
		proton_energy = numpy.logspace(0, 6, 50) * u.MeV

	# proton energy DF for each shell
	PL=[]

	for i in range(nshells):
		PL.append(naima.models.PowerLaw(amplitude[i], E0, alpha))

	# ## pion decay
	#
	# computes pion decay model at each shell
	pion=[]

	for i in range(nshells):
		pion.append( naima.models.PionDecay(PL[i], nh=np[i] * u.cm ** -3) )

	# ## SED
	if photon_energy is None:
		photon_energy = proton_energy

	# compute the radiative contribution of each shell
	seds=[]

	for i in tqdm.tqdm(range(nshells), "Computing gamma-ray spectrum"):
		seds.append( pion[i].sed(photon_energy, distance=0) )

	# sums up all shells in "The SED"
	for i in range(nshells):
		if i==0: 
			thesed=seds[0]
		else:
			thesed=thesed+seds[i]

	# frequency in Hz
	nu=photon_energy.to(u.erg).value/planck
	# luminosity in erg/s
	nuLnu=thesed.value

	# returns SED object 
	return SED(lognu=numpy.log10(nu), ll=numpy.log10(nuLnu), logfmt=1)



