"""
Spectral energy distributions
===============================

Methods and classes for dealing with spectral energy distributions (SEDs).
"""

import numpy
from . import lsd # intrapackage reference

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
		if file!=None:
			if logfmt==0:
				# Reads SED from datafile
				self.nu,self.nlnu = numpy.loadtxt(file,unpack=True,usecols=(0,1))
		
				# Take logs
				self.lognu, self.ll = numpy.log10(self.nu), numpy.log10(self.nlnu)
			else:
				self.lognu, self.ll = numpy.loadtxt(file,unpack=True,usecols=(0,1))
				self.nu,self.nlnu = 10.**self.lognu, 10.**self.ll

		# If the SED is created from the arrays
		if file==None and lognu!=None:
			if logfmt==1:
				self.lognu, self.ll = lognu, ll
				self.nu,self.nlnu = 10.**self.lognu, 10.**self.ll
			else:
				self.nu, self.nlnu = lognu, ll
				self.lognu,self.ll = numpy.log10(self.lognu), numpy.log10(self.ll)
		
		# Checks if ll has NaN or Inf values
		if file!=None or lognu!=None:
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
		import astropy.io.ascii as ascii

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
	import astropy.io.ascii as ascii
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

