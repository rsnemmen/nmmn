"""
Dealing with (GR)(R)(M)HD simulations
========================================

- RAISHIN
- Pluto
- HARM

See jupyter notebooks "grmhd*" for examples on how to use this
module.

TODO:

- [ ] incorporate Pluto class from pluto-tools
- [ ]
"""

import numpy, scipy
import tqdm
import numpy as np




class Raishin:
	"""
Class that reads a RAISHIN VTK datafile and converts to numpy format.

Attributes of the object:

- x,y,z: 1D position arrays for the mesh
- rho: 1D density array
- vx,vy,vz: 1D velocity arrays
- p: 1D pressure
- bx,by,bz: 1D magnetic field arrays
- b2: 1D B^2

Define an empty object:

>>> o=nmmn.grmhd.Raishin()

Reads data from a VTK file, new attributes rho, p, vx, bx etc:

>>> o.vtk("ok200.vtk")

Saves data as an ASCII file with columns corresponding to variables:

>>> o.savetxt("ok200.dat")
	"""

	#def __init__(self):
	# does nothing for now



	def vtk(self, vtkfile):
		"""
	Given a VTK file created with the RAISHIN GRMHD code, this reads the
	data as numpy arrays.
		"""
		import re

		f = open(vtkfile,"r")
		#newf=open("tmp.dat","w")	# file that will hold coordinates

		# booleans that will tell the code when to stop reading
		# data for a given variable:
		# boold for density, boolp for pressure etc
		boolxyz,boold,boolp,boolvx,boolvy,boolvz,boolb2,boolbx,boolby,boolbz=False,False,False,False,False,False,False,False,False,False
		strx,stry,strz,strd,strp,strvx,strvy,strvz,strb2,strbx,strby,strbz='','','','','','','','','','','','' # string that holds values 

		for line in f:
			# gets dimensions
			if re.search(r'DIMENSIONS\s+\d+\s+\d+',line):
				s=re.findall(r'\s*\d+\s*',line.rstrip('\n'))
				self.nx=int(s[0])
				self.ny=int(s[1])
				self.nz=int(s[2])
				boolxyz=True
		    
			# gets arrays
			# these lines are important to tell python when to stop reading shit
			# it must be sequential
			if 'density' in line: 
				boolxyz=False
				boold=True
			if 'pressure' in line: 
				boold=False
				boolp=True
			if 'LorentzW1' in line:
				boolp=False
			if 'util^x' in line:
				boolvx=True
			if 'util^y' in line:
				boolvx=False
				boolvy=True    
			if 'util^z' in line:
				boolvy=False
				boolvz=True  
			if 'b^2' in line:
				boolvz=False
				boolb2=True    		          
			if 'bx' in line:
				boolb2=False
				boolbx=True
			if 'by' in line:
				boolbx=False
				boolby=True    
			if 'bz' in line: 
				boolby=False    
				boolbz=True
		    
			if boolxyz==True and re.search(r'-?\d+\.\d+E?[-+]?\d+\s+-?\d+\.\d+E?[-+]?\d+\s+-?\d+\.\d+E?[-+]?\d+',line):
				s=re.findall(r'\s*-?\d+\.\d+E?[-+]?\d+\s*',line.rstrip('\n'))
				strx=strx+s[0]
				stry=stry+s[1]
				strz=strz+s[2]
			if boold==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strd=strd+line
			if boolp==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strp=strp+line
			if boolvx==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strvx=strvx+line
			if boolvy==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strvy=strvy+line
			if boolvz==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strvz=strvz+line
			if boolb2==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strb2=strb2+line
			if boolbx==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strbx=strbx+line		        
			if boolby==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strby=strby+line
			if boolbz==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
				strbz=strbz+line

		# gets numpy arrays finally
		self.x=numpy.fromstring(strx, sep='\n')
		self.y=numpy.fromstring(stry, sep='\n')
		self.z=numpy.fromstring(strz, sep='\n')
		self.rho=numpy.fromstring(strd, sep='\n')
		self.p=numpy.fromstring(strp, sep='\n')
		self.vx=numpy.fromstring(strvx, sep='\n')
		self.vy=numpy.fromstring(strvy, sep='\n')
		self.vz=numpy.fromstring(strvz, sep='\n')
		self.b2=numpy.fromstring(strb2, sep='\n')
		self.bx=numpy.fromstring(strbx, sep='\n')
		self.by=numpy.fromstring(strby, sep='\n')
		self.bz=numpy.fromstring(strbz, sep='\n')

		# reads mesh positions 
		#self.x,self.y,self.z= numpy.loadtxt('tmp.dat',unpack=True,usecols=(0,1,2))

		# close files        
		f.close()
		#newf.close()




	def savetxt(self,outfile):
		"""Saves data as ASCII file """
		numpy.savetxt(outfile,numpy.transpose((self.x,self.y,self.z,self.rho,self.p,self.vx,self.vy,self.vz,self.bx,self.by,self.bz)))


	def savehdf5(self,outfile):
		"""
	Exports data as compressed HDF5. 7x less space than ASCII.
		"""
		import h5py

		with h5py.File(outfile, 'w') as hf:
			grid=hf.create_group('grid')
			grid.create_dataset('x', data=self.x, compression="gzip", compression_opts=9)
			grid.create_dataset('y', data=self.y, compression="gzip", compression_opts=9)
			grid.create_dataset('z', data=self.z, compression="gzip", compression_opts=9)    

			fields=hf.create_group('fields')
			fields.create_dataset('density', data=self.rho, compression="gzip", compression_opts=9)
			fields.create_dataset('pressure', data=self.p, compression="gzip", compression_opts=9)
			fields.create_dataset('vx', data=self.vx, compression="gzip", compression_opts=9)
			fields.create_dataset('vy', data=self.vy, compression="gzip", compression_opts=9)
			fields.create_dataset('vz', data=self.vz, compression="gzip", compression_opts=9)
			fields.create_dataset('bx', data=self.bx, compression="gzip", compression_opts=9)
			fields.create_dataset('by', data=self.by, compression="gzip", compression_opts=9)
			fields.create_dataset('bz', data=self.bz, compression="gzip", compression_opts=9)


	def savenumpy(self,outfile):
		"""
	Save data as binary Numpy file .npz. 3x less space than ASCII.
		"""
		numpy.savez(outfile,x=self.x,y=self.y,z=self.z,rho=self.rho,p=self.p,vx=self.vx,vy=self.vy,vz=self.vz,bx=self.bx,by=self.by,bz=self.bz)



	def regridAll(self,nboost=5):
		"""
	Regrid all RAISHIN data to a nice cartesian grid for plotting with
	python.

	:param nboost: factor of increase of number of grid points compared to 
		previous grid

	Usage:

	>>> d=nmmn.grmhd.Raishin()
	>>> d.vtk('ok100.vtk')
	>>> d.regridAll()

	Gets interpolated rho:

	>>> print(d.xc)

	TODO:
	- 3D version
	- parallel version
		"""
		#import lsd
		from . import lsd # py3

		# create two new arrays with spatial grid, with more points than the 
		# original grid
		nxnew=self.nx*nboost
		nynew=self.ny*nboost
		nznew=self.nz*nboost
		xnew=numpy.linspace(self.x.min(),round(self.x.max()),nxnew)
		ynew=numpy.linspace(self.y.min(),round(self.y.max()),nynew)
		znew=numpy.linspace(self.z.min(),round(self.z.max()),nznew)

		# 'c' is added to 2D array values
		self.xc,self.yc=numpy.meshgrid(xnew,ynew) # 2D
		self.xc1d,self.yc1d,self.zc1d=xnew,ynew,znew # 1D

		# bottleneck,
		self.rhoc=lsd.regrid(self.x,self.y,self.rho,xnew,ynew)
		self.pc=lsd.regrid(self.x,self.y,self.p,xnew,ynew)
		self.vxc=lsd.regrid(self.x,self.y,self.vx,xnew,ynew)
		self.vyc=lsd.regrid(self.x,self.y,self.vy,xnew,ynew)
		self.vzc=lsd.regrid(self.x,self.y,self.vz,xnew,ynew)
		self.bxc=lsd.regrid(self.x,self.y,self.bx,xnew,ynew)
		self.byc=lsd.regrid(self.x,self.y,self.by,xnew,ynew)
		self.bzc=lsd.regrid(self.x,self.y,self.bz,xnew,ynew)

		self.bc=numpy.sqrt(self.bxc**2+self.byc**2)
		self.vc=numpy.sqrt(self.vxc**2+self.vyc**2)
	


	def regrid(self,var,nboost=5):
		"""
	Regrid one specific RAISHIN array to a nice cartesian grid for 
	plotting with python. Note that RAISHIN's output is already in
	cartesian coordinates.

	:param var: array to be regridded e.g. d.rho
	:param nboost: factor of increase of number of grid points compared to 
		previous grid

	Usage:

	>>> d=nmmn.grmhd.Raishin()
	>>> d.vtk('ok100.vtk')
	>>> d.regrid(d.rho)

	TODO:
	- 3D version
	- parallel version
		"""
		#import lsd
		from . import lsd

		# create two new arrays with spatial grid, with more points than the 
		# original grid
		nxnew=self.nx*nboost
		nynew=self.ny*nboost
		xnew=numpy.linspace(self.x.min(),round(self.x.max()),nxnew)
		ynew=numpy.linspace(self.y.min(),round(self.y.max()),nynew)

		# 'c' is added to 2D array values
		self.xc,self.yc=numpy.meshgrid(xnew,ynew) # 2D
		self.xc1d,self.yc1d=xnew,ynew # 1D

		# bottleneck,
		return lsd.regrid(self.x,self.y,var,xnew,ynew)


	def regridsome(self,listarr,nboost=5):
		"""
	Regrid the selected arrays in the RAISHIN data to a nice cartesian 
	grid for plotting with python. Regridding only some of the arrays
	will, of course, speed up things.

	:param listarr: list of strings specifying the arrays to be regridded.
	  Options are: rho, p, v, b
	:param nboost: factor of increase of number of grid points compared to 
		previous grid

	Usage:

	>>> d=nmmn.grmhd.Raishin()
	>>> d.vtk('ok100.vtk')
	>>> d.regridsome(['rho','v'])

	TODO:
	- 3D version
	- parallel version
		"""
		#import lsd
		from . import lsd # py3

		# create two new arrays with spatial grid, with more points than the 
		# original grid
		nxnew=self.nx*nboost
		nynew=self.ny*nboost
		xnew=numpy.linspace(self.x.min(),round(self.x.max()),nxnew)
		ynew=numpy.linspace(self.y.min(),round(self.y.max()),nynew)

		# 'c' is added to 2D array values
		self.xc,self.yc=numpy.meshgrid(xnew,ynew) # 2D
		self.xc1d,self.yc1d=xnew,ynew # 1D

		# bottleneck
		if 'rho' in listarr:
			self.rhoc=lsd.regrid(self.x,self.y,self.rho,xnew,ynew)
		if 'p' in listarr: 
			self.pc=lsd.regrid(self.x,self.y,self.p,xnew,ynew)
		if 'v' in listarr:
			self.vxc=lsd.regrid(self.x,self.y,self.vx,xnew,ynew)
			self.vyc=lsd.regrid(self.x,self.y,self.vy,xnew,ynew)
			self.vzc=lsd.regrid(self.x,self.y,self.vz,xnew,ynew)
			self.vc=numpy.sqrt(self.vxc**2+self.vyc**2)
		if 'b' in listarr:
			self.bxc=lsd.regrid(self.x,self.y,self.bx,xnew,ynew)
			self.byc=lsd.regrid(self.x,self.y,self.by,xnew,ynew)
			self.bzc=lsd.regrid(self.x,self.y,self.bz,xnew,ynew)
			self.bc=numpy.sqrt(self.bxc**2+self.byc**2)




	def yt2d(self):
		"""
	Converts 2d arrays from raishin to the 3d format that is understood
	by the yt package. Make sure you used regridAll first.

	Inspired by this example: http://stackoverflow.com/questions/7372316/how-to-make-a-2d-numpy-array-a-3d-array
		"""
		self.x3d=self.xc.T[..., numpy.newaxis]
		self.y3d=self.yc.T[..., numpy.newaxis]

		self.rho3d=self.rhoc.T[..., numpy.newaxis]
		self.p3d=self.pc.T[..., numpy.newaxis]

		self.vx3d=self.vxc.T[..., numpy.newaxis]
		self.vy3d=self.vyc.T[..., numpy.newaxis]
		self.vz3d=self.vzc.T[..., numpy.newaxis]
		self.v3d=self.vc.T[..., numpy.newaxis]

		self.bx3d=self.bxc.T[..., numpy.newaxis]
		self.by3d=self.byc.T[..., numpy.newaxis]
		self.bz3d=self.bzc.T[..., numpy.newaxis]
		self.b3d=self.bc.T[..., numpy.newaxis]










class Harm:
	"""
Class that reads a HARM dump datafile and converts to numpy format
for plotting with matplotlib, mayavi etc. Heavily inspired/copied from
harm_script.py.

Reads data from a harm dump file:

>>> d=nmmn.grmhd.Harm("dump019")

The `d` object will then have several attributes related to the 
dump physical fields, grid and coordinate information.

Inspect densities:

>>> d.rho

Convert arrays to cartesian coordinates for plotting with matplotlib:

>>> rhonew,x,y=o.cartesian(o.rho)
>>> pcolormesh(x,y,log10(rhonew))
	"""

	def __init__(self, dump=None, gdump='gdump'):
		"""
		TODO:
		- [ ] input number of snapshot instead of filename
		"""
		self.dump=dump
		self.gdump=gdump

		if dump is not None:
			# read grid information
			self.read_file(gdump,type="gdump")

			# read dump file
			self.read_file(dump,type="dump")
		else:
			print("Please provide a dump file.")



	def read_file(self,dump,type=None,savedump=True,saverdump=False,noround=False):
		"""
	High-level function that reads either MPI or serial gdump's
		"""
		import os,sys
		import glob

		if type is None:
			if dump.startswith("dump"):
				type = "dump"
				print("Reading a dump file %s ..." % dump)
			elif dump.startswith("gdump2"):
				type = "gdump2"
				print("Reading a gdump2 file %s ..." % dump)
			elif dump.startswith("gdump"):
				type = "gdump"
				print("Reading a gdump file %s ..." % dump)
			elif dump.startswith("rdump"):
				type = "rdump"
				print("Reading a rdump file %s ..." % dump)
			elif dump.startswith("fdump"):
				type = "fdump"
				print("Reading a fdump file %s ..." % dump)
			else:
				print("Couldn't guess dump type; assuming it is a data dump")
				type = "dump"

	    #normal dump
		if os.path.isfile(dump):
			headerline = self.read_header(dump, returnheaderline = True)
			gd = self.read_body(dump,nx=self.N1+2*self.N1G,ny=self.N2+2*self.N2G,nz=self.N3+2*self.N3G,noround=1)
			if noround:
				res = self.data_assign(         gd,type=type,nx=self.N1+2*self.N1G,ny=self.N2+2*self.N2G,nz=self.N3+2*self.N3G)
			else:
				res = self.data_assign(myfloat(gd),type=type,nx=self.N1+2*self.N1G,ny=self.N2+2*self.N2G,nz=self.N3+2*self.N3G)
			return res

	    #MPI-type dump that is spread over many files
		else:
			flist = np.sort(glob.glob(dump + "_[0-9][0-9][0-9][0-9]" ))
			if len(flist) == 0:
				print( "Could not find %s or its MPI counterpart" % dump )
				return
			sys.stdout.write( "Reading %s (%d files)" % (dump, len(flist)) )
			sys.stdout.flush()
			ndots = 10
			dndot = len(flist)/ndots
			if dndot == 0: dndot = 1
			for i,fname in enumerate(flist):
				#print( "Reading file %d out of %d..." % (i,len(flist)) )
				#header for each file might be different, so read each
				header = read_header(fname,issilent=1)
				if header is None:
					print( "Error reading header of %s, aborting..." % fname )
					return
				lgd = read_body(fname,nx=N1+2*N1G,ny=N2+2*N2G,nz=N3+2*N3G)
				#this gives an array of dimensions (-1,N1,N2,N3)+potentially ghost cells
				if 0 == i:
				#create full array: of dimensions, (-1,nx,ny,nz)
					fgd = np.zeros( (lgd.shape[0], nx+2*N1G, ny+2*N2G, nz+2*N3G), dtype=np.float32)
				if not type == "rdump":
					#construct full indices: ti, tj, tk
					#fti,ftj,ftk = mgrid[0:nx,0:ny,0:nz]
					lti,ltj,ltk = lgd[0:3,:,:].view();
					lti = np.int64(lti)
					ltj = np.int64(ltj)
					ltk = np.int64(ltk)
					fgd[:,lti+N1G,ltj+N2G,ltk+N3G] = lgd[:,:,:,:]
				else:
					print(starti,startj,startk)
					fgd[:,starti:starti+N1+2*N1G,startj:startj+N2+2*N2G,startk:startk+N3+2*N3G] = lgd[:,:,:,:]
				del lgd
				if i%dndot == 0:
					sys.stdout.write(".")
					sys.stdout.flush()
			res = data_assign(fgd,type=type,nx=nx+2*N1G,ny=ny+2*N2G,nz=nz+2*N3G)
			if savedump:
				#if the full dump file does not exist, create it
				dumpfullname = dump
				if (type == "dump" or type == "gdump") and not os.path.isfile(dumpfullname):
					sys.stdout.write("Saving full dump to %s..." % dumpfullname)
					sys.stdout.flush()
					header[1] = header[4] #N1 = nx
					header[2] = header[5] #N2 = ny
					header[3] = header[6] #N3 = nz
					fout = open( dumpfullname, "wb" )
					#join header items with " " (space) as a glue
					#see http://stackoverflow.com/questions/12377473/python-write-versus-writelines-and-concatenated-strings
					#write it out with a new line char at the end
					fout.write(" ".join(header) + "\n")
					fout.flush()
					os.fsync(fout.fileno())
					#reshape the dump content
					gd1 = fgd.transpose(1,2,3,0)
					gd1.tofile(fout)
					fout.close()
					print( " done!" )
					if res is not None:
						return res
			return res

	def read_header(self,dump,issilent=True,returnheaderline=False):
		"""Read the header for the dump file"""
		# I am replacing all global variables below as attributes
		# of the object
		"""
		global t,nx,ny,nz,N1,N2,N3,N1G,N2G,N3G,starti,startj,startk,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,gn3,gv3,guu,gdd,dxdxp, games, startx1, startx2, startx3, tf, NPR, DOKTOT, BL
		global fractheta
		global fracphi
		global rbr
		global npow2
		global cpow2
		"""

		#read image
		fin = open( dump, "rb" )
		headerline = fin.readline()
		header = headerline.split()
		nheadertot = len(header)
		fin.close()
		# Creates object attributes
		if not dump.startswith("dumps/rdump"):
			if not issilent: print( "dump header: len(header) = %d" % len(header) )
			nheader = 57
			n = 0
			self.t = myfloat(np.float64(header[n])); n+=1
			#per tile resolution
			self.N1 = int(header[n]); n+=1
			self.N2 = int(header[n]); n+=1
			self.N3 = int(header[n]); n+=1
			#total resolution
			self.nx = int(header[n]); n+=1
			self.ny = int(header[n]); n+=1
			self.nz = int(header[n]); n+=1
			#numbers of ghost cells
			self.N1G = int(header[n]); n+=1
			self.N2G = int(header[n]); n+=1
			self.N3G = int(header[n]); n+=1
			self.startx1 = myfloat(float(header[n])); n+=1
			self.startx2 = myfloat(float(header[n])); n+=1
			self.startx3 = myfloat(float(header[n])); n+=1
			self._dx1=myfloat(float(header[n])); n+=1
			self._dx2=myfloat(float(header[n])); n+=1
			self._dx3=myfloat(float(header[n])); n+=1
			self.tf=myfloat(float(header[n])); n+=1
			self.nstep=myfloat(float(header[n])); n+=1
			self.a=myfloat(float(header[n])); n+=1
			self.gam=myfloat(float(header[n])); n+=1
			self.cour=myfloat(float(header[n])); n+=1
			self.DTd=myfloat(float(header[n])); n+=1
			self.DTl=myfloat(float(header[n])); n+=1
			self.DTi=myfloat(float(header[n])); n+=1
			self.DTr=myfloat(float(header[n])); n+=1
			self.DTr01=myfloat(float(header[n])); n+=1
			self.dump_cnt=myfloat(float(header[n])); n+=1
			self.image_cnt=myfloat(float(header[n])); n+=1
			self.rdump_cnt=myfloat(float(header[n])); n+=1
			self.rdump01_cnt=myfloat(float(header[n])); n+=1
			self.dt=myfloat(float(header[n])); n+=1
			self.lim=myfloat(float(header[n])); n+=1
			self.failed=myfloat(float(header[n])); n+=1
			self.Rin=myfloat(float(header[n])); n+=1
			self.Rout=myfloat(float(header[n])); n+=1
			self.hslope=myfloat(float(header[n])); n+=1
			self.R0=myfloat(float(header[n])); n+=1
			self.NPR=int(header[n]); n+=1
			self.DOKTOT=int(header[n]); n+=1
			self.fractheta = myfloat(header[n]); n+=1
			self.fracphi   = myfloat(header[n]); n+=1
			self.rbr       = myfloat(header[n]); n+=1
			self.npow2     = myfloat(header[n]); n+=1
			self.cpow2     = myfloat(header[n]); n+=1
			self.BL = myfloat(header[n]); n+=1
		else:
			print("rdump header")
			nheader = 46
			n = 0
			#per tile resolution
			self.N1 = int(header[n]); n+=1
			self.N2 = int(header[n]); n+=1
			self.N3 = int(header[n]); n+=1
			#total resolution
			self.nx = int(header[n]); n+=1
			self.ny = int(header[n]); n+=1
			self.nz = int(header[n]); n+=1
			#numbers of ghost cells
			self.N1G = int(header[n]); n+=1
			self.N2G = int(header[n]); n+=1
			self.N3G = int(header[n]); n+=1
			#starting indices
			self.starti = int(header[n]); n+=1
			self.startj = int(header[n]); n+=1
			self.startk = int(header[n]); n+=1
			self.t = myfloat(header[n]); n+=1
			self.tf = myfloat(header[n]); n+=1
			self.nstep = int(header[n]); n+=1
			self.a = myfloat(header[n]); n+=1
			self.gam = myfloat(header[n]); n+=1
			self.game = myfloat(header[n]); n+=1
			self.game4 = myfloat(header[n]); n+=1
			self.game5 = myfloat(header[n]); n+=1
			self.cour = myfloat(header[n]); n+=1
			self.DTd = myfloat(header[n]); n+=1
			self.DTl = myfloat(header[n]); n+=1
			self.DTi = myfloat(header[n]); n+=1
			self.DTr = myfloat(header[n]); n+=1
			self.DTr01 = myfloat(header[n]); n+=1
			self.dump_cnt = myfloat(header[n]); n+=1
			self.image_cnt = myfloat(header[n]); n+=1
			self.rdump_cnt = myfloat(header[n]); n+=1
			self.rdump01_cnt=myfloat(float(header[n])); n+=1
			self.dt = myfloat(header[n]); n+=1
			self.lim = myfloat(header[n]); n+=1
			self.failed = myfloat(header[n]); n+=1
			self.Rin = myfloat(header[n]); n+=1
			self.Rout = myfloat(header[n]); n+=1
			self.hslope = myfloat(header[n]); n+=1
			self.R0 = myfloat(header[n]); n+=1
			self.fractheta = myfloat(header[n]); n+=1
			self.fracphi = myfloat(header[n]); n+=1
			self.rbr = myfloat(header[n]); n+=1
			self.npow2 = myfloat(header[n]); n+=1
			self.cpow2 = myfloat(header[n]); n+=1
			self.tdump = myfloat(header[n]); n+=1
			self.trdump = myfloat(header[n]); n+=1
			self.timage = myfloat(header[n]); n+=1
			self.tlog  = myfloat(header[n]); n+=1

		if n != nheader or n != nheadertot:
			print("Wrong number of elements in header: nread = %d, nexpected = %d, nototal = %d: incorrect format?"% (n, nheader, nheadertot) )
			return headerline
		if returnheaderline:
			return headerline
		else:
			return header
	            
	def read_body(self,dump,nx=None,ny=None,nz=None,noround=False):
		fin = open( dump, "rb" )
		header = fin.readline()
		if dump.startswith("dumps/rdump"):
			dtype = np.float64
			body = np.fromfile(fin,dtype=dtype,count=-1)
			gd = body.view().reshape((self.nx,self.ny,self.nz,-1), order='C')
			if noround:
				gd=gd.transpose(3,0,1,2)
			else:
				gd=myfloat(gd.transpose(3,0,1,2))
		elif dump.startswith("dumps/gdump2"):
			dtype = np.float64
			body = np.fromfile(fin,dtype=dtype,count=-1)
			gd = body.view().reshape((self.nx,self.ny,self.nz,-1), order='C')
			if noround:
				gd=gd.transpose(3,0,1,2)
			else:
				gd=myfloat(gd.transpose(3,0,1,2))
		elif dump.startswith("dumps/fdump"):
			dtype = np.int64
			body = np.fromfile(fin,dtype=dtype,count=-1)
			gd = body.view().reshape((-1,self.nz,self.ny,self.nx), order='F')
			gd=myfloat(gd.transpose(0,3,2,1))
		else:
			dtype = np.float32
			body = np.fromfile(fin,dtype=dtype,count=-1)
			gd = body.view().reshape((-1,self.nz,self.ny,self.nx), order='F')
			gd=myfloat(gd.transpose(0,3,2,1))
		return gd

	def data_assign(self,gd,type=None,**kwargs):
		if type is None:
			print("Please specify data type")
			return
		if type == "gdump":
			self.gdump_assign(gd,**kwargs)
			return None
		elif type == "gdump2":
			self.gdump2_assign(gd,**kwargs)
			return None
		elif type == "dump":
			self.dump_assign(gd,**kwargs)
			return None
		elif type == "rdump":
			gd = self.rdump_assign(gd,**kwargs)
			return gd
		elif type == "fdump":
			gd = self.fdump_assign(gd,**kwargs)
			return gd
		else:
			print("Unknown data type: %s" % type)
			return gd
	    
	def gdump_assign(self,gd,**kwargs):
		#global t,nx,ny,nz,N1,N2,N3,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,gn3,gv3,guu,gdd,dxdxp, games
		self.nx = kwargs.pop("nx",self.nx)
		self.ny = kwargs.pop("ny",self.ny)
		self.nz = kwargs.pop("nz",self.nz)
		self.ti,self.tj,self.tk,self.x1,self.x2,self.x3,self.r,self.h,self.ph = gd[0:9,:,:].view();  n = 9
		self.gv3 = gd[n:n+16].view().reshape((4,4,self.nx,self.ny,self.nz),order='F').transpose(1,0,2,3,4); n+=16
		self.gn3 = gd[n:n+16].view().reshape((4,4,self.nx,self.ny,self.nz),order='F').transpose(1,0,2,3,4); n+=16
		self.gcov = self.gv3
		self.gcon = self.gn3
		self.guu = self.gn3
		self.gdd = self.gv3
		self.gdet = gd[n]; n+=1
		self.drdx = gd[n:n+16].view().reshape((4,4,self.nx,self.ny,self.nz),order='F').transpose(1,0,2,3,4); n+=16
		self.dxdxp = self.drdx
		if n != gd.shape[0]:
			print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
			return 1
		return 0

	def gdump2_assign(self,gd,**kwargs):
		#global t,nx,ny,nz,N1,N2,N3,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,gdet,games,rf1,hf1,phf1,rf2,hf2,phf2,rf3,hf3,phf3,rcorn,hcord,phcorn,re1,he1,phe1,re2,he2,phe2,re3,he3,phe3
		self.nx = kwargs.pop("nx",self.nx)
		self.ny = kwargs.pop("ny",self.ny)
		self.nz = kwargs.pop("nz",self.nz)
		self.ti,self.tj,self.tk,self.x1,self.x2,self.x3 = gd[0:6,:,:].view();  n = 6
		self.rf1,self.hf1,self.phf1,self.rf2,self.hf2,self.phf2,self.rf3,self.hf3,self.phf3 = gd[0:9,:,:].view();  n += 9
		self.rcorn,self.hcord,self.phcorn,self.rcent,self.hcent,self.phcen = gd[0:6,:,:].view();  n += 6
		self.re1,self.he1,self.phe1,self.re2,self.he2,self.phe2,self.re3,self.he3,self.phe3 = gd[0:9,:,:].view();  n += 9
		self.gdet = gd[n]; n+=1
		if n != gd.shape[0]:
			print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
			return 1
		return 0

	#read in a dump file
	def dump_assign(self,gd,**kwargs):
		#global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor, ktot, pg
		self.nx = kwargs.pop("nx",self.nx)
		self.ny = kwargs.pop("ny",self.ny)
		self.nz = kwargs.pop("nz",self.nz)
		self.ti,self.tj,self.tk,self.x1,self.x2,self.x3,self.r,self.h,self.ph,self.rho,self.ug = gd[0:11,:,:].view(); n = 11
		self.pg = (self.gam-1)*self.ug
		#lrho=np.log10(self.rho)
		self.vu=np.zeros_like(gd[0:4])
		self.B=np.zeros_like(gd[0:4])
		self.vu[1:4] = gd[n:n+3]; n+=3
		self.B[1:4] = gd[n:n+3]; n+=3
		#if total entropy equation is evolved (on by default)
		if self.DOKTOT == 1:
			self.ktot = gd[n]; n+=1
		self.divb = gd[n]; n+=1
		self.uu = gd[n:n+4]; n+=4
		self.ud = gd[n:n+4]; n+=4
		self.bu = gd[n:n+4]; n+=4
		self.bd = gd[n:n+4]; n+=4
		self.bsq = mdot(self.bu,self.bd)
		self.v1m,self.v1p,self.v2m,self.v2p,self.v3m,self.v3p=gd[n:n+6]; n+=6
		self.gdet=gd[n]; n+=1
		self.rhor = 1+(1-self.a**2)**0.5
		if hasattr(self, 'guu'):
		#if "guu" in globals():
			#lapse
			alpha = (-self.guu[0,0])**(-0.5)
		if n != gd.shape[0]:
			print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
			return 1
		return 0

	def rdump_assign(self,gd,**kwargs):
		#global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor, ktot, Ttot, game, qisosq, pflag, qisodotb, kel, uelvar, Tel4, Tel5,Teldis, Tels, kel4, kel5,ugel,ugeldis, ugcon, sel, ugscon, ugel4, ugel5,stot, uelvar, Telvar, Tsel, sel, ugels, games, phi, keldis, phihat,csphib,lrho
		self.nx = kwargs.pop("nx",self.nx)
		self.ny = kwargs.pop("ny",self.ny)
		self.nz = kwargs.pop("nz",self.nz)
		n = 0
		self.rho = gd[n]; n+=1
		self.ug = gd[n]; n+=1
		self.vu=np.zeros_like(gd[0:4])
		self.B=np.zeros_like(gd[0:4])
		self.vu[1:4] = gd[n:n+3]; n+=3
		self.B[1:4] = gd[n:n+3]; n+=3
		# if n != gd.shape[0]:
		#     print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
		#     return 1
		return gd

	def fdump_assign(self,gd,**kwargs):
		#global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor, ktot, Ttot, game, qisosq, pflag, qisodotb, kel, uelvar, Tel4, Tel5,Teldis, Tels, kel4, kel5,ugel,ugeldis, ugcon, sel, ugscon, ugel4, ugel5,stot, uelvar, Telvar, Tsel, sel, ugels, games, phi, keldis, phihat,csphib,lrho,fail
		self.nx = kwargs.pop("nx",self.nx)
		self.ny = kwargs.pop("ny",self.ny)
		self.nz = kwargs.pop("nz",self.nz)
		self.fail = gd
		return gd



	def cartesian_sasha(self,yourvar,k=0,xy=1,xcoord=None,ycoord=None,symmx=0,mirrorx=0,mirrory=0):
		"""
	This was adapted from `plc` at `harm_script.py`, atchekho/harmpi.
	Need to understand this better. 

	- k: 3D slice (if applicable)
		"""
		r=self.r 
		h=self.h
		ph=self.ph

		# avoids trouble
		myvar=yourvar.copy()

		if np.abs(xy)==1:
			if xcoord is None: xcoord = r * np.sin(h)
			if ycoord is None: ycoord = r * np.cos(h)
			if mirrory: ycoord *= -1
			if mirrorx: xcoord *= -1
		if xcoord is not None and ycoord is not None:
			xcoord = xcoord[:,:,None] if xcoord.ndim == 2 else xcoord[:,:,k:k+1]
			ycoord = ycoord[:,:,None] if ycoord.ndim == 2 else ycoord[:,:,k:k+1]
		if np.abs(xy)==1 and symmx:
			if myvar.ndim == 2:
				myvar = myvar[:,:,None] if myvar.ndim == 2 else myvar[:,:,k:k+1]
				myvar=np.concatenate((myvar[:,::-1],myvar),axis=1)
				xcoord=np.concatenate((-xcoord[:,::-1],xcoord),axis=1)
				ycoord=np.concatenate((ycoord[:,::-1],ycoord),axis=1)
			else:
				if myvar.shape[-1] > 1: 
					symmk = (k+self.nz/2)%self.nz 
				else: 
					symmk = k
				myvar=np.concatenate((myvar[:,self.ny-1:self.ny,k:k+1],myvar[:,::-1,symmk:symmk+1],myvar[:,:,k:k+1]),axis=1)
				xcoord=np.concatenate((xcoord[:,self.ny-1:self.ny,k:k+1],-xcoord[:,::-1],xcoord),axis=1)
				ycoord=np.concatenate((ycoord[:,self.ny-1:self.ny,k:k+1],ycoord[:,::-1],ycoord),axis=1)
		elif np.abs(xy) == 2 and symmx:
			#if fracphi == 0.5 done in a robust way
			if get_fracphi() < 0.75:
				r1 = np.concatenate((r,r,r[...,0:1]),axis=2)
				ph1 = np.concatenate((ph,ph+np.pi,ph[...,0:1]+2*np.pi),axis=2)
				myvar = np.concatenate((myvar,myvar,myvar[...,0:1]),axis=2)
			else:
				r1 = np.concatenate((r,r[...,0:1]),axis=2)
				ph1 = np.concatenate((ph,ph[...,0:1]+2*np.pi),axis=2)
				myvar = np.concatenate((myvar,myvar[...,0:1]),axis=2)
			xcoord=(r1*cos(ph1))[:,self.ny/2,:,None]
			ycoord=(r1*sin(ph1))[:,self.ny/2,:,None]
			myvar = myvar[:,self.ny/2,:,None]
		else:
			if myvar.ndim == 2:
				myvar = myvar[:,:,None]  
			else:
				myvar[:,:,k:k+1]

		return myvar[:,:,k],xcoord[:,:,k],ycoord[:,:,k]


	def cartesian(self,myvar=None):
		"""
	Computes cartesian coordinates.

	Arguments:

	- k: 3D slice (if applicable)
		"""
		r=self.r 
		th=self.h
		phi=self.ph

		if self.rho.shape[2]>1: # 3D
			self.x=r*numpy.sin(th)*numpy.cos(phi)
			self.y=r*numpy.sin(th)*numpy.sin(phi)
			self.z=r*numpy.cos(th)
		else: # 2D
			x=r*numpy.sin(th)
			y=r*numpy.cos(th)

			self.x=x[:,:,0]
			self.y=y[:,:,0]

			if myvar is not None:
				return myvar[:,:,0]













def myfloat(f,acc=1): # Sasha
	""" acc=1 means np.float32, acc=2 means np.float64 """
	if acc==1:
		return( np.float32(f) )
	else:
		return( np.float64(f) )

def mdot(a,b):
	"""
    Computes a contraction of two tensors/vectors.  Assumes
    the following structure: tensor[m,n,i,j,k] OR vector[m,i,j,k], 
    where i,j,k are spatial indices and m,n are variable indices. 
	"""
	if (a.ndim == 3 and b.ndim == 3) or (a.ndim == 4 and b.ndim == 4):
		c = (a*b).sum(0)
	elif a.ndim == 5 and b.ndim == 4:
		c = np.empty(np.maximum(a[:,0,:,:,:].shape,b.shape),dtype=b.dtype)
		for i in range(a.shape[0]):
			c[i,:,:,:] = (a[i,:,:,:,:]*b).sum(0)
	elif a.ndim == 4 and b.ndim == 5:
		c = np.empty(np.maximum(b[0,:,:,:,:].shape,a.shape),dtype=a.dtype)
		for i in range(b.shape[1]):
			c[i,:,:,:] = (a*b[:,i,:,:,:]).sum(0)
	elif a.ndim == 5 and b.ndim == 5:
		c = np.empty((a.shape[0],b.shape[1],a.shape[2],a.shape[3],max(a.shape[4],b.shape[4])),dtype=a.dtype)
		for i in range(c.shape[0]):
			for j in range(c.shape[1]):
				c[i,j,:,:,:] = (a[i,:,:,:,:]*b[:,j,:,:,:]).sum(0)
	elif a.ndim == 5 and b.ndim == 6:
		c = np.empty((a.shape[0],b.shape[1],b.shape[2],max(a.shape[2],b.shape[3]),max(a.shape[3],b.shape[4]),max(a.shape[4],b.shape[5])),dtype=a.dtype)
		for mu in range(c.shape[0]):
			for k in range(c.shape[1]):
				for l in range(c.shape[2]):
					c[mu,k,l,:,:,:] = (a[mu,:,:,:,:]*b[:,k,l,:,:,:]).sum(0)
	else:
		raise Exception('mdot', 'wrong dimensions')
	return c


				
def fixminus(x):
	"""
	Replace nonphysical, negative values in array *x* with the corresponding
	positive numerical values. Returns modified array. Does not 
	touch original array.
	"""
	i=numpy.where(x<0)
	z=x.copy()
	z[i]=numpy.abs(x[i])

	return z




def wolframplot(infile,outfile,script="/Users/nemmen/work/software/mathematica/raishin.wl"):
	"""
Makes a pretty plot of density field and B vector field of RAISHIN
data using the Wolfram Language.

Make sure you point to the appropriate Wolfram script.

- infile: input RAISHIN ASCII file generated with Raishin.vtk above, e.g. ok200.dat
- outfile: format that will produced, e.g. ok200.png
	"""
	import subprocess
	cmd="wolframscript -script "+script+" "+infile+" "+outfile
	subprocess.call(cmd.split())




def regridFast(self, n=None, xlim = None):
	"""
Transforms a mesh in arbitrary coordinates (e.g. nonuniform elements)
into a uniform grid in the same coordinates. Uses a C function to 
speed things up. 

One has to be particularly careful below about using a polar angle
(-pi/2<theta<pi/2) vs a spherical polar angle (0<theta_sph<pi). The
choice can affect some specific transformations.

:param n: New number of elements n^2. If None, figures out by itself
:param xlim: Boundary for the plot and the grid
	"""
	import nmmn.lsd, nmmn.misc

	# C function for fast regridding. Make sure you compile it first
	# with make
	import fastregrid

	# creates copy of current object which will have the new
	# coordinates
	obj=Pluto() # empty pluto object

	# r, theta
	r=self.x1
	th=-(self.x2-numpy.pi/2.) # spherical angle => polar angle
	if(xlim == None):
			xlim = self.x1.max()
	gmtry = self.pp.geometry

	# figures out optimal size of cartesian grid
	if n is None:
		n=self.optimalgrid()

		# let's avoid dealing with arrays which are too large
		if n>3000:
			n=3000

	if(gmtry == "SPHERICAL" or gmtry == "CYLINRICAL"):
		xnew=numpy.linspace(0, xlim, n)
		ynew=numpy.linspace(-xlim, xlim, n)
	else:
		xnew=numpy.linspace(-xlim, xlim, n)
		ynew=numpy.linspace(-xlim, xlim, n)

	rho=numpy.zeros((n,n))
	vx=numpy.zeros((n,n))
	vy=numpy.zeros((n,n))
	vz=numpy.zeros((n,n)) # vphi
	p=rho.copy()

	if(gmtry == "SPHERICAL"):
		fastregrid.regrid(xnew, ynew, r, th, self.rho, self.p, self.v1, self.v2, self.v3, rho, p, vx, vy, vz)		
	else: #polar case for bondi
		print("Geometry not supported. Improve the method.")

	# coordinate arrays
	obj.x1,obj.x2=xnew,ynew # cartesian coords, 1D
	obj.X1,obj.X2=numpy.meshgrid(xnew,ynew) # cartesian coords, 2D
	obj.r, obj.th = nmmn.misc.cart2pol(xnew, ynew) # polar coords, 1D
	obj.R, obj.TH = numpy.meshgrid(obj.r,obj.th) # polar coords, 2D
	obj.rsp, obj.thsp = obj.r, numpy.pi/2.-obj.th # spherical polar angle, 1D
	obj.RSP, obj.THSP = numpy.meshgrid(obj.rsp,obj.thsp) # spherical polar coords, 2D

	# velocities
	obj.v1,obj.v2,obj.v3 = vx.T,vy.T,vz.T # Cartesian components
	obj.vr, obj.vth = nmmn.misc.vel_c2p(obj.TH,obj.v1,obj.v2) # polar components
	obj.speed = numpy.sqrt(obj.v1**2+obj.v2**2+obj.v3**3)

	# fluid variables
	obj.gamma=self.gamma
	obj.rho,obj.p=rho.T,p.T
	obj.entropy=numpy.log(obj.p/obj.rho**obj.gamma)
	obj.am=obj.v3*obj.R*numpy.sin(obj.THSP) # specific a. m., vphi*r*sin(theta)
	obj.Be=obj.speed**2/2.+obj.gamma*obj.p/((obj.gamma-1.)*obj.rho)-1./obj.R	# Bernoulli function
	obj.Omega=obj.v3/obj.R	# angular velocity

	# misc info
	obj.regridded=True # flag to tell whether the object was previously regridded
	obj.t=self.t
	obj.frame=self.frame
	obj.mdot=self.mdot
	obj.mass=self.mass

	return obj



