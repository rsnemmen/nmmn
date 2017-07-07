"""
Dealing with (GR)(R)(M)HD simulations
========================================

- RAISHIN
- Pluto
- HARM

See jupyter notebooks "grmhd*" for examples on how to use this
module.
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
	plotting with python.

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
for plotting with matplotlib, mayavi etc. Trying to do minimal modifications
to Sasha's codes in order to get them working asap.

Attributes of the object:

TBD

Reads data from a harm dump file, generates new attributes rho, p, vx, bx etc:

>>> o=nmmn.grmhd.Harm("dump019")

Saves data as an ASCII file with columns corresponding to variables:

>>> o.savetxt("ok200.dat")
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



	def read_file(dump,type=None,savedump=True,saverdump=False,noround=False):
		"""
	High-level function that reads either MPI or serial gdump's
		"""
		import sys

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
		if os.path.isfile( "dumps/" + dump ):
			headerline = read_header("dumps/" + dump, returnheaderline = True)
		    gd = read_body("dumps/" + dump,nx=N1+2*N1G,ny=N2+2*N2G,nz=N3+2*N3G,noround=1)
		    if noround:
				res = data_assign(         gd,type=type,nx=N1+2*N1G,ny=N2+2*N2G,nz=N3+2*N3G)
			else:
				res = data_assign(myfloat(gd),type=type,nx=N1+2*N1G,ny=N2+2*N2G,nz=N3+2*N3G)
			return res

	    #MPI-type dump that is spread over many files
		else:
			flist = np.sort(glob.glob( "dumps/" + dump + "_[0-9][0-9][0-9][0-9]" ))
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
				dumpfullname = "dumps/" + dump
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

	#read in a header
	def read_header(dump,issilent=True,returnheaderline=False):
		global t,nx,ny,nz,N1,N2,N3,N1G,N2G,N3G,starti,startj,startk,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,gn3,gv3,guu,gdd,dxdxp, games, startx1, startx2, startx3, tf, NPR, DOKTOT, BL
		global fractheta
		global fracphi
		global rbr
		global npow2
		global cpow2
		#read image
		fin = open( dump, "rb" )
		headerline = fin.readline()
		header = headerline.split()
		nheadertot = len(header)
		fin.close()
		if not dump.startswith("dumps/rdump"):
			if not issilent: print( "dump header: len(header) = %d" % len(header) )
			nheader = 45
			n = 0
			t = myfloat(np.float64(header[n])); n+=1
			#per tile resolution
			N1 = int(header[n]); n+=1
			N2 = int(header[n]); n+=1
			N3 = int(header[n]); n+=1
			#total resolution
			nx = int(header[n]); n+=1
			ny = int(header[n]); n+=1
			nz = int(header[n]); n+=1
			#numbers of ghost cells
			N1G = int(header[n]); n+=1
			N2G = int(header[n]); n+=1
			N3G = int(header[n]); n+=1
			startx1 = myfloat(float(header[n])); n+=1
			startx2 = myfloat(float(header[n])); n+=1
			startx3 = myfloat(float(header[n])); n+=1
			_dx1=myfloat(float(header[n])); n+=1
			_dx2=myfloat(float(header[n])); n+=1
			_dx3=myfloat(float(header[n])); n+=1
			tf=myfloat(float(header[n])); n+=1
			nstep=myfloat(float(header[n])); n+=1
			a=myfloat(float(header[n])); n+=1
			gam=myfloat(float(header[n])); n+=1
			cour=myfloat(float(header[n])); n+=1
			DTd=myfloat(float(header[n])); n+=1
			DTl=myfloat(float(header[n])); n+=1
			DTi=myfloat(float(header[n])); n+=1
			DTr=myfloat(float(header[n])); n+=1
			DTr01=myfloat(float(header[n])); n+=1
			dump_cnt=myfloat(float(header[n])); n+=1
			image_cnt=myfloat(float(header[n])); n+=1
			rdump_cnt=myfloat(float(header[n])); n+=1
			rdump01_cnt=myfloat(float(header[n])); n+=1
			dt=myfloat(float(header[n])); n+=1
			lim=myfloat(float(header[n])); n+=1
			failed=myfloat(float(header[n])); n+=1
			Rin=myfloat(float(header[n])); n+=1
			Rout=myfloat(float(header[n])); n+=1
			hslope=myfloat(float(header[n])); n+=1
			R0=myfloat(float(header[n])); n+=1
			NPR=int(header[n]); n+=1
			DOKTOT=int(header[n]); n+=1
			fractheta = myfloat(header[n]); n+=1
			fracphi   = myfloat(header[n]); n+=1
			rbr       = myfloat(header[n]); n+=1
			npow2     = myfloat(header[n]); n+=1
			cpow2     = myfloat(header[n]); n+=1
			BL = myfloat(header[n]); n+=1
		else:
			print("rdump header")
			nheader = 46
			n = 0
			#per tile resolution
			N1 = int(header[n]); n+=1
			N2 = int(header[n]); n+=1
			N3 = int(header[n]); n+=1
			#total resolution
			nx = int(header[n]); n+=1
			ny = int(header[n]); n+=1
			nz = int(header[n]); n+=1
			#numbers of ghost cells
			N1G = int(header[n]); n+=1
			N2G = int(header[n]); n+=1
			N3G = int(header[n]); n+=1
			#starting indices
			starti = int(header[n]); n+=1
			startj = int(header[n]); n+=1
			startk = int(header[n]); n+=1
			t = myfloat(header[n]); n+=1
			tf = myfloat(header[n]); n+=1
			nstep = int(header[n]); n+=1
			a = myfloat(header[n]); n+=1
			gam = myfloat(header[n]); n+=1
			game = myfloat(header[n]); n+=1
			game4 = myfloat(header[n]); n+=1
			game5 = myfloat(header[n]); n+=1
			cour = myfloat(header[n]); n+=1
			DTd = myfloat(header[n]); n+=1
			DTl = myfloat(header[n]); n+=1
			DTi = myfloat(header[n]); n+=1
			DTr = myfloat(header[n]); n+=1
			DTr01 = myfloat(header[n]); n+=1
			dump_cnt = myfloat(header[n]); n+=1
			image_cnt = myfloat(header[n]); n+=1
			rdump_cnt = myfloat(header[n]); n+=1
			rdump01_cnt=myfloat(float(header[n])); n+=1
			dt = myfloat(header[n]); n+=1
			lim = myfloat(header[n]); n+=1
			failed = myfloat(header[n]); n+=1
			Rin = myfloat(header[n]); n+=1
			Rout = myfloat(header[n]); n+=1
			hslope = myfloat(header[n]); n+=1
			R0 = myfloat(header[n]); n+=1
			fractheta = myfloat(header[n]); n+=1
			fracphi = myfloat(header[n]); n+=1
			rbr = myfloat(header[n]); n+=1
			npow2 = myfloat(header[n]); n+=1
			cpow2 = myfloat(header[n]); n+=1
			tdump = myfloat(header[n]); n+=1
			trdump = myfloat(header[n]); n+=1
			timage = myfloat(header[n]); n+=1
			tlog  = myfloat(header[n]); n+=1
		if n != nheader or n != nheadertot:
			print("Wrong number of elements in header: nread = %d, nexpected = %d, nototal = %d: incorrect format?"
				% (n, nheader, nheadertot) )
			return headerline
		if returnheaderline:
			return headerline
		else:
			return header
	            
	def read_body(dump,nx=None,ny=None,nz=None,noround=False):
		fin = open( dump, "rb" )
		header = fin.readline()
		if dump.startswith("dumps/rdump"):
			dtype = np.float64
			body = np.fromfile(fin,dtype=dtype,count=-1)
			gd = body.view().reshape((nx,ny,nz,-1), order='C')
			if noround:
				gd=gd.transpose(3,0,1,2)
			else:
				gd=myfloat(gd.transpose(3,0,1,2))
			elif dump.startswith("dumps/gdump2"):
				dtype = np.float64
				body = np.fromfile(fin,dtype=dtype,count=-1)
				gd = body.view().reshape((nx,ny,nz,-1), order='C')
				if noround:
					gd=gd.transpose(3,0,1,2)
				else:
					gd=myfloat(gd.transpose(3,0,1,2))
			elif dump.startswith("dumps/fdump"):
				dtype = np.int64
				body = np.fromfile(fin,dtype=dtype,count=-1)
				gd = body.view().reshape((-1,nz,ny,nx), order='F')
				gd=myfloat(gd.transpose(0,3,2,1))
			else:
				dtype = np.float32
				body = np.fromfile(fin,dtype=dtype,count=-1)
				gd = body.view().reshape((-1,nz,ny,nx), order='F')
				gd=myfloat(gd.transpose(0,3,2,1))
			return gd

	def data_assign(gd,type=None,**kwargs):
		if type is None:
			print("Please specify data type")
			return
		if type == "gdump":
			gdump_assign(gd,**kwargs)
			return None
		elif type == "gdump2":
			gdump2_assign(gd,**kwargs)
			return None
		elif type == "dump":
			dump_assign(gd,**kwargs)
			return None
		elif type == "rdump":
			gd = rdump_assign(gd,**kwargs)
			return gd
		elif type == "fdump":
			gd = fdump_assign(gd,**kwargs)
			return gd
		else:
			print("Unknown data type: %s" % type)
			return gd
	    
	def gdump_assign(gd,**kwargs):
		global t,nx,ny,nz,N1,N2,N3,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,r,h,ph,gcov,gcon,gdet,drdx,gn3,gv3,guu,gdd,dxdxp, games
		nx = kwargs.pop("nx",nx)
		ny = kwargs.pop("ny",ny)
		nz = kwargs.pop("nz",nz)
		ti,tj,tk,x1,x2,x3,r,h,ph = gd[0:9,:,:].view();  n = 9
		gv3 = gd[n:n+16].view().reshape((4,4,nx,ny,nz),order='F').transpose(1,0,2,3,4); n+=16
		gn3 = gd[n:n+16].view().reshape((4,4,nx,ny,nz),order='F').transpose(1,0,2,3,4); n+=16
		gcov = gv3
		gcon = gn3
		guu = gn3
		gdd = gv3
		gdet = gd[n]; n+=1
		drdx = gd[n:n+16].view().reshape((4,4,nx,ny,nz),order='F').transpose(1,0,2,3,4); n+=16
		dxdxp = drdx
		if n != gd.shape[0]:
			print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
			return 1
		return 0

	def gdump2_assign(gd,**kwargs):
		global t,nx,ny,nz,N1,N2,N3,_dx1,_dx2,_dx3,a,gam,Rin,Rout,hslope,R0,ti,tj,tk,x1,x2,x3,gdet,games,rf1,hf1,phf1,rf2,hf2,phf2,rf3,hf3,phf3,rcorn,hcord,phcorn,re1,he1,phe1,re2,he2,phe2,re3,he3,phe3
		nx = kwargs.pop("nx",nx)
		ny = kwargs.pop("ny",ny)
		nz = kwargs.pop("nz",nz)
		ti,tj,tk,x1,x2,x3 = gd[0:6,:,:].view();  n = 6
		rf1,hf1,phf1,rf2,hf2,phf2,rf3,hf3,phf3 = gd[0:9,:,:].view();  n += 9
		rcorn,hcord,phcorn,rcent,hcent,phcen = gd[0:6,:,:].view();  n += 6
		re1,he1,phe1,re2,he2,phe2,re3,he3,phe3 = gd[0:9,:,:].view();  n += 9
		gdet = gd[n]; n+=1
		if n != gd.shape[0]:
			print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
			return 1
		return 0

	#read in a dump file
	def dump_assign(gd,**kwargs):
		global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor, ktot, pg
		nx = kwargs.pop("nx",nx)
		ny = kwargs.pop("ny",ny)
		nz = kwargs.pop("nz",nz)
		ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug = gd[0:11,:,:].view(); n = 11
		pg = (gam-1)*ug
		lrho=np.log10(rho)
		vu=np.zeros_like(gd[0:4])
		B=np.zeros_like(gd[0:4])
		vu[1:4] = gd[n:n+3]; n+=3
		B[1:4] = gd[n:n+3]; n+=3
		#if total entropy equation is evolved (on by default)
		if DOKTOT == 1:
			ktot = gd[n]; n+=1
		divb = gd[n]; n+=1
		uu = gd[n:n+4]; n+=4
		ud = gd[n:n+4]; n+=4
		bu = gd[n:n+4]; n+=4
		bd = gd[n:n+4]; n+=4
		bsq = mdot(bu,bd)
		v1m,v1p,v2m,v2p,v3m,v3p=gd[n:n+6]; n+=6
		gdet=gd[n]; n+=1
		rhor = 1+(1-a**2)**0.5
		if "guu" in globals():
			#lapse
			alpha = (-guu[0,0])**(-0.5)
		if n != gd.shape[0]:
			print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
			return 1
		return 0

	def rdump_assign(gd,**kwargs):
		global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor, ktot, Ttot, game, qisosq, pflag, qisodotb, kel, uelvar, Tel4, Tel5,Teldis, Tels, kel4, kel5,ugel,ugeldis, ugcon, sel, ugscon, ugel4, ugel5,stot, uelvar, Telvar, Tsel, sel, ugels, games, phi, keldis, phihat,csphib,lrho
		nx = kwargs.pop("nx",nx)
		ny = kwargs.pop("ny",ny)
		nz = kwargs.pop("nz",nz)
		n = 0
		rho = gd[n]; n+=1
		ug = gd[n]; n+=1
		vu=np.zeros_like(gd[0:4])
		B=np.zeros_like(gd[0:4])
		vu[1:4] = gd[n:n+3]; n+=3
		B[1:4] = gd[n:n+3]; n+=3
		# if n != gd.shape[0]:
		#     print("rd: WARNING: nread = %d < ntot = %d: incorrect format?" % (n, gd.shape[0]) )
		#     return 1
		return gd

	def fdump_assign(gd,**kwargs):
		global t,nx,ny,nz,_dx1,_dx2,_dx3,gam,hslope,a,R0,Rin,Rout,ti,tj,tk,x1,x2,x3,r,h,ph,rho,ug,vu,B,pg,cs2,Sden,U,gdetB,divb,uu,ud,bu,bd,v1m,v1p,v2m,v2p,gdet,bsq,gdet,alpha,rhor, ktot, Ttot, game, qisosq, pflag, qisodotb, kel, uelvar, Tel4, Tel5,Teldis, Tels, kel4, kel5,ugel,ugeldis, ugcon, sel, ugscon, ugel4, ugel5,stot, uelvar, Telvar, Tsel, sel, ugels, games, phi, keldis, phihat,csphib,lrho,fail
		nx = kwargs.pop("nx",nx)
		ny = kwargs.pop("ny",ny)
		nz = kwargs.pop("nz",nz)
		fail = gd
		return gd











def pol2cart():
	"""
TBC!!!!!!!!!!!!!!!!!!!!!!
Takes care of converting from polar to cartesian coordinates for HARM
(and potentially other GRMHD codes). 

Taken from `harm_script.py` at atchekho/harmpi.
	"""
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
	            symmk = (k+nz/2)%nz 
	        else: 
	            symmk = k
	        myvar=np.concatenate((myvar[:,ny-1:ny,k:k+1],myvar[:,::-1,symmk:symmk+1],myvar[:,:,k:k+1]),axis=1)
	        xcoord=np.concatenate((xcoord[:,ny-1:ny,k:k+1],-xcoord[:,::-1],xcoord),axis=1)
	        ycoord=np.concatenate((ycoord[:,ny-1:ny,k:k+1],ycoord[:,::-1],ycoord),axis=1)
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
	    xcoord=(r1*cos(ph1))[:,ny/2,:,None]
	    ycoord=(r1*sin(ph1))[:,ny/2,:,None]
	    myvar = myvar[:,ny/2,:,None]
	else:
	    myvar = myvar[:,:,None] if myvar.ndim == 2 else myvar[:,:,k:k+1]





				
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


