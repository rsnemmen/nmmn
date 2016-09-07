"""
Misc. modules useful for dealing with (GR)(R)(M)HD simulations
=========================================================

- RAISHIN
- Pluto
- HARM (soon)

See jupyter notebooks "grmhd*" for examples on how to use this
module.
"""

import numpy, scipy
import tqdm




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
	Exports data as compressed HDF5. 7x less space tham ASCII.
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


