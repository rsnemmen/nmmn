"""
Misc. modules useful for dealing with (GR)(M)HD simulations
=========================================================

- RAISHIN
- Pluto
- HARM (soon)
"""

import numpy
import scipy




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

Reads data from a VTK file:

>>> o.vtk("ok200.vtk")
	"""

	def __init__(self):
		# does nothing for now





	def vtk(self, vtkfile):
		"""
	Given a VTK file created with the RAISHIN GRMHD code, this reads the
	data as numpy arrays.
		"""
		f = open(vtkfile,"r")
		newf=open("tmp.dat","w")	# file that will hold coordinates

		# booleans that will tell the code when to stop reading
		# data for a given variable:
		# boold for density, boolp for pressure etc
		boold,boolp,boolvx,boolvy,boolvz,boolb2,boolbx,boolby,boolbz=False,False,False,False,False,False,False,False,False
		strd,strp,strvx,strvy,strvz,strb2,strbx,strby,strbz='','','','','','','','','' # string that holds values 

		for line in f:
		    # gets dimensions
		    if re.search(r'DIMENSIONS\s+\d+\s+\d+',line):
		        s=re.findall(r'\s*\d+\s*',line.rstrip('\n'))
		        nx=int(s[0])
		        ny=int(s[1])
		        nz=int(s[2])
		        
		    # gets mesh: number number 
		    if re.search(r'-?\d+\.\d+E?[-+]?\d+\s+-?\d+\.\d+E?[-+]?\d+\s+-?\d+\.\d+E?[-+]?\d+',line):
		        newf.write(line)
		            
		    # gets arrays
		    # these lines are important to tell python when to stop reading shit
		    # it must be sequential
		    if 'density' in line: boold=True
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

		        
		f.close()
		newf.close()






