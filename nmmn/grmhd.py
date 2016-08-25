"""
Misc. modules useful for dealing with (GR)(M)HD simulations
=========================================================

- RAISHIN
- Pluto
- HARM (soon)
"""

import numpy
import scipy

f = open(fv,"r")
newf=open("tmp.dat","w")

boold,boolp,boolbx,boolby=False,False,False,False
strd,strbx,strby='','','' # string that holds values 

for line in f:
    # gets dimensions
    if re.search(r'DIMENSIONS\s+\d+\s+\d+',line):
        s=re.findall(r'\s*\d+\s*',line.rstrip('\n'))
        nx=int(s[0])
        ny=int(s[1])
        
    
    # gets mesh: number number 
    if re.search(r'-?\d+\.\d+E?[-+]?\d+\s+-?\d+\.\d+E?[-+]?\d+\s+-?\d+\.\d+E?[-+]?\d+',line):
        newf.write(line)
            
    # gets arrays
    # these lines are important to tell python when to stop reading shit
    # it must be sequential
    if 'density' in line: boold=True
    if 'pressure' in line: boold=False
    if 'bx' in line:
        boolbx=True
    if 'by' in line:
        boolbx=False
        boolby=True    
    if 'bz' in line: boolby=False    
    
    if boold==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
        strd=strd+line
    if boolbx==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
        strbx=strbx+line
    if boolby==True and re.search(r'-?\d+\.\d+E?[-+]?\d+',line):
        strby=strby+line


        
f.close()
newf.close()

