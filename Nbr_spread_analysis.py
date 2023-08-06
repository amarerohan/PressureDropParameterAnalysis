import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sb
import pandas as pd
import time as time

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(60)

myu = 1e-3
a = 1E-3
Ka = 1E-5
Kv = 1E-5
Lambda_a = Ka/myu
Lambda_v = Kv/myu


Dx = 2.5
dx = dy = 0.001*Dx
dz = dx
dVol = 1 

case = 5

E_array = [0,38,49,53,66,71]
E = 100 #E_array[case]
e = E*dx

nbr_a = np.load('case'+str(case)+'/artery_nbrhd_'+str(E)+'.npy', allow_pickle=True)
nbr_v = np.load('case'+str(case)+'/vein_nbrhd_'+str(E)+'.npy', allow_pickle=True)



dom = np.load('case'+str(case)+'/2.5_dom.npy')

nx,ny,nz = np.shape(dom)

nbr = np.zeros((nx, ny, nz), dtype = int)

for k in range(len(nbr_a)):
    for j in range(len(nbr_a[k])):
           x,y,z = nbr_a[k][j]
           nbr[x,y,z] = nbr[x,y,z] + 1
   
# z_array = [1,20,40,60,80]
# nbr_range = np.arange(0,np.max(nbr)+1,1)
# for z in z_array:
#     plt.figure(figsize = (6,4))
#     plt.contourf(nbr[0:-1,0:-1,z],nbr_range)
#     plt.colorbar()
#     plt.axis('off')
#     plt.show()


np.save('case_'+str(case)+'_E_'+str(E)+'_nbr_spread.npy',nbr)