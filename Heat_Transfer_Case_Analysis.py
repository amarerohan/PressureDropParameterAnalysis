import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(40)

c1 = 1
c2 = 3

e1 = 5
e2 = 5

X1 = np.load('case'+str(c1)+'/heat_solutions/'+'Heat_X_' + str(e1) + '_a_0.001_q_1000_Tin_20_Tamb_0_hamb_10_hbt_10_.npy')
X2 = np.load('case'+str(c2)+'/heat_solutions/'+'Heat_X_' + str(e2) + '_a_0.001_q_1000_Tin_20_Tamb_0_hamb_10_hbt_10_.npy')

dom1 = np.load('case'+str(c1)+'/2.5_dom.npy')
dom2 = np.load('case'+str(c2)+'/2.5_dom.npy')

cdom1 = np.load('case'+str(c1)+'/2.5_cdom.npy')
cdom2 = np.load('case'+str(c2)+'/2.5_cdom.npy')

unt1 = np.max(cdom1) + 1
unt2 = np.max(cdom2) + 1


TTis1 = X1[:unt1]
TVes1 = X1[unt1:]

TTis2 = X2[:unt2]
TVes2 = X2[unt2:]



nx, ny, nz = np.shape(dom1)

T1 = np.zeros((nx,ny,nz), dtype = float)
T2 = np.zeros((nx,ny,nz), dtype = float)
error_T = np.zeros((nx,ny,nz), dtype = float)


for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom1[i,j,k] == 0):
                T1[i,j,k] = X1[int(cdom1[i,j,k])]
            if(dom2[i,j,k] == 0):
                T2[i,j,k] = X2[int(cdom2[i,j,k])]


z = 40


T_min = min(np.min(TTis1), np.min(TTis2))
T_max = max(np.max(TTis1), np.max(TTis2))
dT = (T_max - T_min)/50

for z in range(1,nz-1):
    plt.figure(figsize=(20,12))
    color = 'Reds' # plt.cm.get_cmap('Reds') 
    
    plt.subplot(121)
    
    plt.contourf(T1[1:-1,1:-1,z],np.arange(T_min,T_max+2*dT,dT),cmap=color)#.reversed())
    # plt.colorbar()
    plt.axis('off')
    plt.title('case '+str(c1)+' z = '+ str(z),y = -0.05,font=font)
    
    # cax = plt.axes([-0.15, 0.05, 0.022, 0.9])
    # cbar = plt.colorbar(cax=cax)
    # cbar.ax.tick_params(labelsize = 40)#FontProperties=font)# ,labelsize = 35)
    # cbar.set_label('Pressure (Pa)', rotation=270,labelpad=+65,font=font)
    
    
    
    plt.subplot(122)
    plt.contourf(T2[1:-1,1:-1,z],np.arange(T_min,T_max+2*dT,dT),cmap=color)#.reversed())
    # plt.colorbar()
    plt.axis('off')
    plt.title('case '+str(c2)+' z = '+ str(z),y = -0.05,font=font)
    
    cax = plt.axes([1.00, 0.05, 0.022, 0.9])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 40)#FontProperties=font)# ,labelsize = 35)
    cbar.set_label('Temperature', rotation=270,labelpad=+65,font=font)
    
    
    plt.tight_layout()
    plt.show()