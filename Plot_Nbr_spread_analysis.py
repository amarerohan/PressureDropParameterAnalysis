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


E_array = [0,38,49,53,66,71]

c = 1
E = E_array[c]
n1 = np.load('case_'+str(c)+'_E_'+str(E)+'_nbr_spread.npy')

c = 2
E = E_array[c]
n2 = np.load('case_'+str(c)+'_E_'+str(E)+'_nbr_spread.npy')

c = 3
E = E_array[c]
n3 = np.load('case_'+str(c)+'_E_'+str(E)+'_nbr_spread.npy')

c = 4
E = E_array[c]
n4 = np.load('case_'+str(c)+'_E_'+str(E)+'_nbr_spread.npy')

c = 5
E = E_array[c]
n5 = np.load('case_'+str(c)+'_E_'+str(E)+'_nbr_spread.npy')


en2 = n2-n1
en3 = n3-n1
en4 = n4-n1
en5 = n5-n1

z_array = [1,20,40,60,80]

spread = np.arange(-14,15,1)
colors = 'RdBu'
for z in z_array:
    plt.figure(figsize = (45,12), dpi = 300)
    plt.subplot(1,4,1)
    plt.contourf(en2[0:-1,0:-1,z],spread, cmap = colors)
    plt.axis('off')
    
    plt.subplot(1,4,2)
    plt.contourf(en3[0:-1,0:-1,z],spread, cmap = colors)
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.contourf(en4[0:-1,0:-1,z],spread, cmap = colors)
    plt.axis('off')
    
    plt.subplot(1,4,4)
    plt.contourf(en5[0:-1,0:-1,z],spread, cmap = colors)
    plt.axis('off')
    
    cax = plt.axes([1.01, 0.01, 0.022, 0.95])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    # cbar.set_label('mass flow difference', rotation=270,labelpad=+115,font=font, fontsize = 85)
    
    plt.suptitle('z = ' + str(z), y = 1.02, font=font, fontsize = 85)
    
    plt.tight_layout()
    plt.show()
    
