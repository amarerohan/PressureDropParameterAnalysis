import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sb
sb.set(rc={'figure.figsize':(11.7,8.27)})
import pandas as pd

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(60)



case1 = 1
e1 = 38
T_tis1 = np.load('case'+str(case1)+'/heat_solutions/' + str(e1)+'/'+'T_domain_'+str(e1)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

case2 = 2
e2 = 49
T_tis2 = np.load('case'+str(case2)+'/heat_solutions/' + str(e2)+'/'+'T_domain_'+str(e2)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

case3 = 3
e3 = 53
T_tis3 = np.load('case'+str(case3)+'/heat_solutions/' + str(e3)+'/'+'T_domain_'+str(e3)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

case4 = 4
e4 = 66
T_tis4 = np.load('case'+str(case4)+'/heat_solutions/' + str(e4)+'/'+'T_domain_'+str(e4)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

case5 = 5
e5 = 71
T_tis5 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

z_array = [1,20,40,60,80]


colors = 'Reds'
fontsize  = 65

T_array = np.arange(28,36.5,0.5)

for z in z_array:
    plt.figure(figsize=(35,25), dpi = 100)
    
    plt.subplot(2,3,1)
    plt.contourf(T_tis1[:,:,z], T_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 1', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,2)
    plt.contourf(T_tis2[:,:,z], T_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 2', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,3)
    plt.contourf(T_tis3[:,:,z], T_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 3', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,4)
    plt.contourf(T_tis4[:,:,z], T_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 4', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,5)
    plt.contourf(T_tis5[:,:,z], T_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 5', y = -0.075, font = font, fontsize = fontsize)
    
    cax = plt.axes([1.01, 0.05, 0.025, 0.95])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Temperature ($^oC$)', rotation=270,labelpad=+80,font=font, size = 85)
    
    # plt.suptitle('z = ' + str(z), y = -0.12, font = font, fontsize = fontsize)
    plt.tight_layout()
    
    plt.savefig('Temperature_z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()
    
    