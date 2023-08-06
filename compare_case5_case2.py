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


c = 1
e1 = 38
T0 = np.load('case'+str(c)+'/heat_solutions/' + str(e1)+'/'+'T_domain_'+str(e1)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

c = 5
e = 90

kt = 0.5

# if kt == 0.5:
#     T1 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# else:

T3 = np.load('case'+str(5)+'/heat_solutions/' + str(71)+'/'+'T_domain_'+str(71)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
dT3 = np.load('Temperature_error_5_1_E'+str(71)+'_Kt'+str(kt)+'.npy')

T1 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')
dT1 = np.load('Temperature_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')
# th1 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')

kt = 0.5
c = 2
e = 49
T2 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
dT2 = np.load('Temperature_error_2_1.npy')
# th2 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')


dT_max = max( np.max(np.abs(dT1)), np.max(np.abs(dT2)), np.max(np.abs(dT3)) )

z = 80
f = 70

N = 12

dT_array = np.arange(-dT_max, dT_max + 2*dT_max/N, 2*dT_max/N)

z_array = [1,40,80]

for z in z_array:
    plt.figure(figsize = (35,12), dpi = 100)
    plt.subplot(1,3,1)
    plt.contourf(dT2[0:-1,0:-1,z], dT_array, cmap = 'RdBu')
    plt.axis('off')
    plt.title('Case 2 $\epsilon = $' + '122.5 mm', y = -0.085, font = font, fontsize = f)
    
    plt.subplot(1,3,2)
    plt.contourf(dT3[0:-1,0:-1,z], dT_array, cmap = 'RdBu')
    plt.axis('off')
    plt.title('Case 5 $\epsilon = $' + '177.5 mm' , y = -0.085, font = font, fontsize = f)
    
    plt.subplot(1,3,3)
    plt.contourf(dT1[0:-1,0:-1,z], dT_array, cmap = 'RdBu')
    plt.axis('off')
    plt.title('Case 5 $\epsilon = $' + '237.5 mm' , y = -0.085, font = font, fontsize = f)
    
    cax = plt.axes([1.00, 0.01, 0.022, 0.98])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('$\Delta T_t$'+' ($^oC$)', rotation=270,labelpad=+70,font=font, fontsize = 75)
    
    plt.tight_layout()
    
    plt.savefig('New_Temperature_plots/Compare_case2_with_case_5_z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()