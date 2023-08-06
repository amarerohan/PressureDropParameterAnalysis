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

'''
case1 = 1
e1 = 38
T_tis1 = np.load('case'+str(case1)+'/heat_solutions/' + str(e1)+'/'+'T_domain_'+str(e1)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

case2 = 5
e2 = 71
T_tis2 = np.load('case'+str(case2)+'/heat_solutions/' + str(e2)+'/'+'T_domain_'+str(e2)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

case3 = 5
e3 = 71
T_tis3 = np.load('case'+str(case3)+'/heat_solutions/' + str(e3)+'/'+'T_domain_'+str(e3)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_0.75_.npy')

case4 = 5
e4 = 71
T_tis4 = np.load('case'+str(case4)+'/heat_solutions/' + str(e4)+'/'+'T_domain_'+str(e4)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_1.0_.npy')

case5 = 5
e5 = 71
T_tis5 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_1.25_.npy')
'''


dT2 = np.load('Temperature_error_5_1_E71_Kt0.5.npy')
dT3 = np.load('Temperature_error_5_1_E71_Kt0.75.npy')
dT4 = np.load('Temperature_error_5_1_E71_Kt1.25.npy')
dT5 = np.load('Temperature_error_5_1_E71_Kt1.5.npy')

th2 = np.load('Theta_error_5_1_E71_Kt0.5.npy')
th3 = np.load('Theta_error_5_1_E71_Kt0.75.npy')
th4 = np.load('Theta_error_5_1_E71_Kt1.25.npy')
th5 = np.load('Theta_error_5_1_E71_Kt1.5.npy')


z_array = [1,20,40,60,80]

dT2_max = np.max(np.abs(dT2))
dT3_max = np.max(np.abs(dT3))
dT4_max = np.max(np.abs(dT4))
dT5_max = np.max(np.abs(dT5))

th2_max = np.max(np.abs(th2))
th3_max = np.max(np.abs(th3))
th4_max = np.max(np.abs(th4))
th5_max = np.max(np.abs(th5))

n = 12

# dT2_array = np.arange(-dT2_max, dT2_max + 2*dT2_max/n, 2*dT2_max/n)
# dT3_array = np.arange(-dT3_max, dT3_max + 2*dT3_max/n, 2*dT3_max/n)
# dT4_array = np.arange(-dT4_max, dT4_max + 2*dT4_max/n, 2*dT4_max/n)
# dT5_array = np.arange(-dT5_max, dT5_max + 2*dT5_max/n, 2*dT5_max/n)

# th2_array = np.arange(-th2_max, th2_max + 2*th2_max/n, 2*th2_max/n)
# th3_array = np.arange(-th3_max, th3_max + 2*th3_max/n, 2*th3_max/n)
# th4_array = np.arange(-th4_max, th4_max + 2*th4_max/n, 2*th4_max/n)
# th5_array = np.arange(-th5_max, th5_max + 2*th5_max/n, 2*th5_max/n)

dT_max = max(dT2_max, dT3_max, dT4_max, dT5_max)
th_max = max(th2_max, th3_max, th4_max, th5_max)

dT_array = np.arange(-dT_max, dT_max + 2*dT_max/n, 2*dT_max/n)
th_array = np.arange(-th_max, th_max + 2*th_max/n, 2*th_max/n)

for z in z_array:
    plt.figure(figsize=(50,25), dpi = 100)
    
    plt.subplot(2,4,1)
    plt.contourf(dT2[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('$k_t$ = 0.5 ' + '$Wm^{-1o}C^{-1}$', y = 1.01, font=font, fontsize = 70)
    
    
    plt.subplot(2,4,2)
    plt.contourf(dT3[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('$k_t$ = 0.75 ' + '$Wm^{-1o}C^{-1}$', y = 1.01, font=font, fontsize = 70)
    
    
    plt.subplot(2,4,3)
    plt.contourf(dT4[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('$k_t$ = 1.25 ' + '$Wm^{-1o}C^{-1}$', y = 1.01, font=font, fontsize = 70)
    
    plt.subplot(2,4,4)
    plt.contourf(dT5[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('$k_t$ = 1.5 ' + '$Wm^{-1o}C^{-1}$', y = 1.01, font=font, fontsize = 70)
    
    cax = plt.axes([1.00, 0.5, 0.022, 0.45])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Temperature Difference ($^oC$)', rotation=270,labelpad=+65,font=font)
    
    plt.subplot(2,4,5)
    plt.contourf(th2[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    # plt.title('(e)', y = -0.075, font=font)
    
    plt.subplot(2,4,6)
    plt.contourf(th3[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    # plt.title('(f)', y = -0.075, font=font)
    
    
    plt.subplot(2,4,7)
    plt.contourf(th4[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    # plt.title('(g)', y = -0.075, font=font)
    
    
    plt.subplot(2,4,8)
    plt.contourf(th5[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    # plt.title('(h)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.01, 0.022, 0.44])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('$\Theta$', rotation=270,labelpad=+65,font=font)
    
    plt.tight_layout()
    
    plt.savefig('New_Temperature_plots/Case5_E_71_Varying_Kt_5_75_125_15_z_'+str(z)+'.png',dpi=100, bbox_inches='tight')
    plt.show()
    
