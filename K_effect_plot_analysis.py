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
e = 71

kt = 0.5

if kt == 0.5:
    T1 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
else:
    T1 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')
dT1 = np.load('Temperature_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')
th1 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')

kt = 0.75
T2 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')
dT2 = np.load('Temperature_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')
th2 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')

kt = 1.0
T3 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')
dT3 = np.load('Temperature_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')
th3 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')

kt = 1.25
T4 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')
dT4 = np.load('Temperature_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')
th4 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')

kt = 1.5
T5 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')
dT5 = np.load('Temperature_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')
th5 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')

kt = 1.75
T6 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')
dT6 = np.load('Temperature_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')
th6 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')

kt = 2.0
T7 = np.load('case'+str(c)+'/heat_solutions/' + str(e)+'/'+'T_domain_'+str(e)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')
dT7 = np.load('Temperature_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')
th7 = np.load('Theta_error_5_1_E'+str(e)+'_Kt'+str(kt)+'.npy')


dT_max = max(np.max(np.abs(dT1)), np.max(np.abs(dT2)), np.max(np.abs(dT3)), 
             np.max(np.abs(dT4)), np.max(np.abs(dT5)), np.max(np.abs(dT6)), np.max(np.abs(dT7)))
             
th_max = max(np.max(np.abs(th1)), np.max(np.abs(th2)), np.max(np.abs(th3)), 
             np.max(np.abs(th4)), np.max(np.abs(th5)), np.max(np.abs(th6)), np.max(np.abs(th7)))             
             

f = 70

N = 12

dT_array = np.arange(-dT_max, dT_max + 2*dT_max/N, 2*dT_max/N)
th_array = np.arange(-th_max, th_max + 2*th_max/N, 2*th_max/N)


colors = 'RdBu'


z_array = [1,20,40,60,80]

for z in z_array:
    plt.figure(figsize = (45,25), dpi = 200)
    
    
    plt.subplot(2,4,1)
    plt.contourf(dT1[0:-1,0:-1,z], dT_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 0.5 ' + '$W m^{-1o}C^{-1}$', y = -0.085, fontsize = f, font = font)
    
    plt.subplot(2,4,2)
    plt.contourf(dT2[0:-1,0:-1,z], dT_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 0.75 ' + '$W m^{-1o}C^{-1}$', y = -0.085, fontsize = f, font = font)
    
    plt.subplot(2,4,3)
    plt.contourf(dT3[0:-1,0:-1,z], dT_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 1.0 ' + '$W m^{-1o}C^{-1}$', y = -0.085, fontsize = f, font = font)
    
    plt.subplot(2,4,4)
    plt.contourf(dT4[0:-1,0:-1,z], dT_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 1.25 ' + '$W m^{-1o}C^{-1}$', y = -0.085, fontsize = f, font = font)
    
    plt.subplot(2,4,5)
    plt.contourf(dT5[0:-1,0:-1,z], dT_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 1.5 ' + '$W m^{-1o}C^{-1}$', y = -0.085, fontsize = f, font = font)
    
    plt.subplot(2,4,6)
    plt.contourf(dT6[0:-1,0:-1,z], dT_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 1.75 ' + '$W m^{-1o}C^{-1}$', y = -0.085, fontsize = f, font = font)
    
    plt.subplot(2,4,7)
    plt.contourf(dT7[0:-1,0:-1,z], dT_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 2.0 ' + '$W m^{-1o}C^{-1}$', y = -0.085, fontsize = f, font = font)
    
    cax = plt.axes([1.00, 0.01, 0.022, 0.98])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Temperature difference ($^oC$)', rotation=270,labelpad=+70,font=font, fontsize = 75)
    
    plt.tight_layout()
    
    plt.savefig('New_Temperature_plots/Effect_of_Kt_for_SoI_' + str(e) + '_z_' + str(z) + '.png', dpi = 100, bbox_inches='tight')
    
    plt.show()
    
    
'''   
for z in z_array:
    plt.figure(figsize = (45,25), dpi = 200)
    
    
    plt.subplot(2,4,1)
    plt.contourf(th1[0:-1,0:-1,z], th_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 0.5 ' + '$W m^{-1o}C^{-1}$', y = -0.075, fontsize = f, font = font)
    
    plt.subplot(2,4,2)
    plt.contourf(th2[0:-1,0:-1,z], th_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 0.75 ' + '$W m^{-1o}C^{-1}$', y = -0.075, fontsize = f, font = font)
    
    plt.subplot(2,4,3)
    plt.contourf(th3[0:-1,0:-1,z], th_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 1.0 ' + '$W m^{-1o}C^{-1}$', y = -0.075, fontsize = f, font = font)
    
    plt.subplot(2,4,4)
    plt.contourf(th4[0:-1,0:-1,z], th_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 1.25 ' + '$W m^{-1o}C^{-1}$', y = -0.075, fontsize = f, font = font)
    
    plt.subplot(2,4,5)
    plt.contourf(th5[0:-1,0:-1,z], th_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 1.5 ' + '$W m^{-1o}C^{-1}$', y = -0.075, fontsize = f, font = font)
    
    plt.subplot(2,4,6)
    plt.contourf(th6[0:-1,0:-1,z], th_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 1.75 ' + '$W m^{-1o}C^{-1}$', y = -0.075, fontsize = f, font = font)
    
    plt.subplot(2,4,7)
    plt.contourf(th7[0:-1,0:-1,z], th_array, cmap = colors)
    plt.axis('off')
    plt.title('$k_t$ = 2.0 ' + '$W m^{-1o}C^{-1}$', y = -0.075, fontsize = f, font = font)
    
    cax = plt.axes([1.00, 0.01, 0.022, 0.98])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('$\theta$', rotation=270,labelpad=+70,font=font, fontsize = 75)
    
    plt.tight_layout()
    
    plt.show()
'''    
