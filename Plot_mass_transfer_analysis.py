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

c1 = 1
e1 = E_array[c1]
ma1 = np.load('case_'+str(c1)+'_E_' + str(e1) + '_art_comp_mass.npy')
mv1 = np.load('case_'+str(c1)+'_E_' + str(e1) + '_art_comp_mass.npy')
mav1 = np.load('case_'+str(c1)+'_E_' + str(e1) + '_art_ven_comp_mass.npy')

c2 = 2
e2 = E_array[c2]
ma2 = np.load('case_'+str(c2)+'_E_' + str(e2) + '_art_comp_mass.npy')
mv2 = np.load('case_'+str(c2)+'_E_' + str(e2) + '_art_comp_mass.npy')
mav2 = np.load('case_'+str(c2)+'_E_' + str(e2) + '_art_ven_comp_mass.npy')

c3 = 3
e3 = E_array[c3]
ma3 = np.load('case_'+str(c3)+'_E_' + str(e3) + '_art_comp_mass.npy')
mv3 = np.load('case_'+str(c3)+'_E_' + str(e3) + '_art_comp_mass.npy')
mav3 = np.load('case_'+str(c3)+'_E_' + str(e3) + '_art_ven_comp_mass.npy')

c4 = 4
e4 = E_array[c4]
ma4 = np.load('case_'+str(c4)+'_E_' + str(e4) + '_art_comp_mass.npy')
mv4 = np.load('case_'+str(c4)+'_E_' + str(e4) + '_art_comp_mass.npy')
mav4 = np.load('case_'+str(c4)+'_E_' + str(e4) + '_art_ven_comp_mass.npy')

c5 = 5
e5 = E_array[c5]
ma5 = np.load('case_'+str(c5)+'_E_' + str(e5) + '_art_comp_mass.npy')
mv5 = np.load('case_'+str(c5)+'_E_' + str(e5) + '_art_comp_mass.npy')
mav5 = np.load('case_'+str(c5)+'_E_' + str(e5) + '_art_ven_comp_mass.npy')

zero = 0

ma1[:,:,0] = ma1[:,:,-1] = zero
ma1[:,0,:] = ma1[:,-1,:] = zero
ma1[0,:,:] = ma1[-1,:,:] = zero

mv1[:,:,0] = mv1[:,:,-1] = zero
mv1[:,0,:] = mv1[:,-1,:] = zero
mv1[0,:,:] = mv1[-1,:,:] = zero

ema2 = (ma2-ma1)#/ma1*100
emv2 = (mv2-mv1)#/mv1*100

ema3 = (ma3-ma1)#/ma1*100
emv3 = (mv3-mv1)#/mv1*100

ema4 = (ma4-ma1)#/ma1*100
emv4 = (mv4-mv1)#/mv1*100

ema5 = (ma5-ma1)#/ma1*100
emv5 = (mv5-mv1)#/mv1*100


max_err_ac = max(np.max(ema2), np.max(ema3), np.max(ema4), np.max(ema5))
min_err_ac = min(np.min(ema2), np.min(ema3), np.min(ema4), np.min(ema5))

max_err_vc = max(np.max(emv2), np.max(emv3), np.max(emv4), np.max(emv5))
min_err_vc = min(np.min(emv2), np.min(emv3), np.min(emv4), np.min(emv5))

max_of_two_ends = max(abs(max_err_ac), abs(min_err_ac))

d_err_ac = 2*max_of_two_ends/30
d_err_vc = d_err_ac

err_a_bar = np.arange(-max_of_two_ends, max_of_two_ends + d_err_ac, d_err_ac)
err_v_bar = err_a_bar

'''
d_err_ac = (max_err_ac - min_err_ac)/30
d_err_vc = (max_err_vc - min_err_vc)/30

err_a_bar = np.arange(min_err_ac, max_err_ac + d_err_ac, d_err_ac)
err_v_bar = np.arange(min_err_vc, max_err_vc + d_err_vc, d_err_vc)
'''
z_array = [1,20,40,60,80]

colors = 'RdBu'
for z in z_array:
    plt.figure(figsize = (45,25),dpi = 100)
    plt.subplot(2,4,1)
    plt.contourf(ema2[0:-1,0:-1,z], err_a_bar, cmap=colors)
    plt.axis('off')
    
    plt.subplot(2,4,2)
    plt.contourf(ema3[0:-1,0:-1,z], err_a_bar, cmap=colors)
    plt.axis('off')
    
    plt.subplot(2,4,3)
    plt.contourf(ema4[0:-1,0:-1,z], err_a_bar, cmap=colors)
    plt.axis('off')
    
    plt.subplot(2,4,4)
    plt.contourf(ema5[0:-1,0:-1,z], err_a_bar, cmap=colors)
    plt.axis('off')
    
    # cax = plt.axes([1.01, 0.51, 0.022, 0.49])
    # cbar = plt.colorbar(cax=cax)
    # cbar.ax.tick_params(labelsize = 40)
    # # cbar.set_label('Arterial Advection\nEnergy Error (W)', rotation=270,labelpad=+115,font=font)
    
    
    plt.subplot(2,4,5)
    plt.contourf(emv2[0:-1,0:-1,z], err_v_bar, cmap=colors)
    plt.axis('off')
    
    plt.subplot(2,4,6)
    plt.contourf(emv3[0:-1,0:-1,z], err_v_bar, cmap=colors)
    plt.axis('off')
    
    plt.subplot(2,4,7)
    plt.contourf(emv4[0:-1,0:-1,z], err_v_bar, cmap=colors)
    plt.axis('off')
    
    plt.subplot(2,4,8)
    plt.contourf(emv5[0:-1,0:-1,z], err_v_bar, cmap=colors)
    plt.axis('off')
    
    cax = plt.axes([1.01, 0.01, 0.022, 0.95])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('mass flow difference', rotation=270,labelpad=+115,font=font, fontsize = 85)
    
    plt.suptitle('z = ' + str(z), y = 1.02, font=font, fontsize = 85)
    plt.tight_layout()
    plt.show()
    