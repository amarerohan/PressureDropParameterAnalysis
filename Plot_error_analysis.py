import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sb
import matplotlib.gridspec as gridspec

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(60)

vox_mass = 1000*(2.5e-3)**3
Cp = 1000 


dom = np.load('case1/2.5_dom.npy')
case = 1
e = 38

m1 = np.load('m_dot_'+str(e)+'_case_'+str(case)+'.npy')
t1 = np.load('t_sum_'+str(e)+'_case_'+str(case)+'.npy')
v1 = np.load('v_sum_'+str(e)+'_case_'+str(case)+'.npy')

case = 2
e = 49

m2 = np.load('m_dot_'+str(e)+'_case_'+str(case)+'.npy')
t2 = np.load('t_sum_'+str(e)+'_case_'+str(case)+'.npy')
v2 = np.load('v_sum_'+str(e)+'_case_'+str(case)+'.npy')

case = 3
e = 53

m3 = np.load('m_dot_'+str(e)+'_case_'+str(case)+'.npy')
t3 = np.load('t_sum_'+str(e)+'_case_'+str(case)+'.npy')
v3 = np.load('v_sum_'+str(e)+'_case_'+str(case)+'.npy')

case = 4
e = 66

m4 = np.load('m_dot_'+str(e)+'_case_'+str(case)+'.npy')
t4 = np.load('t_sum_'+str(e)+'_case_'+str(case)+'.npy')
v4 = np.load('v_sum_'+str(e)+'_case_'+str(case)+'.npy')

case = 5
e = 71

m5 = np.load('m_dot_'+str(e)+'_case_'+str(case)+'.npy')
t5 = np.load('t_sum_'+str(e)+'_case_'+str(case)+'.npy')
v5 = np.load('v_sum_'+str(e)+'_case_'+str(case)+'.npy')


dm2 = np.copy(m1)
dm2[:,:,:] = 0.0 #-100.0

dm3 = np.copy(dm2)
dm4 = np.copy(dm2)
dm5 = np.copy(dm2)

dt2 = np.copy(dm2)
dt3 = np.copy(dm2)
dt4 = np.copy(dm2)
dt5 = np.copy(dm2)

dv2 = np.copy(dm2)
dv3 = np.copy(dm2)
dv4 = np.copy(dm2)
dv5 = np.copy(dm2)




nx, ny, nz = np.shape(dm2)

for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 0):
                dm2[i,j,k] = m2[i,j,k] - m1[i,j,k]
                dt2[i,j,k] = t2[i,j,k] - t1[i,j,k]
                dv2[i,j,k] = v2[i,j,k] - v1[i,j,k]
                
                dm3[i,j,k] = m3[i,j,k] - m1[i,j,k]
                dt3[i,j,k] = t3[i,j,k] - t1[i,j,k]
                dv3[i,j,k] = v3[i,j,k] - v1[i,j,k]
                
                dm4[i,j,k] = m4[i,j,k] - m1[i,j,k]
                dt4[i,j,k] = t4[i,j,k] - t1[i,j,k]
                dv4[i,j,k] = v4[i,j,k] - v1[i,j,k]
                
                dm5[i,j,k] = m5[i,j,k] - m1[i,j,k]
                dt5[i,j,k] = t5[i,j,k] - t1[i,j,k]
                dv5[i,j,k] = v5[i,j,k] - v1[i,j,k]
                

case1 = 1
e1 = 38
T_tis1 = np.load('case'+str(case1)+'/heat_solutions/' + str(e1)+'/'+'T_domain_'+str(e1)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
dom1 = np.load('case'+str(case1)+'/2.5_dom.npy')

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


e1 = np.copy(dm2)
e1[:,:,:] = 0.0
e2 = np.copy(e1)
e3 = np.copy(e2)
e4 = np.copy(e2)
e5 = np.copy(e2)

de2 = np.copy(e1)
de3 = np.copy(de2)
de4 = np.copy(de3)
de5 = np.copy(de3)



for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if dom[i,j,k] == 0:
                e1[i,j,k] = T_tis1[i,j,k]*vox_mass*Cp
                e2[i,j,k] = T_tis2[i,j,k]*vox_mass*Cp
                e3[i,j,k] = T_tis3[i,j,k]*vox_mass*Cp
                e4[i,j,k] = T_tis4[i,j,k]*vox_mass*Cp
                e5[i,j,k] = T_tis5[i,j,k]*vox_mass*Cp
                
                de2[i,j,k] = e2[i,j,k] - e1[i,j,k]
                de3[i,j,k] = e3[i,j,k] - e1[i,j,k]
                de4[i,j,k] = e4[i,j,k] - e1[i,j,k]
                de5[i,j,k] = e5[i,j,k] - e1[i,j,k]
                
           
                



z_array = [1,20,40,60,80] #[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]

dm_array = np.arange(-0.7,0.8,0.1)
dt_array = np.arange(-4.5,4.6,0.1)
dv_array = np.arange(-10E-3,11E-3,10E-4)
colors = 'RdBu'

rows = 3
cols = 4

font.set_size(80)

# suptitle = ['direct advection Energy from supply arterial terminal', 'convection from neighboring vasculature', 'advection and conduction from neighboring tissue']
for z in z_array:


    plt.figure(figsize = (42,34), dpi = 100)
    # plt.text(s = 'Advection from vasculature error at z = ' + str(z), x = 0.25, y = 0.6, font = font, fontsize = 60)
    plt.subplot(3,4,1)
    plt.contourf(dm2[:,:,z], dm_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 2', y = 1.05, font=font)
    
    plt.subplot(3,4,2)
    plt.contourf(dm3[:,:,z], dm_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 3', y = 1.05, font=font)
    
    plt.subplot(3,4,3)
    plt.contourf(dm4[:,:,z], dm_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 4', y = 1.05, font=font)
    
    plt.subplot(3,4,4)
    plt.contourf(dm5[:,:,z], dm_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 5', y = 1.05, font=font)
    
    cax = plt.axes([1.00, 0.65, 0.022, 0.3])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Arterial Advection\nEnergy Error (W)', rotation=270,labelpad=+165,font=font)
    
    # plt.tight_layout()
    # plt.show()
    
    
    
    # plt.figure(figsize = (42,12), dpi = 300)
    # plt.suptitle('convection from Neighbor vessel at z = ' + str(z), y = -0.05, font = font, fontsize = 60)
    plt.subplot(3,4,5)
    plt.contourf(dv2[:,:,z], dv_array, cmap = colors)
    plt.axis('off')
    # plt.title('case 2', y = 1.05, font=font)
    
    plt.subplot(3,4,6)
    plt.contourf(dv3[:,:,z], dv_array, cmap = colors)
    plt.axis('off')
    # plt.title('case 3', y = 1.05, font=font)
    
    plt.subplot(3,4,7)
    plt.contourf(dv4[:,:,z], dv_array, cmap = colors)
    plt.axis('off')
    # plt.title('case 4', y = 1.05, font=font)
    
    plt.subplot(3,4,8)
    plt.contourf(dv5[:,:,z], dv_array, cmap = colors)
    plt.axis('off')
    # plt.title('case 5', y = 1.05, font=font)
    
    cax = plt.axes([1.00, 0.33, 0.022, 0.29])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Vessel Convection\nEnergy Error (W)', rotation=270,labelpad=+165,font=font)
    
    # plt.tight_layout()
    # plt.show()
    
    
    # plt.figure(figsize = (42,12), dpi = 300)
    # plt.suptitle('Advection and conduction from Neighbor error at z = ' + str(z), y = -0.05, font = font, fontsize = 60)
    plt.subplot(3,4,9)
    plt.contourf(dt2[:,:,z], dt_array, cmap = colors)
    plt.axis('off')
    # plt.title('case 2', y = 1.05, font=font)
    
    plt.subplot(3,4,10)
    plt.contourf(dt3[:,:,z], dt_array, cmap = colors)
    plt.axis('off')
    # plt.title('case 3', y = 1.05, font=font)
    
    plt.subplot(3,4,11)
    plt.contourf(dt4[:,:,z], dt_array, cmap = colors)
    plt.axis('off')
    # plt.title('case 4', y = 1.05, font=font)
    
    plt.subplot(3,4,12)
    plt.contourf(dt5[:,:,z], dt_array, cmap = colors)
    plt.axis('off')
    # plt.title('case 5', y = 1.05, font=font)
    
    cax = plt.axes([1.00, 0.02, 0.022, 0.29])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Neighbor tissue\nEnergy Error (W)', rotation=270,labelpad=+165,font=font)
    
    # plt.suptitle('z = ' + str(z), y = -0.025, font = font, fontsize = 65)
    plt.tight_layout()
    plt.savefig('Heat_transfer_analysis_results/Error_Analysis_plots_z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()

    
'''
# Method 1

dm2p = np.copy(dm2)
dv2p = np.copy(dv2)
dt2p = np.copy(dt2)

dm3p = np.copy(dm3)
dv3p = np.copy(dv3)
dt3p = np.copy(dt3)

dm4p = np.copy(dm4)
dv4p = np.copy(dv4)
dt4p = np.copy(dt4)

dm5p = np.copy(dm5)
dv5p = np.copy(dv5)
dt5p = np.copy(dt5)


for j in range(nz):
    for k in range(ny):
        for i in range(nx):
            if dom[i,j,k] == 0:
                dm2p[i,j,k] = dm2[i,j,k]/m1[i,j,k]*100
                dt2p[i,j,k] = dt2[i,j,k]/t1[i,j,k]*100
                dv2p[i,j,k] = dv2[i,j,k]/v1[i,j,k]*100
                
                dm3p[i,j,k] = dm3[i,j,k]/m1[i,j,k]*100
                dt3p[i,j,k] = dt3[i,j,k]/t1[i,j,k]*100
                dv3p[i,j,k] = dv3[i,j,k]/v1[i,j,k]*100
                
                dm4p[i,j,k] = dm4[i,j,k]/m1[i,j,k]*100
                dt4p[i,j,k] = dt4[i,j,k]/t1[i,j,k]*100
                dv4p[i,j,k] = dv4[i,j,k]/v1[i,j,k]*100
                
                dm5p[i,j,k] = dm5[i,j,k]/m1[i,j,k]*100
                dt5p[i,j,k] = dt5[i,j,k]/t1[i,j,k]*100
                dv5p[i,j,k] = dv5[i,j,k]/v1[i,j,k]*100
                

# Method 2

dm2p2 = np.copy(dm2)
dv2p2 = np.copy(dv2)
dt2p2 = np.copy(dt2)

dm3p2 = np.copy(dm3)
dv3p2 = np.copy(dv3)
dt3p2 = np.copy(dt3)

dm4p2 = np.copy(dm4)
dv4p2 = np.copy(dv4)
dt4p2 = np.copy(dt4)

dm5p2 = np.copy(dm5)
dv5p2 = np.copy(dv5)
dt5p2 = np.copy(dt5)

energy_sum = np.zeros((nx, ny, nz), dtype = float)

for j in range(nz):
    for k in range(ny):
        for i in range(nx):
            if dom[i,j,k] == 0:
                
                e_sum = m1[i,j,k] + t1[i,j,k] + v1[i,j,k]
                energy_sum[i,j,k] = e_sum
                
                # print(m5[i,j,k] , energy_sum, m5[i,j,k]/energy_sum*100, '%')
                
                # if m5[i,j,k]/energy_sum*100 > 50:
                #     break
                
                # dm2p2[i,j,k] = m2[i,j,k]/energy_sum*100
                # dt2p2[i,j,k] = t2[i,j,k]/energy_sum*100
                # dv2p2[i,j,k] = v2[i,j,k]/energy_sum*100
                
                # dm3p2[i,j,k] = m3[i,j,k]/energy_sum*100
                # dt3p2[i,j,k] = t3[i,j,k]/energy_sum*100
                # dv3p2[i,j,k] = v3[i,j,k]/energy_sum*100
                
                # dm4p2[i,j,k] = m4[i,j,k]/energy_sum*100
                # dt4p2[i,j,k] = t4[i,j,k]/energy_sum*100
                # dv4p2[i,j,k] = v4[i,j,k]/energy_sum*100
                
                # dm5p2[i,j,k] = m5[i,j,k]/energy_sum*100
                # dt5p2[i,j,k] = t5[i,j,k]/energy_sum*100
                # dv5p2[i,j,k] = v5[i,j,k]/energy_sum*100
                
'''
