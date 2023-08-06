import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(60)

# T1 = np.load('case1/heat_solutions/38/T_domain_38_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T2 = np.load('case2/heat_solutions/49/T_domain_49_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T3 = np.load('case3/heat_solutions/49/T_domain_49_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T4 = np.load('case4/heat_solutions/66/T_domain_66_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T5 = np.load('case5/heat_solutions/71/T_domain_71_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')


# T1 = np.load('case1/heat_solutions/10/T_domain_10_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T2 = np.load('case2/heat_solutions/10/T_domain_10_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T3 = np.load('case3/heat_solutions/10/T_domain_10_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T4 = np.load('case4/heat_solutions/10/T_domain_10_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T5 = np.load('case5/heat_solutions/10/T_domain_10_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')


T1 = np.load('case1/heat_solutions/10.5/T_domain_10.5_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
T2 = np.load('case2/heat_solutions/14/T_domain_14_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
T3 = np.load('case3/heat_solutions/14/T_domain_14_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
T4 = np.load('case4/heat_solutions/22/T_domain_22_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
# T5 = np.load('case5/heat_solutions/29/T_domain_29_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')



dom = np.load('case1/2.5_dom.npy')

nx, ny, nz = np.shape(dom)

e21 = np.zeros((nx,ny,nz),dtype = float)
e31 = np.zeros((nx,ny,nz),dtype = float)
e41 = np.zeros((nx,ny,nz),dtype = float)
e51 = np.zeros((nx,ny,nz),dtype = float)

e21[:,:,:] = e31[:,:,:] = e41[:,:,:] = e51[:,:,:] = 100

for z in range(nz):
    for y in range(ny):
        for x in range(nx):
            if(dom[x,y,z] == 0):
                e21[x,y,z] = T2[x,y,z] - T1[x,y,z]
                e31[x,y,z] = T3[x,y,z] - T1[x,y,z]
                e41[x,y,z] = T4[x,y,z] - T1[x,y,z]
                # e51[x,y,z] = T5[x,y,z] - T1[x,y,z]
                


e_min = -4 #min(np.min(e21), np.min(e31), np.min(e41), np.min(e51))
e_max = 4 #max(np.max(e21), np.max(e31), np.max(e41), np.max(e51))
de = (e_max - e_min)/50

for z in range(1,nz-1):
    plt.figure(figsize=(20,22))
    color = 'RdBu' # plt.cm.get_cmap('Reds') 
    
    plt.subplot(221)
    
    plt.contourf(e21[1:-1,1:-1,z],np.arange(e_min,e_max+de,de),cmap=color)#.reversed())
    # plt.colorbar()
    plt.axis('off')
    plt.title('E21  z = '+ str(z),y = -0.05,font=font)    
    
    
    plt.subplot(222)
    plt.contourf(e31[1:-1,1:-1,z],np.arange(e_min,e_max+de,de),cmap=color)#.reversed())
    # plt.colorbar()
    plt.axis('off')
    plt.title('E31 z = '+ str(z),y = -0.05,font=font)
    
    plt.subplot(223)
    plt.contourf(e31[1:-1,1:-1,z],np.arange(e_min,e_max+de,de),cmap=color)#.reversed())
    # plt.colorbar()
    plt.axis('off')
    plt.title('E41  z = '+ str(z),y = -0.05,font=font)
    
    # plt.subplot(224)
    # plt.contourf(e31[1:-1,1:-1,z],np.arange(e_min,e_max+de,de),cmap=color)#.reversed())
    # # plt.colorbar()
    # plt.axis('off')
    # plt.title('E51  z = '+ str(z),y = -0.05,font=font)
    
    cax = plt.axes([1.00, 0.05, 0.022, 0.9])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 40)#FontProperties=font)# ,labelsize = 35)
    cbar.set_label('Temperature Error', rotation=270,labelpad=+65,font=font)
    
    
    plt.tight_layout()
    plt.show()
    
    


e_min = 30 # min(np.min(e21), np.min(e31), np.min(e41), np.min(e51))
e_max = 35.05 # max(np.max(e21), np.max(e31), np.max(e41), np.max(e51))
de = (e_max - e_min)/50

for z in range(1,nz-1):
    plt.figure(figsize=(20,22))
    color = 'Reds' # plt.cm.get_cmap('Reds') 
    
    plt.subplot(221)
    
    plt.contourf(T1[1:-1,1:-1,z],np.arange(e_min,e_max+de,de),cmap=color)#.reversed())
    # plt.colorbar()
    plt.axis('off')
    plt.title('E21  z = '+ str(z),y = -0.05,font=font)    
    
    
    plt.subplot(222)
    plt.contourf(T2[1:-1,1:-1,z],np.arange(e_min,e_max+de,de),cmap=color)#.reversed())
    # plt.colorbar()
    plt.axis('off')
    plt.title('E31 z = '+ str(z),y = -0.05,font=font)
    
    plt.subplot(223)
    plt.contourf(T4[1:-1,1:-1,z],np.arange(e_min,e_max+de,de),cmap=color)#.reversed())
    # plt.colorbar()
    plt.axis('off')
    plt.title('E41  z = '+ str(z),y = -0.05,font=font)
    
    # plt.subplot(224)
    # plt.contourf(T5[1:-1,1:-1,z],np.arange(e_min,e_max+de,de),cmap=color)#.reversed())
    # # plt.colorbar()
    # plt.axis('off')
    # plt.title('E51  z = '+ str(z),y = -0.05,font=font)
    
    cax = plt.axes([1.00, 0.05, 0.022, 0.9])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 40)#FontProperties=font)# ,labelsize = 35)
    cbar.set_label('Temperature Error', rotation=270,labelpad=+65,font=font)
    
    
    plt.tight_layout()
    plt.show()