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
T0 = np.load('case'+str(case1)+'/heat_solutions/' + str(e1)+'/'+'T_domain_'+str(e1)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')


case5 = 5
e5 = 71
T1 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

e5 = 71
T2 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_0.75_.npy')
T3 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_1.0_.npy')
T4 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_1.25_.npy')
T5 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_1.5_.npy')
T6 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_1.75_.npy')
T7 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_2.0_.npy')
T8 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_2.25_.npy')
T9 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_2.5_.npy')
T10 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_2.75_.npy')


dom = np.load('case1/2.5_dom.npy')

theta_ref = 35 - 20

nx, ny, nz = np.shape(T0)

theta2 = np.zeros((nx, ny, nz), dtype = float)

theta2[:,:,:] = 0 #-10
theta1 = np.copy(theta2)
theta3 = np.copy(theta2)
theta4 = np.copy(theta2)
theta5 = np.copy(theta2)
theta6 = np.copy(theta2)
theta7 = np.copy(theta2)
theta8 = np.copy(theta2)
theta9 = np.copy(theta2)
theta10 = np.copy(theta2)

delta_T1 = np.copy(theta2)
delta_T2 = np.copy(theta2)
delta_T3 = np.copy(theta2)
delta_T4 = np.copy(theta2)
delta_T5 = np.copy(theta2)
delta_T6 = np.copy(theta2)
delta_T7 = np.copy(theta2)
delta_T8 = np.copy(theta2)
delta_T9 = np.copy(theta2)
delta_T10 = np.copy(theta2)

for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if dom[i,j,k] == 0:
                theta1[i,j,k] = (T1[i,j,k] - T0[i,j,k])/theta_ref
                theta2[i,j,k] = (T2[i,j,k] - T0[i,j,k])/theta_ref
                theta3[i,j,k] = (T3[i,j,k] - T0[i,j,k])/theta_ref
                theta4[i,j,k] = (T4[i,j,k] - T0[i,j,k])/theta_ref
                theta5[i,j,k] = (T5[i,j,k] - T0[i,j,k])/theta_ref
                theta6[i,j,k] = (T6[i,j,k] - T0[i,j,k])/theta_ref
                theta7[i,j,k] = (T7[i,j,k] - T0[i,j,k])/theta_ref
                theta8[i,j,k] = (T8[i,j,k] - T0[i,j,k])/theta_ref
                theta9[i,j,k] = (T9[i,j,k] - T0[i,j,k])/theta_ref
                theta10[i,j,k] = (T10[i,j,k] - T0[i,j,k])/theta_ref
                
                delta_T1[i,j,k] = T1[i,j,k] - T0[i,j,k]
                delta_T2[i,j,k] = T2[i,j,k] - T0[i,j,k]
                delta_T3[i,j,k] = T3[i,j,k] - T0[i,j,k]
                delta_T4[i,j,k] = T4[i,j,k] - T0[i,j,k]
                delta_T5[i,j,k] = T5[i,j,k] - T0[i,j,k]
                delta_T6[i,j,k] = T6[i,j,k] - T0[i,j,k]
                delta_T7[i,j,k] = T7[i,j,k] - T0[i,j,k]
                delta_T8[i,j,k] = T8[i,j,k] - T0[i,j,k]
                delta_T9[i,j,k] = T9[i,j,k] - T0[i,j,k]
                delta_T10[i,j,k] = T10[i,j,k] - T0[i,j,k]



np.save('Temperature_error_5_1_E71_Kt0.5.npy', delta_T1)
np.save('Temperature_error_5_1_E71_Kt0.75.npy', delta_T2)
np.save('Temperature_error_5_1_E71_Kt1.0.npy', delta_T3)
np.save('Temperature_error_5_1_E71_Kt1.25.npy', delta_T4)
np.save('Temperature_error_5_1_E71_Kt1.5.npy', delta_T5)
np.save('Temperature_error_5_1_E71_Kt1.75.npy', delta_T6)
np.save('Temperature_error_5_1_E71_Kt2.0.npy', delta_T7)
np.save('Temperature_error_5_1_E71_Kt2.25.npy', delta_T8)
np.save('Temperature_error_5_1_E71_Kt2.5.npy', delta_T9)
np.save('Temperature_error_5_1_E71_Kt2.75.npy', delta_T10)

np.save('Theta_error_5_1_E71_Kt0.5.npy', theta1)
np.save('Theta_error_5_1_E71_Kt0.75.npy', theta2)
np.save('Theta_error_5_1_E71_Kt1.0.npy', theta3)
np.save('Theta_error_5_1_E71_Kt1.25.npy', theta4)
np.save('Theta_error_5_1_E71_Kt1.5.npy', theta5)
np.save('Theta_error_5_1_E71_Kt1.75.npy', theta6)
np.save('Theta_error_5_1_E71_Kt2.0.npy', theta7)
np.save('Theta_error_5_1_E71_Kt2.25.npy', theta8)
np.save('Theta_error_5_1_E71_Kt2.5.npy', theta9)
np.save('Theta_error_5_1_E71_Kt2.75.npy', theta10)

# np.save('Temperature_error_5_1_E71_Kt1.5.npy',delta_T2)
# np.save('Temperature_error_5_1_E71_Kt1.75.npy',delta_T3)
# np.save('Temperature_error_5_1_E71_Kt2.0.npy',delta_T4)
# np.save('Temperature_error_5_1_E71_Kt2.25.npy',delta_T5)

# np.save('Theta_error_5_1_E71_Kt1.5.npy', theta2)
# np.save('Theta_error_5_1_E71_Kt1.75.npy', theta3)
# np.save('Theta_error_5_1_E71_Kt2.0.npy', theta4)
# np.save('Theta_error_5_1_E71_Kt2.25.npy', theta5)

# print(np.max(theta2[:,:,20:61]), np.min(theta2[:,:,20:61]))
# print(np.max(theta3[:,:,20:61]), np.min(theta3[:,:,20:61]))
# print(np.max(theta4[:,:,20:61]), np.min(theta4[:,:,20:61]))
# print(np.max(theta5[:,:,20:61]), np.min(theta5[:,:,20:61]))

# print('\n')
# print(np.max(delta_T2[:,:,20:61]), np.min(delta_T2[:,:,20:61]))
# print(np.max(delta_T3[:,:,20:61]), np.min(delta_T3[:,:,20:61]))
# print(np.max(delta_T4[:,:,20:61]), np.min(delta_T4[:,:,20:61]))
# print(np.max(delta_T5[:,:,20:61]), np.min(delta_T5[:,:,20:61]))


'''


z_array = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]

dT_array = np.arange(28,37,1)
nrow = 1
ncol = 5
for z in z_array:
    plt.figure(figsize=(55,15), dpi = 300)
    color = 'Reds' 
    
    plt.subplot(nrow,ncol,1)
    plt.contourf(T_tis1[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 1 $\epsilon$ = 95 mm',y = -0.075 ,font=font)    
    
    
    plt.subplot(nrow, ncol, 2)
    plt.contourf(T_tis2[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 2 $\epsilon$ = 122.5 mm',y = -0.075 ,font=font) 
    
    plt.subplot(nrow, ncol, 3)
    plt.contourf(T_tis3[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 3 $\epsilon$ = 132.5 mm',y = -0.075 ,font=font) 
    
    plt.subplot(nrow, ncol, 4)
    plt.contourf(T_tis4[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 4 $\epsilon$ = 165 mm',y = -0.075,font=font) 
    
    
    plt.subplot(nrow, ncol, 5)
    plt.contourf(T_tis5[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 5 $\epsilon$ = 177.5 mm',y = -0.075,font=font) 
    
    
    cax = plt.axes([1.01, 0.075, 0.025, 0.85])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 40)
    cbar.set_label('Temperature $\degree C$', rotation=270,labelpad=+65,font=font)
    
    plt.suptitle(' z = ' + str(z) , y = -0.05, font = font, fontsize = 65)
    plt.tight_layout()
    
    plt.savefig('Heat_transfer_analysis_results/Temperature_z_' + str(z) + '.png', bbox_inches='tight', dpi = 300)
    plt.show()
    

'''

'''

z_array = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]

dT_array = np.arange(-0.2,0.24,0.04)
dT_array2 = np.arange(-2.8,3.2,0.4)

nrow = 2
ncol = 4



for z in z_array:
    plt.figure(figsize=(45,25))
    color = 'RdBu' # plt.cm.get_cmap('Reds') 
    
    plt.subplot(nrow,ncol,1)
    plt.contourf(theta2[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 2\n$\epsilon$ = 122.5 mm',y = 1.05,font=font)    
    
    plt.subplot(nrow,ncol,2)
    plt.contourf(theta3[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 3\n$\epsilon$ = 132.5 mm',y = 1.05,font=font) 
    
    plt.subplot(nrow, ncol, 3)
    plt.contourf(theta4[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 4\n$\epsilon$ = 165 mm',y = 1.05,font=font) 
    
    plt.subplot(nrow, ncol, 4)
    plt.contourf(theta5[1:-1,1:-1,z],dT_array,cmap=color)
    plt.axis('off')
    plt.title('case 5\n$\epsilon$ = 177.5 mm',y = 1.05,font=font) 
    
    
    cax = plt.axes([1.00, 0.45, 0.022, 0.42])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 40)
    cbar.set_label('Dimensionless Temperature Error', rotation=270,labelpad=+65,font=font)
    
    
    
    color = 'RdBu' # plt.cm.get_cmap('Reds') 
    
    plt.suptitle('z = ' + str(z) ,y = -0.04,font=font , fontsize = 65)
    
    
    
    plt.subplot(nrow, ncol, 5)
    plt.contourf(delta_T2[1:-1,1:-1,z],dT_array2,cmap=color)
    plt.axis('off')
    # plt.title('z = ' + str(z) ,y = -0.1,font=font)    
    
    
    plt.subplot(nrow, ncol, 6)
    plt.contourf(delta_T3[1:-1,1:-1,z],dT_array2,cmap=color)
    plt.axis('off')
    # plt.title('z = ' + str(z) ,y = -0.1,font=font)    
    
    plt.subplot(nrow, ncol, 7)
    plt.contourf(delta_T4[1:-1,1:-1,z],dT_array2,cmap=color)
    plt.axis('off')
    # plt.title('z = ' + str(z) ,y = -0.1,font=font)    
    
    plt.subplot(nrow, ncol, 8)
    plt.contourf(delta_T5[1:-1,1:-1,z],dT_array2,cmap=color)
    plt.axis('off')
    # plt.title('z = ' + str(z) ,y = -0.1,font=font)    
    
    
    cax = plt.axes([1.00, 0.01, 0.022, 0.42])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 40)
    cbar.set_label('Temperature Error ($\degree C$)', rotation=270,labelpad=+65,font=font)
    
    
    plt.tight_layout()
    plt.savefig('Heat_transfer_analysis_results/analysis_1_layer_'+str(z) + '.png', dpi = 100, bbox_inches = 'tight')
    plt.show()
    
'''

'''


z_array = [1,20,40,60,80]

for k in z_array:
    D2 = []
    D3 = []
    D4 = []
    D5 = []
    for j in range(ny):
        for i in range(nx):
            if(dom1[i,j,k] == 0):
                d2 = theta2[i,j,k]
                d3 = theta3[i,j,k]
                d4 = theta4[i,j,k]
                d5 = theta5[i,j,k]
                
                D2.append(d2)
                D3.append(d3)
                D4.append(d4)
                D5.append(d5)
    
    x = np.arange(0,len(D2),1)
    # plt.figure(figsize = (20,10), dpi = 300)
    # plt.scatter(x,D2, color = 'red')
    # plt.scatter(x,D3, color = 'green')
    # plt.scatter(x,D4, color = 'purple')
    # plt.scatter(x,D5, color = 'black')
    # plt.title('z = '+ str(k), font = font, fontsize = 40)
    # plt.yticks(np.arange(-0.07,0.025,0.005))
    # plt.show()
    
    db = np.zeros((len(x), 4), dtype = float)
    db[:,0] = D2[:]
    db[:,1] = D3[:]
    db[:,2] = D4[:]
    db[:,3] = D5[:]
    
    x_min = min(min(D2), min(D3), min(D4), min(D5))
    x_max = max(max(D2), max(D3), max(D4), max(D5))
    
    x_max = max(abs(x_min), abs(x_max))
    
    dx_axis = (2*x_max)/ 10
    dx_axis = np.arange(-x_max, x_max + dx_axis, dx_axis)
    
    db = pd.DataFrame((db))
    
    colors = sb.color_palette()
    
    plt.figure(figsize = (10,8), dpi = 300)
    
    g = sb.histplot(D2, element = 'poly', color = colors[0], alpha = 1.0)
    g = sb.histplot(D3, element = 'poly', color = colors[1], alpha = 0.8)    
    g = sb.histplot(D4, element = 'poly', color = colors[2], alpha = 0.6)
    g = sb.histplot(D5, element = 'poly', color = colors[3], alpha = 0.4)
    
    plt.legend( labels = ['case 2', 'case 3','case 4', 'case 5'])
    plt.xticks(dx_axis)
    
    plt.title('z = ' + str(k), font = font, fontsize = 20)
    plt.show()



'''

# data = []
# D2 = np.empty(31634)
# D3 = np.empty(31634)
# D4 = np.empty(31634)
# D5 = np.empty(31634)

# z_array = [1,20,40,60,80]
# count = 0

# x = np.arange(0,31634,1)
# for k in z_array:
#     for j in range(ny):
#         for i in range(nx):
#             if(dom1[i,j,k] == 0):
#                 d2 = theta2[i,j,k]
#                 d3 = theta3[i,j,k]
#                 d4 = theta4[i,j,k]
#                 d5 = theta5[i,j,k]
                
#                 data.append([d2,d3,d4,d5])
                
#                 D2[count] = d2
#                 D3[count] = d3
#                 D4[count] = d4
#                 D5[count] = d5
                
#                 count = count + 1




# plt.figure(figsize = (20,5), dpi = 300)
# plt.plot(x,D2, color = 'red')
# plt.plot(x,D3, color = 'green')
# plt.plot(x,D4, color = 'purple')
# plt.plot(x,D5, color = 'black')
# plt.show()

# plt.figure(figsize = (20,5), dpi = 300)
# plt.scatter(x,D2, color = 'red')
# plt.scatter(x,D3, color = 'green')
# plt.scatter(x,D4, color = 'purple')
# plt.scatter(x,D5, color = 'black')
# plt.show()


# plt.figure(figsize = (10,10), dpi = 300)
# for i in range(1000):#len(data)):
    
#     d = data[i]
#     plt.plot(d)

# plt.show()