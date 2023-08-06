import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sb
# sb.set(rc={'figure.figsize':(11.7,8.27)})
import pandas as pd

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(60)

case1 = 1
e1 = 38
pa1 = np.load('case'+str(case1)+'/flow_solutions/' + str(e1) + '/prs_acomp_' + str(e1) +'_Pin_1000_Pout_1.npy')
pv1 = np.load('case'+str(case1)+'/flow_solutions/' + str(e1) + '/prs_vcomp_' + str(e1) +'_Pin_1000_Pout_1.npy')
dom1 = np.load('case'+str(case1)+'/2.5_dom.npy')

case2 = 2
e2 = 49
pa2 = np.load('case'+str(case2)+'/flow_solutions/' + str(e2) + '/prs_acomp_' + str(e2) +'_Pin_1000_Pout_1.npy')
pv2 = np.load('case'+str(case2)+'/flow_solutions/' + str(e2) + '/prs_vcomp_' + str(e2) +'_Pin_1000_Pout_1.npy')

case3 = 3
e3 = 53
pa3 = np.load('case'+str(case3)+'/flow_solutions/' + str(e3) + '/prs_acomp_' + str(e3) +'_Pin_1000_Pout_1.npy')
pv3 = np.load('case'+str(case3)+'/flow_solutions/' + str(e3) + '/prs_vcomp_' + str(e3) +'_Pin_1000_Pout_1.npy')

case4 = 4
e4 = 66
pa4 = np.load('case'+str(case4)+'/flow_solutions/' + str(e4) + '/prs_acomp_' + str(e4) +'_Pin_1000_Pout_1.npy')
pv4 = np.load('case'+str(case4)+'/flow_solutions/' + str(e4) + '/prs_vcomp_' + str(e4) +'_Pin_1000_Pout_1.npy')

case5 = 5
e5 = 71
pa5 = np.load('case'+str(case5)+'/flow_solutions/' + str(e5) + '/prs_acomp_' + str(e5) +'_Pin_1000_Pout_1.npy')
pv5 = np.load('case'+str(case5)+'/flow_solutions/' + str(e5) + '/prs_vcomp_' + str(e5) +'_Pin_1000_Pout_1.npy')

z_array = [1,10,20,30,40,50,60,70,80]
p_array = np.arange(562,563.75,0.05)
colors = 'Blues'
fontsize  = 65

for z in z_array:
    plt.figure(figsize=(35,25), dpi = 100)
    
    plt.subplot(2,3,1)
    plt.contourf(pa1[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 1', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,2)
    plt.contourf(pa2[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 2', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,3)
    plt.contourf(pa3[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 3', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,4)
    plt.contourf(pa4[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 4', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,5)
    plt.contourf(pa5[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 5', y = -0.075, font = font, fontsize = fontsize)
    
    cax = plt.axes([1.01, 0.05, 0.025, 0.95])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Pressure (Pa)', rotation=270,labelpad=+75,font=font, size = 85)
    
    # plt.suptitle('z = ' + str(z), y = -0.12, font = font, fontsize = fontsize)
    plt.tight_layout()
    
    plt.savefig('Pressure_map_of_cases_z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()
    
    

p_array = np.arange(437.5,438.55,0.05)
for z in z_array:
    plt.figure(figsize=(35,25), dpi = 100)
    
    plt.subplot(2,3,1)
    plt.contourf(pv1[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 1', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,2)
    plt.contourf(pv2[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 2', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,3)
    plt.contourf(pv3[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 3', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,4)
    plt.contourf(pv4[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 4', y = -0.075, font = font, fontsize = fontsize)
    
    plt.subplot(2,3,5)
    plt.contourf(pv5[:,:,z], p_array, cmap = colors)
    plt.axis('off')
    plt.title('Case 5', y = -0.075, font = font, fontsize = fontsize)
    
    cax = plt.axes([1.01, 0.05, 0.025, 0.95])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Pressure (Pa)', rotation=270,labelpad=+75,font=font, size = 85)
    
    # plt.suptitle('z = ' + str(z), y = -0.12, font = font, fontsize = fontsize)
    plt.tight_layout()
    
    plt.savefig('Pressure_map_of_cases_Venous_compartment_z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()




# theta_ref = 35 - 20

nx, ny, nz = np.shape(pa1)

dPa2 = np.zeros((nx, ny, nz), dtype = float)
# dPa[:,:,:] = -10

dPa3 = np.copy(dPa2)
dPa4 = np.copy(dPa2)
dPa5 = np.copy(dPa2)

dPv2 = np.copy(dPa2)
dPv3 = np.copy(dPa2)
dPv4 = np.copy(dPa2)
dPv5 = np.copy(dPa2)




for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if dom1[i,j,k] == 0:
                dPa2[i,j,k] = pa2[i,j,k] - pa1[i,j,k]
                dPv2[i,j,k] = pv2[i,j,k] - pv1[i,j,k]
                
                dPa3[i,j,k] = pa3[i,j,k] - pa1[i,j,k]
                dPv3[i,j,k] = pv3[i,j,k] - pv1[i,j,k]
                
                dPa4[i,j,k] = pa4[i,j,k] - pa1[i,j,k]
                dPv4[i,j,k] = pv4[i,j,k] - pv1[i,j,k]
                
                dPa5[i,j,k] = pa5[i,j,k] - pa1[i,j,k]
                dPv5[i,j,k] = pv5[i,j,k] - pv1[i,j,k]








z_array = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]

z_array = [1,20,40,60,80]

# dPa_array = np.arange(561,564,0.1)
# dPv_array = np.arange(430,445,2)
# nrow = 1
# ncol = 5
# for z in z_array:
#     plt.figure(figsize=(55,15), dpi = 300)
#     color = 'Blues' 
    
#     plt.subplot(nrow,ncol,1)
#     plt.contourf(pa1[1:-1,1:-1,z],dPa_array,cmap=color)
#     plt.axis('off')
#     # plt.title('case 1 $\epsilon$ = 95 mm',y = -0.075 ,font=font)    
    
    
#     plt.subplot(nrow, ncol, 2)
#     plt.contourf(pa2[1:-1,1:-1,z],dPa_array,cmap=color)
#     plt.axis('off')
#     # plt.title('case 2 $\epsilon$ = 122.5 mm',y = -0.075 ,font=font) 
    
#     plt.subplot(nrow, ncol, 3)
#     plt.contourf(pa3[1:-1,1:-1,z],dPa_array,cmap=color)
#     plt.axis('off')
#     # plt.title('case 3 $\epsilon$ = 132.5 mm',y = -0.075 ,font=font) 
    
#     plt.subplot(nrow, ncol, 4)
#     plt.contourf(pa4[1:-1,1:-1,z],dPa_array,cmap=color)
#     plt.axis('off')
#     # plt.title('case 4 $\epsilon$ = 165 mm',y = -0.075,font=font) 
    
    
#     plt.subplot(nrow, ncol, 5)
#     plt.contourf(pa5[1:-1,1:-1,z],dPa_array,cmap=color)
#     plt.axis('off')
#     # plt.title('case 5 $\epsilon$ = 177.5 mm',y = -0.075,font=font) 
    
    
#     cax = plt.axes([1.01, 0.075, 0.025, 0.85])
#     cbar = plt.colorbar(cax=cax)
#     cbar.ax.tick_params(labelsize = 40)
#     cbar.set_label('Pressue (Pa)', rotation=270,labelpad=+65,font=font)
    
#     plt.suptitle(' z = ' + str(z) , y = -0.05, font = font, fontsize = 65)
#     plt.tight_layout()
    
#     # plt.savefig('Heat_transfer_analysis_results/Temperature_z_' + str(z) + '.png', bbox_inches='tight', dpi = 300)
#     plt.show()
    



dPa_array = np.arange(-0.73,0.74,0.01)
dPv_array = np.arange(-0.73,0.74,0.01)

nrow = 2
ncol = 4

font.set_size(75)


for z in z_array:
    plt.figure(figsize=(45,25))
    color = 'RdBu' # plt.cm.get_cmap('Reds') 
    
    plt.subplot(nrow,ncol,1)
    plt.contourf(dPa2[1:-1,1:-1,z],dPa_array,cmap=color)
    plt.axis('off')
    plt.title('Case 2 $\epsilon$ = 122.5 mm \nArterial Compartment',y = 1.05,font=font)    
    
    plt.subplot(nrow,ncol,2)
    plt.contourf(dPa3[1:-1,1:-1,z],dPa_array,cmap=color)
    plt.axis('off')
    plt.title('Case 3 $\epsilon$ = 132.5 mm\nArterial Compartment',y = 1.05,font=font) 
    
    plt.subplot(nrow, ncol, 3)
    plt.contourf(dPa4[1:-1,1:-1,z],dPa_array,cmap=color)
    plt.axis('off')
    plt.title('Case 4 $\epsilon$ = 165 mm\nArterial Compartment',y = 1.05,font=font) 
    
    plt.subplot(nrow, ncol, 4)
    plt.contourf(dPa5[1:-1,1:-1,z],dPa_array,cmap=color)
    plt.axis('off')
    plt.title('Case 5 $\epsilon$ = 177.5 mm\nArterial Compartment',y = 1.05,font=font) 
    
    
    # cax = plt.axes([1.00, 0.45, 0.022, 0.42])
    # cbar = plt.colorbar(cax=cax)
    # cbar.ax.tick_params(labelsize = 40)
    # cbar.set_label('arterial compartment pressure error', rotation=270,labelpad=+65,font=font)
    
    
    color = 'RdBu' # plt.cm.get_cmap('Reds') 
    
    # plt.suptitle('z = ' + str(z) ,y = -0.04,font=font , fontsize = 75)
    
    
    plt.subplot(nrow, ncol, 5)
    plt.contourf(dPv2[1:-1,1:-1,z],dPv_array,cmap=color)
    plt.axis('off')
    plt.title('Venous Compartment' ,y = -0.1,font=font)    
    
    plt.subplot(nrow, ncol, 6)
    plt.contourf(dPv3[1:-1,1:-1,z],dPv_array,cmap=color)
    plt.axis('off')
    plt.title('Venous Compartment' ,y = -0.1,font=font)    
    
    plt.subplot(nrow, ncol, 7)
    plt.contourf(dPv4[1:-1,1:-1,z],dPv_array,cmap=color)
    plt.axis('off')
    plt.title('Venous Compartment' ,y = -0.1,font=font)    
    
    plt.subplot(nrow, ncol, 8)
    plt.contourf(dPv5[1:-1,1:-1,z],dPv_array,cmap=color)
    plt.axis('off')
    plt.title('Venous Compartment' ,y = -0.1,font=font)    
    
    
    cax = plt.axes([1.00, 0.03, 0.022, 0.86])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Pressure Error (Pa)', rotation=270,labelpad=+75,font=font, size = 85)
    
    
    plt.tight_layout()
    plt.savefig('Flow_Analysis_results/analysis_1_layer_'+str(z) + '.png', dpi = 100, bbox_inches = 'tight')
    plt.show()
    











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
'''