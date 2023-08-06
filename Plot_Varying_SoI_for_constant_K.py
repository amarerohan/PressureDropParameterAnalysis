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

case = 5

e1 = 71
e2 = 100


dT1 = np.load('Temperature_error_5_1.npy')
dT2 = np.load('Temperature_error_5_1_E100_Kt0.5.npy')
th1 = np.load('Theta_error_5_1.npy')
th2 = np.load('Theta_error_5_1_E100_Kt0.5.npy')


dT1_max = np.max(np.abs(dT1))
dT2_max = np.max(np.abs(dT2))
th1_max = np.max(np.abs(th1))
th2_max = np.max(np.abs(th2))


n = 12

dT_array = np.arange(-max(dT1_max,dT2_max), max(dT1_max, dT2_max) + 2*max(dT1_max, dT2_max)/n, 2*max(dT1_max, dT2_max)/n)
th_array = np.arange(-max(th1_max,th2_max), max(th1_max, th2_max) + 2*max(th1_max, th2_max)/n, 2*max(th1_max, th2_max)/n)

z_array = [1,20,40,60,80]

for z in z_array:
    plt.figure(figsize=(25,25), dpi = 100)
    
    plt.subplot(2,2,1)
    plt.contourf(dT1[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(a)', y = -0.075, font=font)
    
    
    plt.subplot(2,2,2)
    plt.contourf(dT2[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(b)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.54, 0.022, 0.45])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Temperature Difference ($^oC$)', rotation=270,labelpad=+65,font=font)
    
    
    plt.subplot(2,2,3)
    plt.contourf(th1[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(c)', y = -0.075, font=font)
    
    plt.subplot(2,2,4)
    plt.contourf(th2[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(d)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.02, 0.022, 0.46])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('$\Theta$', rotation=270,labelpad=+65,font=font)
    
    plt.tight_layout()
    # plt.savefig('New_Temperature_plots/case_5_E71_100_Kt0.5_z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()

print(round(dT1_max,3), round(th1_max,3))  
print(round(dT2_max,3), round(th2_max,3))  

'''

dT1 = np.load('Temperature_error_5_1_Kt0.75.npy')
dT2 = np.load('Temperature_error_5_1_E100_Kt0.75.npy')
th1 = np.load('Theta_error_5_1_Kt0.75.npy')
th2 = np.load('Theta_error_5_1_E100_Kt0.75.npy')

dT1_max = np.max(np.abs(dT1))
dT2_max = np.max(np.abs(dT2))
th1_max = np.max(np.abs(th1))
th2_max = np.max(np.abs(th2))

n = 12

dT_array = np.arange(-max(dT1_max,dT2_max), max(dT1_max, dT2_max) + 2*max(dT1_max, dT2_max)/n, 2*max(dT1_max, dT2_max)/n)
th_array = np.arange(-max(th1_max,th2_max), max(th1_max, th2_max) + 1*max(th1_max, th2_max)/n, 2*max(th1_max, th2_max)/n)

z_array = [1,10,20,30,40,50,60,70,80]

for z in z_array:
    plt.figure(figsize=(25,25), dpi = 100)
    
    plt.subplot(2,2,1)
    plt.contourf(dT1[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(a)', y = -0.075, font=font)
    
    
    plt.subplot(2,2,2)
    plt.contourf(dT2[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(b)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.54, 0.022, 0.45])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Temperature Difference ($^oC$)', rotation=270,labelpad=+65,font=font)
    
    
    plt.subplot(2,2,3)
    plt.contourf(th1[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(c)', y = -0.075, font=font)
    
    plt.subplot(2,2,4)
    plt.contourf(th2[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(d)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.02, 0.022, 0.46])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('$\Theta$', rotation=270,labelpad=+65,font=font)
    
    plt.tight_layout()
    plt.savefig('New_Temperature_plots/case_5_E71_100_Kt0.75_z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()
    
  
    
'''   


'''

dT1 = np.load('Temperature_error_5_1_Kt1.0.npy')
dT2 = np.load('Temperature_error_5_1_E100_Kt1.0.npy')
th1 = np.load('Theta_error_5_1_Kt1.0.npy')
th2 = np.load('Theta_error_5_1_E100_Kt1.0.npy')

dT1_max = np.max(np.abs(dT1))
dT2_max = np.max(np.abs(dT2))
th1_max = np.max(np.abs(th1))
th2_max = np.max(np.abs(th2))

n = 12

dT_array = np.arange(-max(dT1_max,dT2_max), max(dT1_max, dT2_max) + 2*max(dT1_max, dT2_max)/n, 2*max(dT1_max, dT2_max)/n)
th_array = np.arange(-max(th1_max,th2_max), max(th1_max, th2_max) + 2*max(th1_max, th2_max)/n, 2*max(th1_max, th2_max)/n)

z_array = [1,10,20,30,40,50,60,70,80]

for z in z_array:
    plt.figure(figsize=(25,25), dpi = 100)
    
    plt.subplot(2,2,1)
    plt.contourf(dT1[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(a)', y = -0.075, font=font)
    
    
    plt.subplot(2,2,2)
    plt.contourf(dT2[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(b)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.54, 0.022, 0.45])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Temperature Difference ($^oC$)', rotation=270,labelpad=+65,font=font)
    
    
    plt.subplot(2,2,3)
    plt.contourf(th1[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(c)', y = -0.075, font=font)
    
    plt.subplot(2,2,4)
    plt.contourf(th2[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(d)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.02, 0.022, 0.46])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('$\Theta$', rotation=270,labelpad=+65,font=font)
    
    plt.tight_layout()
    plt.savefig('New_Temperature_plots/case_5_E71_100_Kt1.0z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()
    
print(round(dT1_max,3), round(th1_max,3))  
print(round(dT2_max,3), round(th2_max,3))  
  
'''  





dT1 = np.load('Temperature_error_5_1_Kt1.25.npy')
dT2 = np.load('Temperature_error_5_1_E100_Kt1.25.npy')
th1 = np.load('Theta_error_5_1_Kt1.25.npy')
th2 = np.load('Theta_error_5_1_E100_Kt1.25.npy')

dT1_max = np.max(np.abs(dT1))
dT2_max = np.max(np.abs(dT2))
th1_max = np.max(np.abs(th1))
th2_max = np.max(np.abs(th2))

# n = 12

dT_array = np.arange(-max(dT1_max,dT2_max), max(dT1_max, dT2_max) + 2*max(dT1_max, dT2_max)/n, 2*max(dT1_max, dT2_max)/n)
th_array = np.arange(-max(th1_max,th2_max), max(th1_max, th2_max) + 2*max(th1_max, th2_max)/n, 2*max(th1_max, th2_max)/n)

# z_array = [1,10,20,30,40,50,60,70,80]

for z in z_array:
    plt.figure(figsize=(25,25), dpi = 100)
    
    plt.subplot(2,2,1)
    plt.contourf(dT1[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(a)', y = -0.075, font=font)
    
    
    plt.subplot(2,2,2)
    plt.contourf(dT2[0:-1,0:-1,z], dT_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(b)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.54, 0.022, 0.45])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('Temperature Difference ($^oC$)', rotation=270,labelpad=+65,font=font)
    
    
    plt.subplot(2,2,3)
    plt.contourf(th1[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(c)', y = -0.075, font=font)
    
    plt.subplot(2,2,4)
    plt.contourf(th2[0:-1,0:-1,z], th_array, cmap = 'RdBu')#, np.arange(()))
    plt.axis('off')
    plt.title('(d)', y = -0.075, font=font)
    
    cax = plt.axes([1.00, 0.02, 0.022, 0.46])
    cbar = plt.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize = 65)
    cbar.set_label('$\Theta$', rotation=270,labelpad=+65,font=font)
    
    plt.tight_layout()
    # plt.savefig('New_Temperature_plots/case_5_E71_100_Kt1.25_z_'+str(z)+'.png', dpi = 100, bbox_inches='tight')
    plt.show()

print(round(dT1_max,3), round(th1_max,3))  
print(round(dT2_max,3), round(th2_max,3))  
  
