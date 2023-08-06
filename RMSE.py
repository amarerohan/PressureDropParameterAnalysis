import numpy as np

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
from matplotlib import cm



dom = np.load('case5/2.5_dom.npy')
# E_array = [71,75,80,85,90,95,100,105]

# for E in E_array:
#     # E = 71
    
#     dT1 = np.load('Temperature_error_5_1_E' + str(E)+'_Kt0.5.npy')
#     dT2 = np.load('Temperature_error_5_1_E' + str(E)+'_Kt0.75.npy')
#     dT3 = np.load('Temperature_error_5_1_E' + str(E)+'_Kt1.25.npy')
#     dT4 = np.load('Temperature_error_5_1_E' + str(E)+'_Kt1.5.npy')
#     dT5 = np.load('Temperature_error_5_1_E' + str(E)+'_Kt1.75.npy')
    
#     sum_T1 = 0
#     sum_T2 = 0
#     sum_T3 = 0
#     sum_T4 = 0
#     sum_T5 = 0
    
#     sum_T1 = round(np.sum(dT1),2)
#     sum_T2 = round(np.sum(dT2),2)
#     sum_T3 = round(np.sum(dT3),2)
#     sum_T4 = round(np.sum(dT4),2)
#     sum_T5 = round(np.sum(dT5),2)
    
#     print('E = ', E)
#     print(sum_T1, 'RMSE ', round(np.sqrt(np.mean((np.square(dT1)))),5))
#     print(sum_T2, 'RMSE ', round(np.sqrt(np.mean((np.square(dT2)))),5))
#     print(sum_T3, 'RMSE ', round(np.sqrt(np.mean((np.square(dT3)))),5))
#     print(sum_T4, 'RMSE ', round(np.sqrt(np.mean((np.square(dT4)))),5))
#     print(sum_T5, 'RMSE ', round(np.sqrt(np.mean((np.square(dT5)))),5))
    
nx, ny, nz = np.shape(dom)

k_array = [0.50,0.75,1.00,1.25,1.50,1.75]
E_array = [177.5, 187.5, 200.0, 212.5, 225.0, 237.5, 250.0, 262.5]

rmse = np.zeros((len(k_array), 8), dtype = float)
sum_T = np.zeros((len(k_array), 8), dtype = float)


sumt8 = []

for i in range(len(k_array)):
    
    k = k_array[i]
    dT1 = np.load('Temperature_error_5_1_E71_Kt'+str(k)+'.npy')
    dT2 = np.load('Temperature_error_5_1_E75_Kt'+str(k)+'.npy')
    dT3 = np.load('Temperature_error_5_1_E80_Kt'+str(k)+'.npy')
    dT4 = np.load('Temperature_error_5_1_E85_Kt'+str(k)+'.npy')
    dT5 = np.load('Temperature_error_5_1_E90_Kt'+str(k)+'.npy')
    dT6 = np.load('Temperature_error_5_1_E95_Kt'+str(k)+'.npy')
    dT7 = np.load('Temperature_error_5_1_E100_Kt'+str(k)+'.npy')
    dT8 = np.load('Temperature_error_5_1_E105_Kt'+str(k)+'.npy')
    
    
    sum_T1 = rmse_1 = 0
    sum_T2 = rmse_2 = 0
    sum_T3 = rmse_3 = 0
    sum_T4 = rmse_4 = 0
    sum_T5 = rmse_5 = 0
    sum_T6 = rmse_6 = 0
    sum_T7 = rmse_7 = 0
    sum_T8 = rmse_8 = 0
    
    
    sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = sum7 = sum8 = 0.0
    sum_sq1 = sum_sq2 = sum_sq3 = sum_sq4 = sum_sq5 = sum_sq6 = sum_sq7 = sum_sq8 = 0
    
    N = 0
    
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if(dom[x,y,z] == 0):
                    sum1 = sum1 + dT1[x,y,z]
                    sum2 = sum2 + dT2[x,y,z]
                    sum3 = sum3 + dT3[x,y,z]
                    sum4 = sum4 + dT4[x,y,z]
                    sum5 = sum5 + dT5[x,y,z]
                    sum6 = sum6 + dT6[x,y,z]
                    sum7 = sum7 + dT7[x,y,z]
                    sum8 = sum8 + dT8[x,y,z]
                    
                    sum_sq1 = sum_sq1 + (dT1[x,y,z])**2
                    sum_sq2 = sum_sq2 + (dT2[x,y,z])**2
                    sum_sq3 = sum_sq3 + (dT3[x,y,z])**2
                    sum_sq4 = sum_sq4 + (dT4[x,y,z])**2
                    sum_sq5 = sum_sq5 + (dT5[x,y,z])**2
                    sum_sq6 = sum_sq6 + (dT6[x,y,z])**2
                    sum_sq7 = sum_sq7 + (dT7[x,y,z])**2
                    sum_sq8 = sum_sq8 + (dT8[x,y,z])**2
                    
                    N = N + 1
    
    
    rmse_1 = np.sqrt(sum_sq1/N)
    rmse_2 = np.sqrt(sum_sq2/N)
    rmse_3 = np.sqrt(sum_sq3/N)
    rmse_4 = np.sqrt(sum_sq4/N)
    rmse_5 = np.sqrt(sum_sq5/N)
    rmse_6 = np.sqrt(sum_sq6/N)
    rmse_7 = np.sqrt(sum_sq7/N)
    rmse_8 = np.sqrt(sum_sq8/N)
    
    sum_T1 = sum1
    sum_T2 = sum2
    sum_T3 = sum3
    sum_T4 = sum4
    sum_T5 = sum5
    sum_T6 = sum6
    sum_T7 = sum7
    sum_T8 = sum8
    
    # print('\nk = ', k)
    # print(round(rmse_1,3), round(np.max(np.abs(dT1)),2), round(sum_T1,1), round(sum_T1/N,2))
    # print(round(rmse_2,3), round(np.max(np.abs(dT2)),2), round(sum_T2,1), round(sum_T2/N,2))
    # print(round(rmse_3,3), round(np.max(np.abs(dT3)),2), round(sum_T3,1), round(sum_T3/N,2))
    # print(round(rmse_4,3), round(np.max(np.abs(dT4)),2), round(sum_T4,1), round(sum_T4/N,2))
    # print(round(rmse_5,3), round(np.max(np.abs(dT5)),2), round(sum_T5,1), round(sum_T5/N,2))
    # print(round(rmse_6,3), round(np.max(np.abs(dT6)),2), round(sum_T6,1), round(sum_T6/N,2))
    # print(round(rmse_7,3), round(np.max(np.abs(dT7)),2), round(sum_T7,1), round(sum_T7/N,2))
    # print(round(rmse_8,3), round(np.max(np.abs(dT8)),2), round(sum_T8,1), round(sum_T8/N,2))
    
    rmse[i,0] = rmse_1
    rmse[i,1] = rmse_2 
    rmse[i,2] = rmse_3 
    rmse[i,3] = rmse_4 
    rmse[i,4] = rmse_5 
    rmse[i,5] = rmse_6 
    rmse[i,6] = rmse_7 
    rmse[i,7] = rmse_8
    
    sum_T[i,0] = sum_T1
    sum_T[i,1] = sum_T2
    sum_T[i,2] = sum_T3
    sum_T[i,3] = sum_T4
    sum_T[i,4] = sum_T5
    sum_T[i,5] = sum_T6
    sum_T[i,6] = sum_T7
    sum_T[i,7] = sum_T8
    

poly_degree = 4

r1 = np.polyfit(E_array, rmse[0,:],poly_degree)
r2 = np.polyfit(E_array, rmse[1,:],poly_degree)
r3 = np.polyfit(E_array, rmse[2,:],poly_degree)
r4 = np.polyfit(E_array, rmse[3,:],poly_degree)
r5 = np.polyfit(E_array, rmse[4,:],poly_degree)
r6 = np.polyfit(E_array, rmse[5,:],poly_degree)


s1 = np.polyfit(E_array, sum_T[0,:],poly_degree)
s2 = np.polyfit(E_array, sum_T[1,:],poly_degree)
s3 = np.polyfit(E_array, sum_T[2,:],poly_degree)
s4 = np.polyfit(E_array, sum_T[3,:],poly_degree)
s5 = np.polyfit(E_array, sum_T[4,:],poly_degree)
s6 = np.polyfit(E_array, sum_T[5,:],poly_degree)



def R1(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + r1[i]*x^(poly_degree-i) 
    return sum_y

def R2(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + r2[i]*x^(poly_degree-i) 
    return sum_y

def R3(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + r3[i]*x^(poly_degree-i) 
    return sum_y

def R4(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + r4[i]*x^(poly_degree-i) 
    return sum_y

def R5(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + r5[i]*x^(poly_degree-i) 
    return sum_y

def R6(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + r6[i]*x^(poly_degree-i) 
    return sum_y



def S1(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + s1[i]*x^(poly_degree-i) 
    return sum_y

def S2(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + s2[i]*x^(poly_degree-i) 
    return sum_y

def S3(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + s3[i]*x^(poly_degree-i) 
    return sum_y

def S4(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + s4[i]*x^(poly_degree-i) 
    return sum_y

def S5(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + s5[i]*x^(poly_degree-i) 
    return sum_y

def S6(x):
    sum_y = 0
    for i in range(poly_degree+1):
        sum_y = sum_y + s6[i]*x^(poly_degree-i) 
    return sum_y


# sR_sq1 = R_sq2 = R_sq3 = R_s
# 

'''
x = E_array[0]
y1 = s1[0]*x**4 + s1[1]*x**3 + s1[2]*x**2 + s1[3]*x + s1[4]

def y(x):
    y1 = s1[0]*x**4 + s1[1]*x**3 + s1[2]*x**2 + s1[3]*x + s1[4]
    print(y1)

def z1(x):
    y1 = r1[0]*x**4 + r1[1]*x**3 + r1[2]*x**2 + r1[3]*x + r1[4]
    return(y1)

def z2(x):
    y1 = r2[0]*x**4 + r2[1]*x**3 + r2[2]*x**2 + r2[3]*x + r2[4]
    return(y1)
   
def z3(x):
    y1 = r3[0]*x**4 + r3[1]*x**3 + r3[2]*x**2 + r3[3]*x + r3[4]
    return(y1)

def z4(x):
    y1 = r4[0]*x**4 + r4[1]*x**3 + r4[2]*x**2 + r4[3]*x + r4[4]
    return(y1)

def z5(x):
    y1 = r5[0]*x**4 + r5[1]*x**3 + r5[2]*x**2 + r5[3]*x + r5[4]
    return(y1)

def z6(x):
    y1 = r6[0]*x**4 + r6[1]*x**3 + r6[2]*x**2 + r6[3]*x + r6[4]
    return(y1)
    

x = [191.73, 191.60, 191.22, 190.90, 190.41, 190.02]



z1(x[0])
z2(x[1])
z3(x[2])
z4(x[3])
z5(x[4])
z6(x[5])


x = np.arange(200,211,0.25)

rsme_error = np.zeros((len(x),6), dtype = float)

for i in range(len(x)):
    rsme_error[i,0] = z1(x[i])
    rsme_error[i,1] = z2(x[i])
    rsme_error[i,2] = z3(x[i])
    rsme_error[i,3] = z4(x[i])
    rsme_error[i,4] = z5(x[i])
    rsme_error[i,5] = z6(x[i])
'''   


font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(20)

colormaps = cm.get_cmap('tab10',8)


plt.figure(figsize = (12,5), dpi = 100)
for i in range(len(k_array)):
    plt.plot(E_array, rmse[i,:], "o:", markersize = 10, label='$k_t$ = ' + str(k_array[i]) + ' $Wm^{-1o}C^{-1} $', color = colormaps(i))

plt.yticks(np.arange(0.05,0.14,0.01), font=font)
plt.xticks(np.arange(175,275,10),font = font)
plt.legend(prop = font, ncol = 2)

plt.xlabel('$\epsilon$ (mm)', font = font)
plt.ylabel('RMSE ($^oC$)', font = font)

plt.grid('True', color = 'lightgrey')
# plt.savefig('RMSE.png', dpi = 300, bbox_inches='tight')
plt.show()




plt.figure(figsize = (12,5), dpi = 100)
for i in range(len(k_array)):
    plt.plot(E_array, sum_T[i,:], "o--", alpha = 1, markersize = 10, label='$k_t$ = ' + str(k_array[i]) + ' $Wm^{-1o}C^{-1} $', color = colormaps(i))

plt.yticks(np.arange(-10000,11000,2500), font=font)
plt.xticks(np.arange(175,275,10),font = font)
plt.legend(prop = font, ncol = 2)

plt.xlabel('$\epsilon$ (mm)', font = font)
plt.ylabel('$ \sum{\Delta T_t}$' + ' $(^oC)$', font = font)

plt.grid('True', color = 'lightgrey')
# plt.savefig('sum_T.png', dpi = 300, bbox_inches='tight')
plt.show()



plt.figure(figsize = (10,15), dpi = 100)
plt.subplot(2,1,1)
for i in range(len(k_array)):
    plt.plot(E_array, rmse[i,:], "o:", markersize = 10, label='$k_t$ = ' + str(k_array[i]) + ' $Wm^{-1o}C^{-1} $', color = colormaps(i))

plt.yticks(np.arange(0.05,0.14,0.01), font=font)
plt.xticks(np.arange(175,275,10),font = font)
plt.legend(prop = font, ncol = 2, bbox_to_anchor=(-0.0, 0.7, 1.0, 0.7))
plt.title('(a)', font=font, y = -0.25)
plt.xlabel('$\epsilon$ (mm)', font = font)
plt.ylabel('RMSE ($^oC$)', font = font)

plt.grid('True', color = 'lightgrey')


plt.subplot(2,1,2)
for i in range(len(k_array)):
    plt.plot(E_array, sum_T[i,:], "o--", alpha = 1, markersize = 10, label='$k_t$ = ' + str(k_array[i]) + ' $Wm^{-1o}C^{-1} $', color = colormaps(i))

plt.yticks(np.arange(-10000,11000,2500), font=font)
plt.xticks(np.arange(175,275,10),font = font)
# plt.legend(prop = font, ncol = 2, bbox_to_anchor=(-0.0, 1.275, 1.0, 1.275))

plt.xlabel('$\epsilon$ (mm)', font = font)
plt.ylabel('$ \sum{\Delta T_t}$' + ' $(^oC)$', font = font)

plt.grid('True', color = 'lightgrey')
plt.title('(b)', font=font, y = -0.25)

plt.tight_layout()

# plt.savefig('RSME_sum_Combo_plot.png', dpi = 300, bbox_inches='tight')
