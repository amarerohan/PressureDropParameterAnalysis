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

kt = 1.75


case5 = 5

e5 = 71
if kt == 0.5:
    T1 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
else:
    T1 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')

e5 = 75
T2 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')

e5 = 80
T3 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')

e5 = 85
T4 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')

e5 = 90
T5 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')

e5 = 95
T6 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')

e5 = 100
T7 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')

e5 = 105
T8 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_'+str(kt)+'_.npy')


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


delta_T1 = np.copy(theta2)
delta_T2 = np.copy(theta2)
delta_T3 = np.copy(theta2)
delta_T4 = np.copy(theta2)
delta_T5 = np.copy(theta2)
delta_T6 = np.copy(theta2)
delta_T7 = np.copy(theta2)
delta_T8 = np.copy(theta2)


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
                
                
                delta_T1[i,j,k] = T1[i,j,k] - T0[i,j,k]
                delta_T2[i,j,k] = T2[i,j,k] - T0[i,j,k]
                delta_T3[i,j,k] = T3[i,j,k] - T0[i,j,k]
                delta_T4[i,j,k] = T4[i,j,k] - T0[i,j,k]
                delta_T5[i,j,k] = T5[i,j,k] - T0[i,j,k]
                delta_T6[i,j,k] = T6[i,j,k] - T0[i,j,k]
                delta_T7[i,j,k] = T7[i,j,k] - T0[i,j,k]
                delta_T8[i,j,k] = T8[i,j,k] - T0[i,j,k]
                



np.save('Temperature_error_5_1_E71_Kt'+str(kt)+'.npy', delta_T1)
np.save('Temperature_error_5_1_E75_Kt'+str(kt)+'.npy', delta_T2)
np.save('Temperature_error_5_1_E80_Kt'+str(kt)+'.npy', delta_T3)
np.save('Temperature_error_5_1_E85_Kt'+str(kt)+'.npy', delta_T4)
np.save('Temperature_error_5_1_E90_Kt'+str(kt)+'.npy', delta_T5)
np.save('Temperature_error_5_1_E95_Kt'+str(kt)+'.npy', delta_T6)
np.save('Temperature_error_5_1_E100_Kt'+str(kt)+'.npy', delta_T7)
np.save('Temperature_error_5_1_E105_Kt'+str(kt)+'.npy', delta_T8)


np.save('Theta_error_5_1_E71_Kt'+str(kt)+'.npy', theta1)
np.save('Theta_error_5_1_E75_Kt'+str(kt)+'.npy', theta2)
np.save('Theta_error_5_1_E80_Kt'+str(kt)+'.npy', theta3)
np.save('Theta_error_5_1_E85_Kt'+str(kt)+'.npy', theta4)
np.save('Theta_error_5_1_E90_Kt'+str(kt)+'.npy', theta5)
np.save('Theta_error_5_1_E95_Kt'+str(kt)+'.npy', theta6)
np.save('Theta_error_5_1_E100_Kt'+str(kt)+'.npy', theta7)
np.save('Theta_error_5_1_E105_Kt'+str(kt)+'.npy', theta8)


print('Kt = ', kt)
print(71, round(np.max(np.abs(delta_T1)),3), round(np.max(np.abs(theta1)),3))
print(75, round(np.max(np.abs(delta_T2)),3), round(np.max(np.abs(theta2)),3))
print(80, round(np.max(np.abs(delta_T3)),3), round(np.max(np.abs(theta3)),3))
print(85, round(np.max(np.abs(delta_T4)),3), round(np.max(np.abs(theta4)),3))
print(90, round(np.max(np.abs(delta_T5)),3), round(np.max(np.abs(theta5)),3))
print(95, round(np.max(np.abs(delta_T6)),3), round(np.max(np.abs(theta6)),3))
print(100, round(np.max(np.abs(delta_T7)),3), round(np.max(np.abs(theta7)),3))
print(105, round(np.max(np.abs(delta_T8)),3), round(np.max(np.abs(theta8)),3))

