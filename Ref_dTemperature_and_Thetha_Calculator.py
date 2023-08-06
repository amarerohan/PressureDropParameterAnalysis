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
T_tis1 = np.load('case'+str(case1)+'/heat_solutions/' + str(e1)+'/'+'T_domain_'+str(e1)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
dom1 = np.load('case'+str(case1)+'/2.5_dom.npy')

case2 = 5
e2 = 71
T_tis2 = np.load('case'+str(case2)+'/heat_solutions/' + str(e2)+'/'+'T_domain_'+str(e2)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_1.5_.npy')

case3 = 5
e3 = 71
T_tis3 = np.load('case'+str(case3)+'/heat_solutions/' + str(e3)+'/'+'T_domain_'+str(e3)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_1.75_.npy')

case4 = 5
e4 = 71
T_tis4 = np.load('case'+str(case4)+'/heat_solutions/' + str(e4)+'/'+'T_domain_'+str(e4)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_2.0_.npy')

case5 = 5
e5 = 71
T_tis5 = np.load('case'+str(case5)+'/heat_solutions/' + str(e5)+'/'+'T_domain_'+str(e5)+'_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_Kt_2.25_.npy')



theta_ref = 35 - 20

nx, ny, nz = np.shape(T_tis1)

theta2 = np.zeros((nx, ny, nz), dtype = float)

theta2[:,:,:] = 0 #-10
theta3 = np.copy(theta2)
theta4 = np.copy(theta2)
theta5 = np.copy(theta2)

delta_T2 = np.copy(theta2)
delta_T3 = np.copy(theta2)
delta_T4 = np.copy(theta2)
delta_T5 = np.copy(theta2)

for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if dom1[i,j,k] == 0:
                theta2[i,j,k] = (T_tis2[i,j,k] - T_tis1[i,j,k])/theta_ref
                theta3[i,j,k] = (T_tis3[i,j,k] - T_tis1[i,j,k])/theta_ref
                theta4[i,j,k] = (T_tis4[i,j,k] - T_tis1[i,j,k])/theta_ref
                theta5[i,j,k] = (T_tis5[i,j,k] - T_tis1[i,j,k])/theta_ref
                
                delta_T2[i,j,k] = T_tis2[i,j,k] - T_tis1[i,j,k]
                delta_T3[i,j,k] = T_tis3[i,j,k] - T_tis1[i,j,k]
                delta_T4[i,j,k] = T_tis4[i,j,k] - T_tis1[i,j,k]
                delta_T5[i,j,k] = T_tis5[i,j,k] - T_tis1[i,j,k]


# np.save('Temperature_error_5_1_E71_Kt1.5.npy',delta_T2)
# np.save('Temperature_error_5_1_E71_Kt1.75.npy',delta_T3)
# np.save('Temperature_error_5_1_E71_Kt2.0.npy',delta_T4)
# np.save('Temperature_error_5_1_E71_Kt2.25.npy',delta_T5)

# np.save('Theta_error_5_1_E71_Kt1.5.npy', theta2)
# np.save('Theta_error_5_1_E71_Kt1.75.npy', theta3)
# np.save('Theta_error_5_1_E71_Kt2.0.npy', theta4)
# np.save('Theta_error_5_1_E71_Kt2.25.npy', theta5)

