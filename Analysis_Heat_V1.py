import numpy as np
import matplotlib.pyplot as plt

case = 1
dx = 2.5
E = 38



Pin = 1000 
Pout = 1


'''
solution = 'case'+str(case)+'/flow_solutions/X_' + str(E) + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_acomp = 'case'+str(case)+'/flow_solutions/prs_acomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_vcomp = 'case'+str(case)+'/flow_solutions/prs_vcomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_a = 'case'+str(case)+'/flow_solutions/prs_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_v = 'case'+str(case)+'/flow_solutions/prs_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qa = 'case'+str(case)+'/flow_solutions/Q_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qv = 'case'+str(case)+'/flow_solutions/Q_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'

X = np.load(solution)
pa_comp = np.load(prs_acomp)
pv_comp = np.load(prs_vcomp)
PA = np.load(prs_a)
PV = np.load(prs_v)
QA = np.load(qa)
QV = np.load(qv)

del(solution, prs_acomp, prs_vcomp, prs_a, prs_v, qa, qv)

min_pa = round(np.min(X[0:un_t]),3)
max_pa = round(np.max(X[0:un_t]),3)

min_pv = round(np.min(X[un_t:2*un_t]),3)
max_pv = round(np.max(X[un_t:2*un_t]),3)

z = 20
plt.figure(figsize = (12,10))

plt.contourf(pa_comp[1:-1,1:-1,z], np.arange(min_pa, max_pa, 0.005))
plt.colorbar()

plt.figure(figsize = (12,10))
plt.contourf(pv_comp[1:-1,1:-1,z], np.arange(min_pv, max_pv, 0.005))
plt.colorbar()

'''
dom = np.load('case1/2.5_dom.npy')
c_dom = np.load('case1/2.5_cdom.npy')

nx, ny, nz = np.shape(dom)

un_t = np.max(c_dom) + 1

# Xt = np.load('case2/heat_solutions/Heat_X_5_a_0.001_q_1000_Tin_20_Tamb_0_hamb_1E-05_hbt_10_.npy')
# Xt2 = np.load('case2/heat_solutions/Heat_X_5_a_0.001_q_1000_Tin_20_Tamb_0_hamb_10_hbt_10_.npy')
Xt = np.load('case1/heat_solutions/Heat_X_10_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')

# error_T = Xt2 - Xt

T_Tis = Xt[:un_t]
T_Art = Xt[un_t:un_t+63]
T_Ven = Xt[un_t+63:]



nx,ny,nz = np.shape(dom)
T = np.zeros((nx,ny,nz), dtype = float)
T[:,:,:] = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 0):
                T[i,j,k] = Xt[int(c_dom[i,j,k])]

min_T = np.min(T_Tis)
max_T = np.max(T_Tis)
dT = (max_T - min_T)/50

colors = 'viridis' #'Reds'
z = 1

for z in range(1,nz-1):
    plt.figure(figsize = (12,10))
    
    plt.contourf(T[1:-1,1:-1,z], np.arange(min_T, max_T+dT, dT), cmap=colors)
    plt.colorbar()
    plt.title('z = ' + str(z))
    plt.show()
    

QA = np.load('case1/flow_solutions/Q_arteries_10_Pin_1000_Pout_1.npy')
QV = np.load('case1/flow_solutions/Q_veins_10_Pin_1000_Pout_1.npy')

Cp = 3421
rho = 1000

Energy_in = round(round(QA[0],10)*round(T_Art[0],10)*Cp*rho,10)
Energy_out = round(round(QV[0],10)*round(T_Ven[0],10)*Cp*rho,10)
Energy_gen = un_t*(2.5E-3)**3*1000
Energy_conv = 0


tissue_index = 0
Kt = 0.5
h_amb = 10 # 1E-5
dx = dy = dz = 2.5E-3
air_index = -1
T_amb = 20

conv_energy = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == tissue_index):
                if(dom[i+1,j,k] == air_index):                    
                    conv_energy = conv_energy + 1/(0.5*dx/Kt + 1/h_amb)*(dy*dz)*(T[i,j,k] - T_amb)
                if(dom[i-1,j,k] == air_index):
                    conv_energy = conv_energy + 1/(0.5*dx/Kt + 1/h_amb)*(dy*dz)*(T[i,j,k] - T_amb)
                if(dom[i,j+1,k] == air_index):                    
                    conv_energy = conv_energy + 1/(0.5*dx/Kt + 1/h_amb)*(dy*dz)*(T[i,j,k] - T_amb)
                if(dom[i,j-1,k] == air_index):
                    conv_energy = conv_energy + 1/(0.5*dx/Kt + 1/h_amb)*(dy*dz)*(T[i,j,k] - T_amb)
                if(dom[i,j,k+1] == air_index):                    
                    conv_energy = conv_energy + 1/(0.5*dx/Kt + 1/h_amb)*(dy*dz)*(T[i,j,k] - T_amb)
                if(dom[i,j,k-1] == air_index):
                    conv_energy = conv_energy + 1/(0.5*dx/Kt + 1/h_amb)*(dy*dz)*(T[i,j,k] - T_amb)


Energy_conv = conv_energy

error = Energy_in + Energy_out + Energy_gen - Energy_conv

print(error)
