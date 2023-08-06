import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from multiprocessing import Process, Pool, Manager
import os


case = 1
E = 38
dx = dy = dz = 2.5*1e-3

q = 1000 
T_in = 35
T_amb = 20
h_amb = 10
h_bt = 10
a = 1E-3
tissue_index = 0
air_index = -1
Kt = 0.5
rho_b = 1000 
Cp = 1000 
Pin = 1000
Pout = 1

QA = np.load('case'+str(case)+ '/flow_solutions/' + str(E) + '/Q_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy')
QV = np.load('case'+str(case)+ '/flow_solutions/' + str(E) + '/Q_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy')

path = 'case'+str(case)+'/heat_solutions/' + str(E)

row = np.load(path+'/Heat_row_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy' )      
col = np.load(path+'/Heat_col_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy')
data = np.load(path+'/Heat_data_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy')    
Bt = np.load(path+'/Heat_B_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'  )

c_dom = np.load('case'+str(case)+'/2.5_cdom.npy')
dom = np.load('case'+str(case)+'/2.5_dom.npy')



nx,ny,nz = np.shape(c_dom)
un_t = np.max(c_dom) + 1
un_a = int((len(Bt) - un_t)/2)
un_v = un_a

Bt[0:un_t] = Bt[0:un_t] + q*dx*dy*dz     

'''
Nt = len(Bt)
At = sp.csc_matrix((data,(row,col)), shape = (Nt, Nt))  

max_iter = 1000
tol = 1e-12

solver_time_start = time.time()

LU = spla.spilu(At)
M = spla.LinearOperator(np.shape(LU), LU.solve)    
x0 = np.zeros((Nt,1), dtype = float)
x0[:] = T_amb
X = spla.lgmres(At,Bt,M=M,x0=x0, maxiter = max_iter, tol = tol, inner_m = 500, atol = tol)#,tol=1e-15, atol = 1e-15)[0]
reason = X[1]
Xt = X[0]

solver_time_stop = time.time()
print('Time taken for solving the matrix = ', round((solver_time_stop - solver_time_start)/60,3), ' mins')

print('Reason for lgmres termination = ', reason)

filename = path+'/Heat_X_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'

# filename = 'case'+str(case)+'/Heat_X_TEST.npy'
np.save(filename,Xt)

heat_stop = time.time()


T_Tis = Xt[:un_t]
T_Art = Xt[un_t:un_t+un_a]
T_Ven = Xt[un_t+un_a:]

Tissue_filename = path+'/T_tissue_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'
Art_filename = path+'/T_Art_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'
Vein_filename = path+'/T_Vein_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'

np.save(Tissue_filename, T_Tis)
np.save(Art_filename, T_Art)
np.save(Vein_filename, T_Ven)



T = np.zeros((nx,ny,nz), dtype = float)
T[:,:,:] = T_amb
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == tissue_index):
                c_index = c_dom[i,j,k]
                T[i,j,k] = T_Tis[c_index]
            
T_dom_filename = path+'/T_domain_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'

np.save(T_dom_filename, T)

# CHECK ENERGY CONSERVATION

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



energy_in = round(QA[0],10)*rho_b*Cp*T_in
energy_out = round(QV[0],10)*rho_b*Cp*T_Ven[0]
energy_generated = q*un_t*dx*dy*dz

energy_error = abs(energy_generated) + abs(energy_in) - abs(energy_out) - abs(conv_energy)

# print('Flow going in = ', QA[0], T_in, Cp, rho_b, QA[0]*rho_b*Cp*T_in)
# print('Flow going out = ', QV[0], T_Ven[0], Cp, rho_b, QV[0]*rho_b*Cp*T_Ven[0])

print('Temperature of blood going in = ', round(T_in,2))
print('Temperature of blood leaving  = ', round(T_Ven[0],2))



print('Energy Conv = ', round(conv_energy,3))
print('Energy Gen  = ', round(energy_generated,3))
print('Energy in   = ', round(energy_in,3))
print('Energy out  = ', round(energy_out,3))
print('Energy err  = ', round(energy_error,3))


res = At.multiply(Xt)
res = res.sum(axis = 1) - Bt
residual = np.linalg.norm(res)
print('residual = ', format(np.linalg.norm(residual),'.2E'))
'''
