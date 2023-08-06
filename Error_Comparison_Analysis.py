import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from multiprocessing import Process, Pool, Manager
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(15)
# font.set_style('italic')
f = 20

Dx = 2.5
dx = dy = 0.001*Dx
dz = dx
dVol = 1 

case = 1
E = 38
e =  E*dx



myu = 1e-3
a = 1E-3
Ka = 1E-5
Kv = 1E-5
gamma_a = 1E-1
gamma_v = 1E-1
Lambda_a = Ka/myu
Lambda_v = Kv/myu

a_inlet = 0
v_outlet = 0

Pin = 1000
Pout = 1

equal = False

# # # #  LOAD THE REQUIRED FILES # # # 


a_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_db.csv')
v_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_db.csv')
a_out = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_out_pts.csv').to_numpy()
v_out = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_out_pts.csv').to_numpy()


dom = np.load('case'+str(case)+'/'+str(Dx)+'_dom.npy')
c_dom = np.load('case'+str(case)+'/'+'ref_cdom.npy')

nbr_a = np.load('case'+str(case)+'/artery_nbrhd_'+str(E)+'.npy', allow_pickle=True)
nbr_v = np.load('case'+str(case)+'/vein_nbrhd_'+str(E)+'.npy', allow_pickle=True)

a_nbr = np.load('case'+str(case)+'/a_nbr.npy', allow_pickle=True)
v_nbr = np.load('case'+str(case)+'/v_nbr.npy', allow_pickle=True)


Ca = np.load('case'+str(case)+'/Ca_' + str(E) +'.npy')
Cv = np.load('case'+str(case)+'/Cv_' + str(E) +'.npy')

nx, ny, nz = np.shape(dom)


         # CALCULATE THE NUMBER OF UNKNONWS # 
         
old_cdom = np.load('case' + str(case)+'/2.5_cdom.npy')
un_a = len(a_db) + 1
un_v = len(v_db) + 1
un_t = np.max(old_cdom) + 1
N_unk = un_a + un_v + 2*un_t


# # # # LOAD FLOW SOLUTIONS # # # # # 

prs_acomp = 'case'+str(case)+'/flow_solutions/' + str(E) + '/prs_acomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_vcomp = 'case'+str(case)+'/flow_solutions/' + str(E) +'/prs_vcomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_a = 'case'+str(case)+'/flow_solutions/' + str(E) +'/prs_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_v = 'case'+str(case)+'/flow_solutions/' + str(E) +'/prs_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qa = 'case'+str(case)+'/flow_solutions/' + str(E) +'/Q_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qv = 'case'+str(case)+'/flow_solutions/' + str(E) +'/Q_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'

pa = np.load(prs_acomp)
pv = np.load(prs_vcomp)
QA = np.load(qa)
QV = np.load(qv)
PArt = np.load(prs_a)
PVrt = np.load(prs_v)



a_ele = np.load('case'+str(case) + '/a_ele.npy')
v_ele = np.load('case'+str(case) + '/v_ele.npy')



# # # # # # # # HEAT TRANSFER # # # # # # # # # # # 

heat_start = time.time()

q = 1000
Kt = 0.5
Kb = 0.5
h_amb = 10
h_bt = 10
Cp = 3421

T_in = 35
T_amb = 20

rho_t = 1000# 1090
rho_b = 1000 #1050
rho = (rho_t + rho_b)/2.0

X = np.load('case'+str(case)+'/heat_solutions/' + str(E) + '/Heat_X_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy')


def SoI(center, point):
    x0, y0, z0 = center
    x1, y1, z1 = point
    
    s = np.sqrt( ((x0-x1)*dx)**2 + ((y0-y1)*dy)**2 + ((z0-z1)*dz)**2)
    
    return(s)

def eta(x,e,C,nC):
    if(x/e < 1):
        if equal == True:
            eta_x = 1/nC
        else:
            eta_x = C*np.exp(1/((abs(x/e))**2 - 1))#/(e**3) # !!!! NOT DIVIDED BY VOLUME 
    else:
        eta_x = 0
    # print(x,eta_x,C)
    return eta_x

def mass_flow_across_voxels(nbr, cell, ds, T):
    x1,y1,z1 = nbr
    x0,y0,z0 = cell
    
    t_ij_a = 2*ds*ds*(1/(ds/Lambda_a + ds/Lambda_a))
    Qa = t_ij_a*(pa[x1,y1,z1]-pa[x0,y0,z0])
    ma = rho_b*Qa
        
    t_ij_v = 2*ds*ds*(1/(ds/Lambda_v + ds/Lambda_v))
    Qv = t_ij_v*(pv[x1,y1,z1] - pv[x0,y0,z0])
    mv = rho_b*Qv
    
    m = ma + mv
    
    if m <= 0:
        m = 0
    
    mCp = m*Cp*T
    return(mCp)

def conduction(T):
    UA = 1/(dx/(2*Kt) + dx/(2*Kt))*dy*dz
    UAT = UA*T
    return(UAT)

def convection(T,h):
    UA = 1/(1/h + dx/(2*Kt))*dy*dz
    UAT = UA*T
    return(UAT)


tissue_index = 0
air_index = -1
artery_index = 1
vein_index = 2



def nbr_tissue(nbr, cell ,ds, T, h, index, t_sum, v_sum):
    mCp = mass_flow_across_voxels(nbr, cell, ds, T) 
    UAT = conduction(T)
    
    t_sum = t_sum + mCp + UAT

    return mCp, UAT, index, t_sum, v_sum


def nbr_convection(nbr, cell ,ds, T,h, index, t_sum, v_sum):
    mCp = 0
    UA = 1/(1/h + dx/(2*Kt))*dy*dz
    UAT = UA*T
    
    if index == air_index:
        t_sum = t_sum + mCp + UAT 
    else:
        v_sum = v_sum + UAT
    
    return mCp, UAT, index, t_sum, v_sum

    



un_a = len(a_db)
un_v = len(v_db)
Nt = un_t + len(a_db) + len(v_db)





func_db = {tissue_index: (lambda nbr,cell,ds, T, h, index, t_sum, v_sum: nbr_tissue(nbr,cell,ds, T, h, index, t_sum, v_sum)), 
           air_index: (lambda nbr,cell,ds, T, h, index, t_sum, v_sum: nbr_convection(nbr,cell,ds, T, h, index, t_sum, v_sum)),
           artery_index: (lambda nbr,cell,ds, T, h, index, t_sum, v_sum: nbr_convection(nbr,cell,ds, T, h, index, t_sum, v_sum)),
           vein_index: (lambda nbr,cell,ds, T, h, index, t_sum, v_sum: nbr_convection(nbr,cell,ds, T, h, index, t_sum, v_sum))}




print('case = ', case, ' E = ', E)

tissue_energy_matrix = np.zeros((nx,ny,nz), dtype = float)
vessel_energy_matrix = np.zeros((nx,ny,nz), dtype = float)
vessel_mass_matrix = np.zeros((nx,ny,nz), dtype = float)



             # # # TISSUE VOXEL # # # 
def tissue(input_variables):
    global tissue_energy_matrix
    global vessel_energy_matrix
    
    nz1, nz2 = input_variables
    print(nz1, nz2)
    
    for k in range(nz1,nz2):
        for j in range(ny):
            for i in range(nx):
                if(dom[i,j,k] == tissue_index):
                    
                    
                    t_sum = 0
                    v_sum = 0
                    
                    # NORTH
                    
                    mCp_N, uaT_N, ix_N, t_sum, v_sum = func_db[dom[i+1,j,k]]([i+1,j,k],[i,j,k],dx,X[c_dom[i+1,j,k]],h_amb, dom[i+1,j,k], t_sum, v_sum)
                    mCp_S, uaT_S, ix_S, t_sum, v_sum = func_db[dom[i-1,j,k]]([i-1,j,k],[i,j,k],dx,X[c_dom[i-1,j,k]],h_amb, dom[i-1,j,k], t_sum, v_sum)
                    mCp_E, uaT_E, ix_E, t_sum, v_sum = func_db[dom[i,j+1,k]]([i,j+1,k],[i,j,k],dx,X[c_dom[i,j+1,k]],h_amb, dom[i,j+1,k], t_sum, v_sum)
                    mCp_W, uaT_W, ix_W, t_sum, v_sum = func_db[dom[i,j-1,k]]([i,j-1,k],[i,j,k],dx,X[c_dom[i,j-1,k]],h_amb, dom[i,j-1,k], t_sum, v_sum)
                    mCp_F, uaT_F, ix_F, t_sum, v_sum = func_db[dom[i,j,k+1]]([i,j,k+1],[i,j,k],dx,X[c_dom[i,j,k+1]],h_amb, dom[i,j,k+1], t_sum, v_sum)
                    mCp_B, uaT_B, ix_B, t_sum, v_sum = func_db[dom[i,j,k-1]]([i,j,k-1],[i,j,k],dx,X[c_dom[i,j,k-1]],h_amb, dom[i,j,k-1], t_sum, v_sum)
                    
                    
                    tissue_energy_matrix[i,j,k] = t_sum
                    vessel_energy_matrix[i,j,k] = v_sum
                    
                    
                    

def vessel(nz1, nz2):
    global vessel_mass_matrix
    
    for k in range(nz1,nz2):
        for j in range(ny):
            for i in range(nx):
                if(dom[i,j,k] == tissue_index):
                    m_dot = 0
                                                            
                    for b in range(len(a_out)): 
                        if ([i,j,k] in nbr_a[b]):
                            ele = a_db.loc[a_db.iloc[:,2] == a_out[b,0]]
                            x0,y0,z0 = a_out[b,1:4].tolist()
                            s = SoI([x0,y0,z0],[i,j,k])
                            n_ex = eta(s,e,Ca[b], len(nbr_a[b]))
                            Q = abs(QA[int(ele.iloc[0,0])])
                            Q_dot = Q*rho_b*n_ex
                            m_dot = m_dot + Q_dot*Cp*X[un_t + int(ele.iloc[0,0])]
                            
                    vessel_mass_matrix[i,j,k] = m_dot
                    
    


    
start = time.time()
tissue([0,81])
stop = time.time()
np.save('t_sum_'+str(E)+'_case_'+str(case)+'.npy', tissue_energy_matrix)
print('Time taken for tissue energy matrix = ', round((stop - start)/60,3))


start = time.time()
vessel(0,81)
stop = time.time()
np.save('v_sum_'+str(E)+'_case_'+str(case)+'.npy', vessel_energy_matrix)
print('Time taken for vessel energy matrix = ', round((stop - start)/60,3))
# print(np.max(vessel_energy_matrix))



start = time.time()
for k in range(0,81):
    print(k)
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == tissue_index):
                m_dot = 0
                for b in range(len(a_out)):
                    if ([i,j,k] in nbr_a[b]):
                        ele = a_db.loc[a_db.iloc[:,2] == a_out[b,0]]
                        x0,y0,z0 = a_out[b,1:4].tolist()
                        s = SoI([x0,y0,z0],[i,j,k])
                        n_ex = eta(s,e,Ca[b], len(nbr_a[b]))
                        Q = abs(QA[int(ele.iloc[0,0])])
                        Q_dot = Q*rho_b*n_ex
                        m_dot = m_dot + Q_dot*Cp*X[un_t + int(ele.iloc[0,0])]
                        
                vessel_mass_matrix[i,j,k] = m_dot

stop = time.time()
np.save('m_dot_'+str(E)+'_case_'+str(case)+'.npy', vessel_mass_matrix)
print('Time taken for vessel mass_matrix = ', round((stop - start)/60))



    


