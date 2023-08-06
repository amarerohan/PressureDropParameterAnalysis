# Flow Solver 2: MATRIX GENERATION

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import time 
import pandas as pd
from matplotlib.font_manager import FontProperties
from multiprocessing import Manager, Pool, Process

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(15)
# font.set_style('italic')
f = 20

# import Flow_Solver_1 as fs1

plotting = True

# # # # # #  PARAMETERS
Dx = 2.5
dx = dy = 0.001*Dx
dz = dx
dVol = 1 


E = 10
e =  E*dx

case = 1

myu = 1e-3
a = 1e-3
Ka = 1E-5
Kv = 1E-5
gamma_a = 1E-1
gamma_v = 1E-1
Lambda_a = Ka/myu
Lambda_v = Kv/myu

a_inlet = 0
v_outlet = 0


# # # #  LOAD THE REQUIRED FILES # # # 


a_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_db.csv')
v_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_db.csv')
a_out = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_out_pts.csv').to_numpy()
v_out = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_out_pts.csv').to_numpy()


dom = np.load('case'+str(case)+'/'+str(Dx)+'_dom.npy')
c_dom = np.load('case'+str(case)+'/'+str(Dx)+'_cdom.npy')

nbr_a = np.load('case'+str(case)+'/artery_nbrhd_'+str(E)+'.npy', allow_pickle=True)
nbr_v = np.load('case'+str(case)+'/vein_nbrhd_'+str(E)+'.npy', allow_pickle=True)


Ca = np.load('case'+str(case)+'/Ca_' + str(E) +'.npy')
Cv = np.load('case'+str(case)+'/Cv_' + str(E) +'.npy')

nx, ny, nz = np.shape(dom)


         # CALCULATE THE NUMBER OF UNKNONWS # 
un_a = len(a_db) + 1 #len(a_pt)
un_v = len(v_db) + 1 #len(v_pt)
un_t = np.max(c_dom) + 1
N_unk = un_a + un_v + 2*un_t


equal = False


def solve(A,B):
    fm_start = time.time()
    X = spla.spsolve(A,B)
    fm_stop = time.time()
    return (X, fm_start, fm_stop)




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


start = time.time()

row = []
col = []
data = []

tissue_tag = 0
air_tag = -1
a_tag = 1
v_tag = 2
            # # # 1. ARTERIAL COMPARTMENT EQUATIONS # # # 

def arterial_compartment():
    ac_time_start = time.time()
    
    row = []
    col = []
    data = []

    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                if(dom[i,j,k] == tissue_tag):
                    eqn_n = c_dom[i,j,k]
                    
                    # ARTERIAL COMPARTMENT ONLY
                    ij_sum = 0
                    if(dom[i+1,j,k]==tissue_tag):
                        tx = 2*dy*dz*(1/(dx/Lambda_a + dx/Lambda_a))
                        row.append(eqn_n) ; col.append(c_dom[i+1,j,k]) ; data.append(-tx)
                        ij_sum = ij_sum + tx
                    if(dom[i-1,j,k]==tissue_tag):
                        tx = 2*dy*dz*(1/(dx/Lambda_a + dx/Lambda_a))
                        row.append(eqn_n) ; col.append(c_dom[i-1,j,k]) ; data.append(-tx)
                        ij_sum = ij_sum + tx
                    if(dom[i,j+1,k]==tissue_tag):
                        ty = 2*dx*dz*(1/(dy/Lambda_a + dy/Lambda_a))
                        row.append(eqn_n) ; col.append(c_dom[i,j+1,k]) ; data.append(-ty)
                        ij_sum = ij_sum + ty
                    if(dom[i,j-1,k]==tissue_tag):
                        ty = 2*dx*dz*(1/(dy/Lambda_a + dy/Lambda_a))
                        row.append(eqn_n) ; col.append(c_dom[i,j-1,k]) ; data.append(-ty)
                        ij_sum = ij_sum + ty
                    if(dom[i,j,k+1]==tissue_tag):
                        tz = 2*dx*dy*(1/(dz/Lambda_v + dz/Lambda_v))
                        row.append(eqn_n) ; col.append(c_dom[i,j,k+1]) ; data.append(-tz)
                        ij_sum = ij_sum + tz
                    if(dom[i,j,k-1]==tissue_tag):
                        tz = 2*dx*dy*(1/(dz/Lambda_v + dz/Lambda_v))
                        row.append(eqn_n) ; col.append(c_dom[i,j,k-1]) ; data.append(-tz)
                        ij_sum = ij_sum + tz
                    
                    ij_sum = ij_sum + a*dx*dy*dz
                    row.append(eqn_n) ; col.append(c_dom[i,j,k]) ; data.append(ij_sum)
                    row.append(eqn_n) ; col.append(un_t + c_dom[i,j,k]), data.append(-a*dx*dy*dz)
                    
                    for b in range(len(a_out)):
                        if([i,j,k] in nbr_a[b]):
                            ele = a_db.loc[a_db.iloc[:,2]==a_out[b,0]]
                            r = ele.iloc[0,4]
                            l = ele.iloc[0,3]                        
                            k1 = np.pi*(r)**4/(8*myu*l)
                            x0,y0,z0 = a_out[b,1:4]
                            s = SoI([x0,y0,z0],[i,j,k])
                            # print('outside ', s)
                            n_ex = eta(s,e,Ca[b],len(nbr_a[b]))
                            # print('outside eta ', n_ex)
                                                    
                            G = k1*n_ex*dVol
                            
                            
                            
                            row.append(eqn_n) ; col.append(2*un_t + ele.iloc[0,1]) ; data.append(-G) 
                            row.append(eqn_n) ; col.append(2*un_t + ele.iloc[0,2]) ; data.append(G)
    
    print('arterial_compartment = ', eqn_n) 
    row_name = 'case' + str(case) + '/row_arterial_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    col_name = 'case' + str(case) + '/col_arterial_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    data_name = 'case' + str(case) + '/data_arterial_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    
    np.save(row_name, row)
    np.save(col_name, col)
    np.save(data_name, data)
    
    ac_time_stop = time.time()
    print('Arterial Compartment function time taken = ', round((ac_time_stop - ac_time_start)/60, 3), ' mins')
     

                    
            # # # 2. VENOUS COMPARTMENT EQUATIONS # # # 
def venous_compartment():
    vc_time_start = time.time()
    row = []
    col = [] 
    data = []
    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                if(dom[i,j,k]==tissue_tag):
                    eqn_n = un_t + c_dom[i,j,k]
                    
                    ij_sum = 0
                    if(dom[i+1,j,k]==tissue_tag):
                        tx = 2*dy*dz*(1/(dx/Lambda_v + dx/Lambda_v))
                        row.append(eqn_n) ; col.append(un_t + c_dom[i+1,j,k]) ; data.append(-tx)
                        ij_sum = ij_sum + tx
                    if(dom[i-1,j,k]==tissue_tag):
                        tx = 2*dy*dz*(1/(dx/Lambda_v + dx/Lambda_v))
                        row.append(eqn_n) ; col.append(un_t + c_dom[i-1,j,k]) ; data.append(-tx)
                        ij_sum = ij_sum + tx
                    if(dom[i,j+1,k]==tissue_tag):
                        ty = 2*dx*dz*(1/(dy/Lambda_v + dy/Lambda_v))
                        row.append(eqn_n) ; col.append(un_t + c_dom[i,j+1,k]) ; data.append(-ty)
                        ij_sum = ij_sum + ty
                    if(dom[i,j-1,k]==tissue_tag):
                        ty = 2*dx*dz*(1/(dy/Lambda_v + dy/Lambda_v))
                        row.append(eqn_n) ; col.append(un_t + c_dom[i,j-1,k]) ; data.append(-ty)
                        ij_sum = ij_sum + ty
                    if(dom[i,j,k+1]==tissue_tag):
                        tz = 2*dx*dy*(1/(dz/Lambda_v + dz/Lambda_v))
                        row.append(eqn_n) ; col.append(un_t + c_dom[i,j,k+1]) ; data.append(-tz)
                        ij_sum = ij_sum + tz
                    if(dom[i,j,k-1]==tissue_tag):
                        tz = 2*dx*dy*(1/(dz/Lambda_v + dz/Lambda_v))
                        row.append(eqn_n) ; col.append(un_t + c_dom[i,j,k-1]) ; data.append(-tz)
                        ij_sum = ij_sum + tz
                        
                    ij_sum = ij_sum + a*dx*dy*dz
                    row.append(eqn_n) ; col.append(un_t + c_dom[i,j,k]) ; data.append(ij_sum)
                    row.append(eqn_n) ; col.append(c_dom[i,j,k]), data.append(-a*dx*dy*dz)
                    
                    for b in range(len(v_out)):
                        if([i,j,k] in nbr_v[b]):
                            ele = v_db.loc[v_db.iloc[:,2]==v_out[b,0]]
                            r = ele.iloc[0,4]
                            l = ele.iloc[0,3]                        
                            k1 = np.pi*(r)**4/(8*myu*l)
                            x0,y0,z0 = v_out[b,1:4]
                            s = SoI([x0,y0,z0],[i,j,k])
                            
                            n_ex = eta(s,e,Cv[b],len(nbr_v[b]))
                            
                            G = k1*n_ex*dVol
                            
                            
                            
                            row.append(eqn_n) ; col.append(2*un_t + un_a + ele.iloc[0,1]) ; data.append(-G)
                            row.append(eqn_n) ; col.append(2*un_t + un_a + ele.iloc[0,2]) ; data.append(G)
    
    print('venous_compartment = ', eqn_n) 
    row_name = 'case' + str(case) + '/row_venous_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    col_name = 'case' + str(case) + '/col_venous_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    data_name = 'case' + str(case) + '/data_venous_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    
    np.save(row_name, row)
    np.save(col_name, col)
    np.save(data_name, data)
    
    vc_time_stop = time.time()
    print('Venous compartment function time taken = ', round((vc_time_stop - vc_time_start)/60, 3), ' mins')



              # # # 3. ARTERIAL AND VENOUS TREE EQUATIONS # # # 
# 3.1. ARTERIAL MASS BALANCE #
# eqn_n = 2*un_t
def vasculature():
    vas_time_start = time.time()
    row = []
    col = []
    data = []
    for i in range(un_a):
        eqn_n = 2*un_t + i 
        ij_sum = 0  # ; print('a', i, eqn_n)
        if(i != a_inlet and i not in a_out[:,0]):
            ele = a_db.loc[a_db.iloc[:,1] == i]
            for j in range(len(ele)):
                r = ele.iloc[j,4]
                L = ele.iloc[j,3]
                phi = np.pi*(r)**4/(8*myu*L)
                ij_sum = ij_sum + phi                
                row.append(eqn_n) ; col.append(2*un_t + ele.iloc[j,2]); data.append(-phi)            
            ele = a_db.loc[a_db.iloc[:,2] == i]        
            for j in range(len(ele)):
                r = ele.iloc[j,4]
                L = ele.iloc[j,3]
                phi = np.pi*(r)**4/(8*myu*L)
                ij_sum = ij_sum + phi                
                row.append(eqn_n) ; col.append(2*un_t + ele.iloc[j,1]); data.append(-phi)           
            row.append(eqn_n) ; col.append(2*un_t + i) ; data.append(ij_sum) 
            # eqn_n = eqn_n + 1
    print(' arterial_tree =', eqn_n)   
    
    # 3.2. VENOUS MASS BALANCE #
    # eqn_n = 2*un_t + un_a - 1 - len(a_out)
    for i in range(un_v):
        eqn_n = 2*un_t + un_a + i  # ; print('v',i, eqn_n)
        ij_sum = 0
        if(i != v_outlet and i not in v_out[:,0]):
            ele = v_db.loc[v_db.iloc[:,1] == i]
            for j in range(len(ele)):
                r = ele.iloc[j,4]
                L = ele.iloc[j,3]
                phi = np.pi*(r)**4/(8*myu*L)
                ij_sum = ij_sum + phi
                row.append(eqn_n) ; col.append(2*un_t + un_a + ele.iloc[j,2]); data.append(-phi)
            ele = v_db.loc[v_db.iloc[:,2] == i]
            for j in range(len(ele)):
                r = ele.iloc[j,4]
                L = ele.iloc[j,3]
                phi = np.pi*(r)**4/(8*myu*L)
                ij_sum = ij_sum + phi
                row.append(eqn_n) ; col.append(2*un_t + un_a + ele.iloc[j,1]); data.append(-phi)
            row.append(eqn_n) ; col.append(2*un_t + un_a + i) ; data.append(ij_sum)
            # eqn_n = eqn_n + 1
    print('venous_tree = ',eqn_n)   


    # 3.3. TERMINAL ARTERIES
    a_out_db = pd.DataFrame(a_out)
    # eqn_n = 2*un_t + (un_a - 1 - len(a_out)) + (un_v - 1 - len(v_out))
    for i in range(len(a_out)):
        
        j = a_out_db.iloc[i,0]
        ele = a_db.loc[a_db.iloc[:,2] == j]
        eqn_n = 2*un_t + j # int(ele.iloc[0,0])
        r = ele.iloc[0,4]
        L = ele.iloc[0,3]
        k = np.pi*(r)**4/(8*myu*L)
        row.append(eqn_n) ; col.append(2*un_t + ele.iloc[0,1]) ; data.append(-k) 
        row.append(eqn_n) ; col.append(2*un_t + j) ; data.append((k+gamma_a/myu)) 
        for m in range(len(nbr_a[i])):
            x,y,z = nbr_a[i][m]
            x0, y0, z0 = (a_out_db.iloc[i,1:4]).tolist()
            s = SoI([x0,y0,z0],[x,y,z])
            n_ex = eta(s,e,Ca[i], len(nbr_a[i]))
            chi = n_ex*gamma_a/myu*dVol 
            # integral_a.append(j) ; sum_integral_a.append(n_ex)
            row.append(eqn_n) ; col.append(c_dom[x,y,z]) ; data.append(-chi)
        # eqn_n = eqn_n + 1
    
    print('terminal arteries = ', eqn_n)   
    
    # if(plotting == True):
    #     A = sp.csc_matrix((data,(row,col)), shape = (N_unk, N_unk))
        
    #     plt.figure(figsize = (10,10), dpi=300)
    #     plt.spy(A)
    #     plt.title('AFTER 3.3')    
    
    # 3.4. TERMINAL VEINS
    v_out_db = pd.DataFrame(v_out)
    # eqn_n = 2*un_t + (un_a - 1) + (un_v - 1 - len(v_out))
    for i in range(len(v_out)):
        j = v_out_db.iloc[i,0]
        ele = v_db.loc[v_db.iloc[:,2] == j]
        eqn_n = 2*un_t + un_a + j #int(ele.iloc[0,0])
        r = ele.iloc[0,4]
        L = ele.iloc[0,3]
        k = np.pi*(r)**4/(8*myu*L)
        row.append(eqn_n) ; col.append(2*un_t + un_a + ele.iloc[0,1]) ; data.append(-k)
        row.append(eqn_n) ; col.append(2*un_t + un_a + j) ; data.append((k+gamma_v/myu)) 
        for m in range(len(nbr_v[i])):
            x,y,z = nbr_v[i][m]
            x0, y0, z0 = (v_out_db.iloc[i,1:4]).tolist()
            s = SoI([x0,y0,z0],[x,y,z])
            n_ex = eta(s,e,Cv[i], len(nbr_v[i]))
            chi = n_ex*gamma_v/myu*dVol
            # integral_v.append(j) ; sum_integral_v.append(n_ex)
            row.append(eqn_n) ; col.append(un_t + c_dom[x,y,z]) ; data.append(-chi)
        # eqn_n = eqn_n + 1
    
    print('venous_tree = ', eqn_n)   


    
                # # # 4. BOUNDARY CONDITIONS # # # 
    # eqn_n = 2*un_t + (un_a - 1) + (un_v - 1)
    row.append(2*un_t + 0) ; col.append(2*un_t + a_inlet) ; data.append(1) # ; eqn_n = eqn_n + 1
    row.append(2*un_t + un_a + 0) ; col.append(2*un_t + un_a + v_outlet) ; data.append(1)  
    
    row_name = 'case' + str(case) + '/row_vasculature_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    col_name = 'case' + str(case) + '/col_vasculature_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    data_name = 'case' + str(case) + '/data_vasculature_a_' + str(a) + '_G_' + str(gamma_a) + '_.npy'
    
    np.save(row_name, row)
    np.save(col_name, col)
    np.save(data_name, data)
    
    vas_time_stop = time.time()
    print('Vascular function time taken = ', round((vas_time_stop - vas_time_start)/60, 3), ' mins')
    

if __name__=='__main__':
    out_time_start = time.time()
    p_ac = Process(target=arterial_compartment)
    p_vc = Process(target=venous_compartment)
    p_avt = Process(target=vasculature)
    
    p_ac.start()
    p_vc.start()
    p_avt.start()
    
    out_time_stop = time.time()
    print('Outer loop time = ', round((out_time_stop - out_time_start)/60,3), ' mins')
    # p_ac.close()
    # p_vc.close()
    # p_avt.close()


