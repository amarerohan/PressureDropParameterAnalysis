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


E = 38
e =  E*dx

case = 1

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
c_dom = np.load('case'+str(case)+'/'+str(Dx)+'_cdom.npy')

nbr_a = np.load('case'+str(case)+'/artery_nbrhd_'+str(E)+'.npy', allow_pickle=True)
nbr_v = np.load('case'+str(case)+'/vein_nbrhd_'+str(E)+'.npy', allow_pickle=True)

a_nbr = np.load('case'+str(case)+'/a_nbr.npy', allow_pickle=True)
v_nbr = np.load('case'+str(case)+'/v_nbr.npy', allow_pickle=True)


Ca = np.load('case'+str(case)+'/Ca_' + str(E) +'.npy')
Cv = np.load('case'+str(case)+'/Cv_' + str(E) +'.npy')

nx, ny, nz = np.shape(dom)


         # CALCULATE THE NUMBER OF UNKNONWS # 
un_a = len(a_db) + 1
un_v = len(v_db) + 1
un_t = np.max(c_dom) + 1
N_unk = un_a + un_v + 2*un_t


# # # # LOAD FLOW SOLUTIONS # # # # # 

prs_acomp = 'case'+str(case)+'/prs_acomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_vcomp = 'case'+str(case)+'/prs_vcomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_a = 'case'+str(case)+'/prs_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_v = 'case'+str(case)+'/prs_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qa = 'case'+str(case)+'/Q_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qv = 'case'+str(case)+'/Q_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'

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

T_in = 20
T_amb = 0

rho_t = 1000# 1090
rho_b = 1000 #1050
rho = (rho_t + rho_b)/2.0


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

def mass_flow_across_voxels(nbr, cell, ds):
    x1,y1,z1 = nbr
    x0,y0,z0 = cell
    
    t_ij_a = 2*ds*ds*(1/(ds/Lambda_a + ds/Lambda_a))
    Qa = t_ij_a*(pa[x1,y1,z1]-pa[x0,y0,z0])
    ma = rho_b*Qa
        
    t_ij_v = 2*ds*ds*(1/(ds/Lambda_v + ds/Lambda_v))
    Qv = t_ij_v*(pv[x1,y1,z1] - pv[x0,y0,z0])
    mv = rho_b*Qv
    
    m = ma + mv
    return(m)

def conduction():
    UA = 1/(dx/(2*Kt) + dx/(2*Kt))*dy*dz
    UAT = 0
    return(UA, UAT)

def convection(T,h):
    UA = 1/(1/h + dx/(2*Kt))*dy*dz
    UAT = UA*T
    return(UA, UAT)

un_a = len(a_db)
un_v = len(v_db)
Nt = un_t + len(a_db) + len(v_db)
row = []
col = []
data = []
B_index = []
B_data = []


tissue_index = 0
air_index = -1
artery_index = 1
vein_index = 2

             # # # TISSUE VOXEL # # # 
def tissue(input_variables):
    nz1, nz2 = input_variables
    print(nz1, nz2)
    tissue_start = time.time()
    row = []
    col = []
    data = []    
    B_index = []
    B_data = []
    
    for k in range(nz1,nz2):
        for j in range(ny):
            for i in range(nx):
                if(dom[i,j,k] == tissue_index):
                    mav_n = mav_s = mav_e = mav_w = mav_f = mav_b = 0.0
                    UA_n = UA_s = UA_e = UA_w =  UA_f = UA_b = 0.0
                    UAT_n = UAT_s = UAT_e = UAT_w = UAT_f = UAT_b = 0.0
                    UA_na = UA_sa = UA_ea = UA_wa = UA_fa = UA_ba = 0.0
                    UA_nv = UA_sv = UA_ev = UA_wv = UA_fv = UA_bv = 0.0
                    UAT_na = UAT_sa = UAT_ea = UAT_wa = UAT_fa = UAT_ba = 0.0
                    UAT_nv = UAT_sv = UAT_ev = UAT_wv = UAT_fv = UAT_bv = 0.0
                    
                    # NORTH
                    if(dom[i+1,j,k] == tissue_index):
                        mav_n = mass_flow_across_voxels([i+1,j,k],[i,j,k],dx)
                        UA_n, UAT_n = conduction()
                        if(mav_n > 0):
                            row.append(c_dom[i,j,k]) ; col.append(c_dom[i+1,j,k]) ; data.append(mav_n*Cp)
                        elif(mav_n <= 0):
                            mav_n = 0
                        row.append(c_dom[i,j,k]) ; col.append(c_dom[i+1,j,k]) ; data.append(UA_n)
                    
                    if(dom[i+1,j,k] == air_index):
                        mav_n = 0
                        UA_n, UAT_n = convection(T_amb,h_amb)
                    
                    if(dom[i+1,j,k] == artery_index):
                        mav_n = 0
                        UA_na, UAT_na = convection(0, h_bt)                    
                        row.append(c_dom[i,j,k]) ; col.append(un_t + a_ele[i+1,j,k]) ; data.append(UA_na)
                    
                    if(dom[i+1,j,k] == vein_index):
                        mav_n = 0
                        UA_nv, UAT_nv = convection(0, h_bt)                    
                        row.append(c_dom[i,j,k]) ; col.append(un_t + un_a + v_ele[i+1,j,k]) ; data.append(UA_nv)
                        
                    
                    
                    
                    # SOUTH
                    if(dom[i-1,j,k] == tissue_index):
                        mav_s = mass_flow_across_voxels([i-1,j,k],[i,j,k],dx)
                        UA_s, UAT_s = conduction()
                        if(mav_s > 0):
                            row.append(c_dom[i,j,k]) ; col.append(c_dom[i-1,j,k]) ; data.append(mav_s*Cp)
                        elif(mav_s <= 0):
                            mav_s = 0
                        row.append(c_dom[i,j,k]) ; col.append(c_dom[i-1,j,k]) ; data.append(UA_s)
                    
                    if(dom[i-1,j,k] == air_index):
                        mav_s = 0
                        UA_s, UAT_s = convection(T_amb,h_amb)
                    
                    if(dom[i-1,j,k] == artery_index):
                        mav_s = 0
                        UA_sa, UAT_sa = convection(0, h_bt)                    
                        row.append(c_dom[i,j,k]) ; col.append(un_t + a_ele[i-1,j,k]) ; data.append(UA_sa)
                    
                    if(dom[i-1,j,k] == vein_index):
                        mav_s = 0
                        UA_sv, UAT_sv = convection(0, h_bt)                    
                        row.append(c_dom[i,j,k]) ; col.append(un_t + un_a + v_ele[i-1,j,k]) ; data.append(UA_sv)
                        
                    
                    
                    
                    # EAST
                    if(dom[i,j+1,k] == tissue_index):
                        mav_e = mass_flow_across_voxels([i,j+1,k], [i,j,k], dx)
                        UA_e, UAT_e = conduction()
                        if(mav_e > 0):
                            row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j+1,k]) ; data.append(mav_e*Cp)
                        if(mav_e <= 0):
                            mav_e = 0
                        row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j+1,k]) ; data.append(UA_e)
                        
                    if(dom[i,j+1,k] == air_index):
                        mav_e = 0
                        UA_e, UAT_e = convection(T_amb, h_amb)
                    
                    if(dom[i,j+1,k] == artery_index):
                        UA_ea, UAT_ea = convection(0, h_bt)
                        row.append(c_dom[i,j,k]) ; col.append(un_t + a_ele[i,j+1,k]) ; data.append(UA_ea)
                        
                    if(dom[i,j+1,k] == vein_index):
                        UA_ev, UAT_ev = convection(0, h_bt)
                        row.append(c_dom[i,j,k]) ; col.append(un_t + un_a + v_ele[i,j+1,k]) ; data.append(UA_ev)
                        
                        
                    
                    
                    # WEST
                    if(dom[i,j-1,k] == tissue_index):
                        mav_w = mass_flow_across_voxels([i,j-1,k], [i,j,k], dx)
                        UA_w, UAT_w = conduction()
                        if(mav_w > 0):
                            row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j-1,k]) ; data.append(mav_w*Cp)
                        if(mav_w <= 0):
                            mav_w = 0
                        row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j-1,k]) ; data.append(UA_w)
                        
                    if(dom[i,j-1,k] == air_index):
                        mav_w = 0
                        UA_w, UAT_w = convection(T_amb, h_amb)
                    
                    if(dom[i,j-1,k] == artery_index):
                        mav_w = 0
                        UA_wa, UAT_wa = convection(0, h_bt)
                        row.append(c_dom[i,j,k]) ; col.append(un_t + a_ele[i,j-1,k]) ; data.append(UA_wa)
                        
                    if(dom[i,j-1,k] == vein_index):
                        mav_w = 0
                        UA_wv, UAT_wv = convection(0, h_bt)
                        row.append(c_dom[i,j,k]) ; col.append(un_t + un_a + v_ele[i,j-1,k]) ; data.append(UA_wv)
                        
                    
                        
                    
                    # FRONT
                    if(dom[i,j,k+1] == tissue_index):
                        mav_f = mass_flow_across_voxels([i,j,k+1], [i,j,k], dz)
                        UA_f, UAT_f = conduction()
                        if(mav_f > 0):
                            row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j,k+1]) ; data.append(mav_f*Cp)
                        if(mav_f <= 0):
                            mav_f = 0
                        row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j,k+1]) ; data.append(UA_f)
                        
                    if(dom[i,j,k+1] == air_index):
                        mav_f = 0
                        UA_f, UAT_f = convection(T_amb, h_amb)
                        
                    if(dom[i,j,k+1] == artery_index):
                        mav_f = 0 
                        UA_fa, UAT_fa = convection(0, h_bt)
                        row.append(c_dom[i,j,k]) ; col.append(un_t + a_ele[i,j,k+1]) ; data.append(UA_fa)
                    
                    if(dom[i,j,k+1] == vein_index):
                        mav_f = 0
                        UA_fv, UAT_fv = convection(0, h_bt)
                        row.append(c_dom[i,j,k]) ; col.append(un_t + un_a + v_ele[i,j,k+1]) ; data.append(UA_fv)
                        
                        
                        
                    
                    # BACK
                    if(dom[i,j,k-1] == tissue_index):
                        mav_b = mass_flow_across_voxels([i,j,k-1], [i,j,k], dz)
                        UA_b, UAT_b = conduction()
                        if(mav_b > 0):
                            row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j,k-1]) ; data.append(mav_b*Cp)
                        if(mav_b <= 0):
                            mav_b = 0
                        row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j,k-1]) ; data.append(UA_b)
                        
                    if(dom[i,j,k-1] == air_index):
                        mav_b = 0
                        UA_b, UAT_b = convection(T_amb, h_amb)
                    
                    if(dom[i,j,k-1] == artery_index):
                        mav_b = 0
                        UA_ba, UAT_ba = convection(0,h_bt)
                        row.append(c_dom[i,j,k]) ; col.append(un_t + a_ele[i,j,k-1]) ; data.append(UA_ba)
                    
                    if(dom[i,j,k-1] == vein_index):
                        mav_b = 0
                        UA_bv, UAT_bv = convection(0,h_bt)
                        row.append(c_dom[i,j,k]) ; col.append(un_t + un_a + v_ele[i,j,k-1]) ; data.append(UA_bv)
                        
                        
                    
                    # ARTERY TO VOXEL
                    m_dot = 0
                    for b in range(len(a_out)):
                        if ([i,j,k] in nbr_a[b]):
                            ele = a_db.loc[a_db.iloc[:,2] == a_out[b,0]]
                            x0,y0,z0 = a_out[b,1:4].tolist()
                            s = SoI([x0,y0,z0],[i,j,k])
                            n_ex = eta(s,e,Ca[b], len(nbr_a[b]))
                            Q = abs(QA[int(ele.iloc[0,0])])
                            Q_dot = Q*rho_b*n_ex
                            m_dot = m_dot + Q_dot
                            row.append(c_dom[i,j,k]) ; col.append(un_t + int(ele.iloc[0,0])) ; data.append(Q_dot*Cp)
                    mA = m_dot
                    
                    sum_mav = mav_n + mav_s + mav_e + mav_w + mav_f + mav_b + mA
                    sum_UA = UA_n + UA_s + UA_e + UA_w + UA_f + UA_b + UA_na + UA_sa + UA_ea + UA_wa + UA_fa + UA_ba + UA_nv + UA_sv + UA_ev + UA_wv + UA_fv + UA_bv
                    sum_UAT = UAT_n + UAT_s + UAT_e + UAT_w + UAT_f + UAT_b + UAT_na + UAT_sa + UAT_ea + UAT_wa + UAT_fa + UAT_ba + UAT_nv + UAT_sv + UAT_ev + UAT_wv + UAT_fv + UAT_bv
                    
                    row.append(c_dom[i,j,k]) ; col.append(c_dom[i,j,k]) ; data.append(-sum_UA -sum_mav*Cp)
                    B_index.append(c_dom[i,j,k]) ; B_data.append(-q*dx*dy*dz - sum_UAT)

    rowname = 'case'+str(case)+'/' + str(nz1) + '_' + str(nz2) + 'Heat_row_tissue_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    colname = 'case'+str(case)+'/' + str(nz1) + '_' + str(nz2) + 'Heat_col_tissue_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    dataname = 'case'+str(case)+'/' + str(nz1) + '_' + str(nz2) + 'Heat_data_tissue_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    B_index_name = 'case'+str(case)+'/' + str(nz1) + '_' + str(nz2) + 'Heat_Bindex_tissue_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    B_data_name = 'case'+str(case)+'/' + str(nz1) + '_' + str(nz2) + 'Heat_Bdata_tissue_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       

            
    np.save(rowname, row)
    np.save(colname, col)
    np.save(dataname, data)
    np.save(B_index_name, B_index)
    np.save(B_data_name, B_data)

    tissue_stop = time.time()
    print('Time taken for tissue equations = ', round((tissue_stop - tissue_start)/60,3), ' mins')
    
    
            # # # ARTERIAL HEAT TRANSFER # # # 

def vascular():
    vas_start = time.time()
    
    row = []
    col = []
    data = []
    B_index = []
    B_data = []
    
    for i in range(len(a_db)):
        ele = a_db.loc[a_db.iloc[:,0] == i]
        sum_m_dotCp = 0
        r = ele.iloc[0,4]
        L = ele.iloc[0,3]
        eVol = np.pi*r**4*L
        
        # CONVECTION WITH SURROUNDING TISSUE
        UA_sum = 0
        for j in range(len(a_nbr[i])):
            x,y,z = a_nbr[i][j]
            UA, UAT = convection(0, h_bt)
            UA_sum = UA_sum + UA
            row.append(un_t + i) ; col.append(c_dom[x,y,z]) ; data.append(-UA)
        row.append(un_t + i) ; col.append(un_t + i) ; data.append(UA_sum)
            
        # MASS CONSERVATION WITHIN VASCULATURE
        if i==0:
            m_dot = QA[i]*rho_b
            sum_m_dotCp = sum_m_dotCp + m_dot*Cp
            row.append(un_t + i) ; col.append(un_t + i) ; data.append(m_dot*Cp)
            B_index.append(un_t + i)
            B_data.append(m_dot*Cp*T_in)
        
        else:
            m_ele=abs(QA[i])*rho_b
            sum_m_dotCp = sum_m_dotCp + m_ele*Cp
            
            n1 = int(ele.iloc[0,1])
            n2 = int(ele.iloc[0,2])
            
            if(PArt[n1] > PArt[n2]):
                ele_1 = a_db.loc[((a_db.iloc[:,1] == n1) | (a_db.iloc[:,2] == n1)) & (a_db.iloc[:,0] != int(ele.iloc[0,0]))]
                sum_m_dot_in = 0
                
                for j in range(len(ele_1)):
                    if( (n1 == int(ele_1.iloc[j,1])) and (PArt[int(ele_1.iloc[j,2])] > PArt[n1]) ):
                        m_dot = abs(QA[int(ele_1.iloc[j,0])])*rho
                        sum_m_dot_in = sum_m_dot_in + m_dot
                        
                    elif( (n1 == int(ele_1.iloc[j,2])) and (PArt[int(ele_1.iloc[j,1])] > PArt[n1]) ):
                        m_dot = abs(QA[int(ele_1.iloc[j,0])])*rho
                        sum_m_dot_in = sum_m_dot_in + m_dot
                        
                
                for j in range(len(ele_1)):
                    if( (n1 == int(ele_1.iloc[j,1])) and (PArt[int(ele_1.iloc[j,2])] > PArt[n1]) ):
                        m_dot = abs(QA[int(ele_1.iloc[j,0])])*rho
                        row.append(un_t + i) ; col.append(un_t + ele_1.iloc[j,0]) ; data.append(-m_ele*m_dot/sum_m_dot_in*Cp)
                    elif( (n1 == int(ele_1.iloc[j,2])) and (PArt[int(ele_1.iloc[j,1])] > PArt[n1]) ):
                        m_dot = abs(QA[int(ele_1.iloc[j,0])])*rho
                        row.append(un_t + i) ; col.append(un_t + ele_1.iloc[j,0]) ; data.append(-m_ele*m_dot/sum_m_dot_in*Cp)
                        
                row.append(un_t + i) ; col.append(un_t + i) ; data.append(m_ele*Cp)
            
            if(PArt[n1] < PArt[n2]):
                ele_2 = a_db.loc[((a_db.iloc[:,1] == n2) | (a_db.iloc[:,2] == n2)) & (a_db.iloc[:,0] != int(ele.iloc[0,0]))]
                sum_m_dot_in = 0
                
                for j in range(len(ele_2)):
                    if( (n2 == int(ele_2.iloc[j,1])) and (PArt[int(ele_2.iloc[j,2])] > PArt[n2])):
                        m_dot = abs(QA[int(ele_2.iloc[j,0])])*rho
                        sum_m_dot_in = sum_m_dot_in + m_dot
                    elif( (n2 == int(ele_2.iloc[j,2])) and (PArt[int(ele_2.iloc[j,1])] > PArt[n2])):
                        m_dot = abs(QA[int(ele_2.iloc[j,0])])*rho
                        sum_m_dot_in = sum_m_dot_in + m_dot
                        
                
                for j in range(len(ele_2)):
                    if( (n2 == int(ele_2.iloc[j,1])) and (PArt[int(ele_2.iloc[j,2])] > PArt[n2])):
                        m_dot = abs(QA[int(ele_2.iloc[j,0])])*rho
                        row.append(un_t + i) ; col.append(un_t + ele_2.iloc[j,0]) ; data.append(-m_ele*m_dot/sum_m_dot_in*Cp)
                    elif( (n2 == int(ele_2.iloc[j,2])) and (PArt[int(ele_2.iloc[j,1])] > PArt[n2])):
                        m_dot = abs(QA[int(ele_2.iloc[j,0])])*rho
                        row.append(un_t + i) ; col.append(un_t + ele_2.iloc[j,0]) ; data.append(-m_ele*m_dot/sum_m_dot_in*Cp)
                        
                row.append(un_t + i) ; col.append(un_t + i) ; data.append(m_ele*Cp)
                    

    
    
    
                 # # # VENAL HEAT EQUATIONS # # # 
                        
    for i in range(len(v_db)):
        ele = v_db.loc[v_db.iloc[:,0] == i]
        sum_m_dotCp = 0
        r = ele.iloc[0,4]
        L = ele.iloc[0,3]
        
        # CONVECTION WITH SURROUNDING TISSUE
        UA_sum = 0
        for j in range(len(v_nbr[i])):
            x,y,z = v_nbr[i][j]
            UA, UAT = convection(0, h_bt)
            UA_sum = UA_sum + UA
            row.append(un_t + un_a + i) ; col.append(c_dom[x,y,z]) ; data.append(-UA)
        row.append(un_t + un_a + i) ; col.append(un_t + un_a + i) ; data.append(UA_sum)
        
        # MASS CONSERVATION WITHIN VASCULATURE
        if (ele.iloc[0,2] not in v_out[:,0]):
            m_ele = abs(QV[i])*rho_b
            sum_m_dotCp = sum_m_dotCp + m_ele*Cp
            
            n1 = int(ele.iloc[0,1])
            n2 = int(ele.iloc[0,2])
            
            if (PVrt[n1] > PVrt[n2]):
                ele_1 = v_db.loc[((v_db.iloc[:,1] == n1) | (v_db.iloc[:,2] == n1)) & (v_db.iloc[:,0] != i)]
                sum_m_dot_in = 0
                
                for j in range(len(ele_1)):
                    if( (n1 == int(ele_1.iloc[j,1])) and (PVrt[int(ele_1.iloc[j,2])] > PVrt[n1]) ):
                        m_dot = abs(QV[int(ele_1.iloc[j,0])])*rho
                        sum_m_dot_in = sum_m_dot_in + m_dot
                        
                    elif( (n1 == int(ele_1.iloc[j,2])) and (PVrt[int(ele_1.iloc[j,1])] > PVrt[n1]) ):
                        m_dot = abs(QV[int(ele_1.iloc[j,0])])*rho
                        sum_m_dot_in = sum_m_dot_in + m_dot
                
                for j in range(len(ele_1)):
                    if( (n1 == int(ele_1.iloc[j,1])) and (PVrt[int(ele_1.iloc[j,2])] > PVrt[n1]) ):
                        m_dot = abs(QV[int(ele_1.iloc[j,0])])*rho
                        row.append(un_t + un_a + i) ; col.append(un_t + un_a + ele_1.iloc[j,0]) ; data.append(-m_ele*m_dot/sum_m_dot_in*Cp)
                        
                    elif( (n1 == int(ele_1.iloc[j,2])) and (PVrt[int(ele_1.iloc[j,1])] > PVrt[n1]) ):
                        m_dot = abs(QV[int(ele_1.iloc[j,0])])*rho
                        row.append(un_t + un_a + i) ; col.append(un_t + un_a + ele_1.iloc[j,0]) ; data.append(-m_ele*m_dot/sum_m_dot_in*Cp)
                        
                row.append(un_t + un_a + i) ; col.append(un_t + un_a + i) ; data.append(m_ele*Cp)
                
                    
                
            if (PVrt[n2] > PVrt[n1]):
                ele_2 = v_db.loc[((v_db.iloc[:,1] == n2) | (v_db.iloc[:,2] == n2)) & (v_db.iloc[:,0] != i)]
                sum_m_dot_in = 0
                
                for j in range(len(ele_2)):
                    if( (n2 == int(ele_2.iloc[j,1])) and (PVrt[int(ele_2.iloc[j,2])] > PVrt[n2]) ):
                        m_dot = abs(QV[int(ele_2.iloc[j,0])])*rho
                        sum_m_dot_in = sum_m_dot_in + m_dot
                        
                    elif( (n2 == int(ele_2.iloc[j,2])) and (PVrt[int(ele_2.iloc[j,1])] > PVrt[n2]) ):
                        m_dot = abs(QV[int(ele_2.iloc[j,0])])*rho
                        sum_m_dot_in = sum_m_dot_in + m_dot
                
                for j in range(len(ele_2)):
                    if( (n2 == int(ele_2.iloc[j,1])) and (PVrt[int(ele_2.iloc[j,2])] > PVrt[n2]) ):
                        m_dot = abs(QV[int(ele_2.iloc[j,0])])*rho
                        row.append(un_t + un_a + i) ; col.append(un_t + un_a + ele_2.iloc[j,0]) ; data.append(-m_ele*m_dot/sum_m_dot_in*Cp)
                        
                    elif( (n2 == int(ele_2.iloc[j,2])) and (PVrt[int(ele_2.iloc[j,1])] > PVrt[n2]) ):
                        m_dot = abs(QV[int(ele_2.iloc[j,0])])*rho
                        row.append(un_t + un_a + i) ; col.append(un_t + un_a + ele_2.iloc[j,0]) ; data.append(-m_ele*m_dot/sum_m_dot_in*Cp)
                        
                row.append(un_t + un_a + i) ; col.append(un_t + un_a + i) ; data.append(m_ele*Cp)
                
        if (ele.iloc[0,2] in v_out[:,0]):
            Layer = 5
            out_index = int(ele.iloc[0,2])- 2**Layer # ; print(out_index)
            # print(i, out_index, un_t + un_a + i)
            m_ele = abs(QV[i])*rho
            for b in range(len(nbr_v[out_index])):
                x0, y0, z0 = v_out[out_index,1:4].tolist()
                x1, y1, z1 = nbr_v[out_index][b]
                s = SoI([x0,y0,z0], [x1,y1,z1])
                n_ex = eta(s,e,Cv[out_index], len(nbr_v[out_index]))
                m_dot = n_ex*m_ele
                
                row.append(un_t + un_a + i) ; col.append(c_dom[x1,y1,z1]) ; data.append(-m_dot*Cp)
            row.append(un_t + un_a + i) ; col.append(un_t + un_a + i) ; data.append(m_ele*Cp)
            
    
    
    rowname = 'case'+str(case)+'/Heat_row_vascular_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    colname = 'case'+str(case)+'/Heat_col_vascular_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    dataname = 'case'+str(case)+'/Heat_data_vascular_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    B_index_name = 'case'+str(case)+'/Heat_Bindex_vascular_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    B_data_name = 'case'+str(case)+'/Heat_Bdata_vascular_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       

            
    np.save(rowname, row)
    np.save(colname, col)
    np.save(dataname, data)
    np.save(B_index_name, B_index)
    np.save(B_data_name, B_data)
    
    vas_stop = time.time()
    print('Time taken fro vascula equations = ', round((vas_stop - vas_start)/60,3), ' mins')


if __name__ == '__main__':
    rcdb = []
    dnz = 5
    pool_n = 5
    nz_list = []
    
    for i in range(dnz):
        nz_list.append([int(i*nz/dnz),int((i+1)*nz/dnz)])
    
    p = Pool(pool_n)
    input_variables = [nz_list[Z] for Z in range(dnz)]
    rcdb.append(p.map(tissue,input_variables))
    p.close()
    p.join()
    
    p1 = Pool(1)
    rcdb.append(p1.map(vascular))
    
    row = []
    col = []
    data = []
    B_index = []
    B_data = []
    
    for i in range(len(rcdb[0])):
        row = row + rcdb[0][i][0]
        col = col + rcdb[0][i][1]
        data = data + rcdb[0][i][2]
        B_index = B_index + rcdb[0][i][3]
        B_data = B_data + rcdb[0][i][4]
    
    rowname = 'case'+str(case)+'/Heat_row_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    colname = 'case'+str(case)+'/Heat_col_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    dataname = 'case'+str(case)+'/Heat_data_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
            
            
    np.save(rowname, row)
    np.save(colname, col)
    np.save(dataname, data)
    
    B_col = np.zeros((len(B_index)), dtype=int).tolist()       
    Bt = sp.csr_matrix((B_data,(B_index,B_col)), shape=(Nt,1))
    Bt = Bt.todense()
    Bname = 'case'+str(case)+'/Heat_B_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
    np.save(Bname, Bt)  
    
    
    
    At = sp.csc_matrix((data,(row,col)), shape = (Nt, Nt))
    
    
    
    
    max_iter = 1000
    tol = 1e-10
    
    solver_time_start = time.time()

    LU = spla.spilu(At)
    M = spla.LinearOperator(np.shape(LU), LU.solve)    
    x0 = np.zeros((Nt,1), dtype = float)
    x0[:] = T_amb
    X = spla.lgmres(At,Bt,M=M,x0=x0, maxiter = max_iter, tol = tol, inner_m = 200, atol = tol)#,tol=1e-15, atol = 1e-15)[0]
    reason = X[1]
    Xt = X[0]
    
    solver_time_stop = time.time()
    print('Time taken for solving the matrix = ', round((solver_time_stop - solver_time_start)/60,3), ' mins')

    print('Reason for lgmres termination = ', reason)

    filename = 'case'+str(case)+'/Heat_X_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'
    np.save(filename,Xt)

    heat_stop = time.time()


    T_Tis = Xt[:un_t]
    T_Art = Xt[un_t:un_t+un_a]
    T_Ven = Xt[un_t+un_a:]



    T = np.zeros((nx,ny,nz), dtype = float)
    T[:,:,:] = T_amb
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if(dom[i,j,k] == tissue_index):
                    c_index = c_dom[i,j,k]
                    T[i,j,k] = T_Tis[c_index]
                

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



    print('Energy Conv = ', round(conv_energy,3))
    print('Energy Gen  = ', round(energy_generated,3))
    print('Energy in   = ', round(energy_in,3))
    print('Energy out  = ', round(energy_out,3))
    print('Energy err  = ', round(energy_error,3))


    res = At.multiply(Xt)
    res = res.sum(axis = 1) - Bt
    residual = np.linalg.norm(res)
    print('residual = ', round(np.linalg.norm(residual),5))
    
    
    
    '''
    proc_tissue = Process(target = tissue)
    proc_vasc = Process(target = vascular)
    proc_tissue.start()
    proc_vasc.start()
    '''
'''
rowname = 'case'+str(case)+'/Heat_row_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
colname = 'case'+str(case)+'/Heat_col_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
dataname = 'case'+str(case)+'/Heat_data_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
        
        
np.save(rowname, row)
np.save(colname, col)
np.save(dataname, data)
        
At = sp.csc_matrix((data,(row,col)), shape = (Nt, Nt))


            # # # # CREATE HEAT MATRIX AND SOLVE # # # # 
B_col = np.zeros((len(B_index)), dtype=int).tolist()
At = sp.csc_matrix((data,(row,col)), shape = (Nt, Nt))
Bt = sp.csr_matrix((B_data,(B_index,B_col)), shape=(Nt,1))
Bt = Bt.todense()

Bname = 'case'+str(case)+'/Heat_B_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
np.save(Bname, Bt)


max_iter = 1000
tol = 1e-10

LU = spla.spilu(At)
M = spla.LinearOperator(np.shape(LU), LU.solve)    
x0 = np.zeros((Nt,1), dtype = float)
x0[:] = T_amb
X = spla.lgmres(At,Bt,M=M,x0=x0, maxiter = max_iter, tol = tol, inner_m = 200, atol = tol)#,tol=1e-15, atol = 1e-15)[0]
reason = X[1]
Xt = X[0]

print('Reason for lgmres termination = ', reason)

filename = 'case'+str(case)+'/Heat_X_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'
np.save(filename,Xt)

heat_stop = time.time()


T_Tis = Xt[:un_t]
T_Art = Xt[un_t:un_t+un_a]
T_Ven = Xt[un_t+un_a:]



T = np.zeros((nx,ny,nz), dtype = float)
T[:,:,:] = T_amb
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == tissue_index):
                c_index = c_dom[i,j,k]
                T[i,j,k] = T_Tis[c_index]
            

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



print('Energy Conv = ', round(conv_energy,3))
print('Energy Gen  = ', round(energy_generated,3))
print('Energy in   = ', round(energy_in,3))
print('Energy out  = ', round(energy_out,3))
print('Energy err  = ', round(energy_error,3))


res = At.multiply(Xt)
res = res.sum(axis = 1) - Bt
residual = np.linalg.norm(res)
# residual = np.matmul(A.todense(),X).transpose() - b
print('residual = ', round(np.linalg.norm(residual),5))

'''
