import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
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
# a_pt = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_pt.csv').to_numpy()
# v_pt = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_pt.csv').to_numpy()
# a_term = pd.read_csv('a_term.csv').to_numpy()
# v_term = pd.read_csv('v_term.csv').to_numpy()
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
# method = 'spsolve'
# pa = np.load('case' + str(case) + '/prs_acomp_' + str(E) + '_a_' + str(a) + '_G_' + str(gamma_a) + '_' + method + '.npy')
# pv = np.load('case' + str(case) + '/prs_vcomp_' + str(E) + '_a_' + str(a) + '_G_' + str(gamma_a) + '_' + method + '.npy')

# QA = np.load('case' + str(case) + '/Q_arteries_' + str(E) + '_a_' + str(a) + '_G_' + str(gamma_a) + '_' + method + '.npy')
# QV = np.load('case' + str(case) + '/Q_veins_' + str(E) + '_a_' + str(a) + '_G_' + str(gamma_a) + '_' + method + '.npy')

# PArt = np.load('case' + str(case) + '/prs_arteries_' + str(E) + '_a_' + str(a) + '_G_' + str(gamma_a) + '_' + method + '.npy')
# PVrt = np.load('case' + str(case) + '/prs_veins_' + str(E) + '_a_' + str(a) + '_G_' + str(gamma_a) + '_' + method + '.npy')



# solution = 'case'+str(case)+'/X_' + str(E) + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
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
for k in range(nz):
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

# At = sp.csc_matrix((data,(row,col)), shape = (Nt, Nt))
# plt.figure(figsize=(10,10),dpi=300)
# plt.spy(At)
# plt.show()

# At_dense = At.todense()[:50,:50]

            
            # # # ARTERIAL HEAT TRANSFER # # # 


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
                
# At = sp.csc_matrix((data,(row,col)), shape = (Nt, Nt))
# plt.figure(figsize=(10,10),dpi=300)
# plt.spy(At)
# # plt.title('Artery Heat Equations')        
# plt.show()    

# A_sub = At.todense()[un_t:,un_t:]      




             # # # VENAL HEAT EQUATIONS # # # 
                    
for i in range(len(v_db)):
    ele = v_db.loc[v_db.iloc[:,0] == i]
    sum_m_dotCp = 0
    r = ele.iloc[0,4]
    L = ele.iloc[0,3]
    eVol = np.pi*r**4*L
    
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
        


rowname = 'case'+str(case)+'/Heat_row_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
colname = 'case'+str(case)+'/Heat_col_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
dataname = 'case'+str(case)+'/Heat_data_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
        
        
np.save(rowname, row)
np.save(colname, col)
np.save(dataname, data)
        
At = sp.csc_matrix((data,(row,col)), shape = (Nt, Nt))

# plt.figure(figsize=(10,10),dpi=300)
# plt.spy(At)
# # plt.title('Artery Heat Equations')        
# plt.show()    



            # # # # CREATE HEAT MATRIX AND SOLVE # # # # 
B_col = np.zeros((len(B_index)), dtype=int).tolist()
At = sp.csc_matrix((data,(row,col)), shape = (Nt, Nt))
Bt = sp.csr_matrix((B_data,(B_index,B_col)), shape=(Nt,1))
Bt = Bt.todense()

Bname = 'case'+str(case)+'/Heat_B_'+str(E)+'_a_'+ str(a) + '_q_' + str(q) + '_Tin_'+ str(T_in) + '_Tamb_' + str(T_amb) + '_hamb_' + str(h_amb) + '_hbt_' + str(h_bt) + '_.npy'       
np.save(Bname, Bt)

# plt.figure(figsize = (10,10), dpi=300)
# plt.spy(At)
# # plt.title('Temperature Matrix E = ' + str(E) + '\n', font = font, size = f) 
# plt.xticks(font = font, size=f)
# plt.yticks(font = font, size=f)
# # plt.savefig(str(Layer) + '/Heat_Matrix_E_' + str(E) + '.png',dpi=300)
# plt.show()   

# Xt = spla.spsolve(At,Bt)
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


# for i in range(len(nbr_v)):
#     sum_T = 0
#     for j in range(len(nbr_v[i])):
#         x,y,z = nbr_v[i][j]
#         sum_T = sum_T + T_Tis[c_dom[x,y,z]]
#     print(sum_T/len(nbr_v[i]))


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

del(mav_n, mav_s, mav_e, mav_w, mav_f, mav_b,UA_n, UA_s, UA_e, UA_w, UA_f, UA_b, UAT_n, UAT_s, UAT_e, UAT_w, UAT_f, UAT_b, UA_na, UA_sa, UA_ea, UA_wa, UA_fa, UA_ba)
del(UA_nv , UA_sv , UA_ev ,UA_wv , UA_fv , UA_bv)
del(UAT_na , UAT_sa , UAT_ea , UAT_wa , UAT_fa , UAT_ba)
del(UAT_nv , UAT_sv , UAT_ev , UAT_wv , UAT_fv , UAT_bv)
del(x0,y0,z0,x1,y1,z1,x,y,z)
# del(a,b,i,j,k,ele,ele_1,ele_2)
# del(n1,n2,n_ex,out_index,q,r,s,sum_m_dot_in, sum_m_dotCp, sum_mav, sum_T, sum_UA, sum_UAT)

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


# A = At.todense()
# I = np.copy(Bt)
# I[:] = 1.0
# Y = np.matmul(A,I)
# error = Y-Bt

'''

print('Time taken for heat matrix = ', round(heat_stop - heat_start,2))

dom_T = np.zeros((nx,ny), dtype = float)
for i in range(nx):
    for j in range(ny):
        if(dom[i,j] == 1):
            dom_T[i,j] = Xt[c_dom[i,j]]

# plt.figure(figsize = (12,10), dpi = 300)
# plt.contourf(dom_T[1:-1,1:-1],100)
# plt.colorbar()
# plt.show()


colors = 'Reds'               
plt.figure(figsize = (12,10), dpi=300)
plt.contourf(dom_T[1:-1,1:-1], np.arange(np.min(dom_T[1:-1,1:-1]), np.max(dom_T[1:-1,1:-1])+0.2, 0.2), cmap=colors)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize = f)
cbar.ax.set_ylabel('Temperature (C)',size = 20, font=font)
plt.xticks(x_ticks, font=font, size = f)
plt.yticks(y_ticks, font=font, size = f)
fig_title = 'T | E ' + str(E) + ' | a ' + str(a) + ' | Ka ' + str(Ka) + ' | Kv ' + str(Kv) + ' | Ga ' + str(gamma_a) + ' | Gv ' + str(gamma_v) + '\n Kt ' + str(Kt) + ' | h ' + str(h) + ' | q ' + str(q) + ' | Cp ' + str(Cp)  + ' Pin ' + str(P_in) + ' Pout ' + str(P_out) + '\n' 

plt.title(fig_title, font = font, size = f)
plt.xlabel('mm', font=font, size = f)
plt.ylabel('mm', font=font, size = f)

save_title = str(Layer) + '/Temperature E ' + str(E) + ' a ' + str(a) + ' Ka ' + str(Ka) + ' Kv ' + str(Kv) + ' Ga ' + str(gamma_a) + ' Gv ' + str(gamma_v) + ' Kt ' + str(Kt) + ' h ' + str(h) + ' q ' + str(q) + ' Cp ' + str(Cp)  + ' Pin ' + str(P_in) + ' Pout ' + str(P_out) + '.png' 

plt.savefig(save_title, dpi=300)

plt.show()

if (Layer == ref_Layer and E == ref_E):
    np.save('ref_T.npy',dom_T)
    np.save('ref_X.npy', Xt)
if (Layer != ref_Layer or E != ref_E):
    ref_T = np.load('ref_T.npy')
    error_T = ref_T - dom_T
    dT = (np.max(error_T[1:-1,1:-1]) - np.min(error_T[1:-1,1:-1]))/10
    color_range = np.arange(np.min(error_T[1:-1,1:-1]), np.max(error_T[1:-1,1:-1])+dT, dT)
    colors = 'Reds'               
    plt.figure(figsize = (12,10), dpi=300)
    plt.contourf(error_T[1:-1,1:-1], color_range , cmap=colors)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize = fc)
    cbar.ax.set_ylabel('Temperature (C)',size = 20, font=font)
    plt.xticks(x_ticks, font=font, size = f)
    plt.yticks(y_ticks, font=font, size = f)
    fig_title = 'T | E ' + str(E) + ' | a ' + str(a) + ' | Ka ' + str(Ka) + ' | Kv ' + str(Kv) + ' | Ga ' + str(gamma_a) + ' | Gv ' + str(gamma_v) + '\n Kt ' + str(Kt) + ' | h ' + str(h) + ' | q ' + str(q) + ' | Cp ' + str(Cp)  + ' Pin ' + str(P_in) + ' Pout ' + str(P_out) + '\n' 

    plt.title(fig_title, font = font, size = f)
    plt.xlabel('mm', font=font, size = f)
    plt.ylabel('mm', font=font, size = f)


print('Temperature at venous outlet =', round(Xt[un_t + un_a],2))
print('Average tissue temperature = ', round(np.average(dom_T[1:-1,1:-1]),2))


# A_inv = spla.inv(A)
# At_inv = spla.inv(At)

'''


