import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sb
import pandas as pd
import time as time

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(60)

myu = 1e-3
a = 1E-3
Ka = 1E-5
Kv = 1E-5
Lambda_a = Ka/myu
Lambda_v = Kv/myu


Dx = 2.5
dx = dy = 0.001*Dx
dz = dx
dVol = 1 

case = 5

E_array = [0,38,49,53,66,71]
E = E_array[case]
e = E*dx

nbr_a = np.load('case'+str(case)+'/artery_nbrhd_'+str(E)+'.npy', allow_pickle=True)
nbr_v = np.load('case'+str(case)+'/vein_nbrhd_'+str(E)+'.npy', allow_pickle=True)


Ca = np.load('case'+str(case)+'/Ca_' + str(E) +'.npy')
Cv = np.load('case'+str(case)+'/Cv_' + str(E) +'.npy')


pa = np.load('case'+str(case)+'/flow_solutions/' + str(E) + '/prs_acomp_' + str(E) +'_Pin_1000_Pout_1.npy')
pv = np.load('case'+str(case)+'/flow_solutions/' + str(E) + '/prs_vcomp_' + str(E) +'_Pin_1000_Pout_1.npy')
dom = np.load('case'+str(case)+'/2.5_dom.npy')
cdom = np.load('case'+str(case)+'/2.5_cdom.npy')
qa = np.load('case'+str(case)+'/flow_solutions/' + str(E) +'/Q_arteries_' + str(E) +'_Pin_1000_Pout_1.npy')
qv = np.load('case'+str(case)+'/flow_solutions/' + str(E) +'/Q_veins_' + str(E) +'_Pin_1000_Pout_1.npy')

a_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_db.csv')
v_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_db.csv')
a_out = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_out_pts.csv').to_numpy()
v_out = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_out_pts.csv').to_numpy()





equal = False 



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

def nbr_tissue(nbr,cell, pa, pv):
    ta = 2*dy*dz*(1/(dx/Lambda_a + dx/Lambda_a))
    tv = 2*dy*dz*(1/(dx/Lambda_v + dx/Lambda_v))
    
    ma = (pa[nbr[0],nbr[1],nbr[2]] - pa[cell[0],cell[1],cell[2]])*ta
    mv = (pv[nbr[0],nbr[1],nbr[2]] - pv[cell[0],cell[1],cell[2]])*tv
    # m_av = a*(pa[cell[0],cell[1],cell[2]] - pv[cell[0],cell[1],cell[2]])
    
    return (ma, mv)#, m_av)

def nbr_other(nbr, cell, pa, pv):
    return (0, 0)


tissue_index = 0
air_index = -1
artery_index = 1
vein_index = 2


func_db = {tissue_index: (lambda nbr,cell, pa, pv: nbr_tissue(nbr,cell,pa, pv)), 
           air_index: (lambda nbr,cell, pa, pv: nbr_other(nbr,cell,pa, pv)),
           artery_index: (lambda nbr,cell, pa, pv: nbr_other(nbr,cell,pa, pv)),
           vein_index: (lambda nbr,cell, pa, pv: nbr_other(nbr,cell,pa, pv))}



nx, ny, nz = np.shape(dom)

ma = np.zeros((nx,ny,nz), dtype = float)
mv = np.zeros((nx,ny,nz), dtype = float)
mav = np.zeros((nx, ny, nz), dtype = float)

n_v = np.max(cdom) + 1
n_a = len(qa) + 1

# def tissue_mass(nz1, nz2):
start = time.time()
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
                if(dom[i,j,k] == tissue_index):
                    
                    
                    ma_N, mv_N = func_db[dom[i+1,j,k]]([i+1,j,k],[i,j,k],pa,pv)
                    ma_S, mv_S = func_db[dom[i-1,j,k]]([i-1,j,k],[i,j,k],pa,pv)
                    ma_E, mv_E = func_db[dom[i,j+1,k]]([i,j+1,k],[i,j,k],pa,pv)
                    ma_W, mv_W = func_db[dom[i,j-1,k]]([i,j-1,k],[i,j,k],pa,pv)
                    ma_F, mv_F = func_db[dom[i,j,k+1]]([i,j,k+1],[i,j,k],pa,pv)
                    ma_B, mv_B = func_db[dom[i,j,k-1]]([i,j,k-1],[i,j,k],pa,pv)
                    
                    
                    ma[i,j,k] = ma_N + ma_S + ma_E + ma_W + ma_F + ma_B
                    mv[i,j,k] = mv_N + mv_S + mv_E + mv_W + mv_F + mv_B
                    
                    mav[i,j,k] = a*(pa[i,j,k] - pv[i,j,k])
        
stop = time.time()
print(round(stop - start)/60, 'mins')

np.save('case_'+str(case)+'_E_' + str(E) + '_art_comp_mass.npy',ma)
np.save('case_'+str(case)+'_E_' + str(E) + '_ven_comp_mass.npy',mv)
np.save('case_'+str(case)+'_E_' + str(E) + '_art_ven_comp_mass.npy',mav)


# # # RUN THE FOLLOWING SECTION ON BEOCAT
'''

vessel_mass_matrix = np.zeros((nx,ny,nz), dtype = float)

start = time.time()
for k in range(nz):
    # print(k)
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
                        Q = abs(qa[int(ele.iloc[0,0])])
                        Q_dot = Q*n_ex
                        m_dot = m_dot + Q_dot
                        
                vessel_mass_matrix[i,j,k] = m_dot

stop = time.time()
print(round((stop - start)/60,3), 'mins')


np.save('case_'+str(case)+'_E_' + str(E) + '_vessel_supply_mass.npy',vessel_mass_matrix)

'''