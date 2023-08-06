# # # CALCULATE C VALUE

import numpy as np
import Neighbourhood_matrix as nbr
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import os 
import pandas as pd

equal = False

# dx = nbr.dx
# dy = nbr.dy
# dz = nbr.dz
# Ng = 6
Layer  = 1

case = 5
DX = 2.5
dx = dy = dz = 0.001*DX

E = 105
e = E*dx

arterial_db = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_db.csv').to_numpy()
venous_db = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_db.csv').to_numpy()
# artery_pts = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_pt.csv').to_numpy()
# venous_pts = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_pt.csv').to_numpy()
# a_term = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_term.csv').to_numpy()
# v_term = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_term.csv').to_numpy()
a_outlets = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_out_pts.csv').to_numpy()
v_outlets = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_out_pts.csv').to_numpy()


dom = np.load('case'+str(case)+'/'+str(DX)+'_dom.npy')

nbr_a = np.load('case'+str(case)+'/artery_nbrhd_'+str(E)+'.npy', allow_pickle=True)
nbr_v = np.load('case'+str(case)+'/vein_nbrhd_'+str(E)+'.npy', allow_pickle=True)



ny, nx, nz = np.shape(dom)

CA = 0
CV = 0

def SoI(center, point):
    x0, y0, z0 = center
    x1, y1, z1 = point
    
    # s = np.sqrt( ((x0-x1*dx))**2 + ((y0-y1*dy))**2 + ((z0-z1*dz))**2)
    s = np.sqrt( ((x0-x1)*dx)**2 + ((y0-y1)*dy)**2 + ((z0-z1)*dz)**2)
    
    return(s)

def artery():
    Ca = np.empty(len(a_outlets),dtype=float)
    
    for i in range(len(a_outlets)):
        sum_exp = 0.0
        x0, y0, z0 = a_outlets[i,1:4]
        for j in range(len(nbr_a[i])):
            x1,y1,z1 = nbr_a[i][j]
            s = SoI([x0,y0,z0], [x1,y1,z1])
            # print(i,j,[x0,y0], [x1,y1],s)
            if(s<e):
                if equal == True:
                    exp = 1/len(nbr_a[i])
                else:
                    exp = np.exp(1/(abs(s/e)**2 - 1))
                sum_exp = sum_exp + exp#/(e**3)
    
    
        Ca[i] = 1/sum_exp # ; print(1/sum_exp, sum_exp)
    # print('CA', Ca)
    np.save('case'+str(case)+'/Ca_' + str(E) +'.npy', Ca)

def vein():
    Cv = np.empty(len(v_outlets),dtype=float)
    
    for i in range(len(v_outlets)):
        sum_exp = 0.0
        x0, y0, z0 = v_outlets[i,1:4]
        for j in range(len(nbr_v[i])):
            x1,y1,z1 = nbr_v[i][j]
            s = SoI([x0,y0,z0], [x1,y1,z1])
            # print(i,j,[x0,y0], [x1,y1],s)
            if(s<e):
                if equal == True:
                    exp = 1/len(nbr_v[i])
                else:
                    exp = np.exp(1/(abs(s/e)**2 - 1))
                sum_exp = sum_exp + exp#/(e**3)
    
        Cv[i] = 1/sum_exp
    # print('CV', Cv)
    np.save('case'+str(case)+'/Cv_' + str(E) +'.npy', Cv)

    
def main():
    with ProcessPoolExecutor(max_workers=1) as executor_p:
        executor_p.submit(artery)
        executor_p.submit(vein)
        
if __name__ == '__main__':
    main()

