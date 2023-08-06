import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor, ceil

DX = 2.5
# case = 1

dx = dy = dz = 0.001*DX

# E = 10 #Number of voxels
# e = E*dx  




# arterial_db = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_db.csv').to_numpy()
# venous_db = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_db.csv').to_numpy()
# a_outlets = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_out_pts.csv').to_numpy()
# v_outlets = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_out_pts.csv').to_numpy()


# dom = np.load('case'+str(case)+'/'+str(DX)+'_dom.npy')
# c_dom = np.load('case'+str(case)+'/'+str(DX)+'_cdom.npy')
# ref_volume = (np.max(c_dom)+1)*dx*dy*dz







def SoI(center, point):
    x0, y0, z0 = center
    x1, y1, z1 = point
    # print(center, point)
    s = np.sqrt( ((x0-x1)*dx)**2 + ((y0-y1)*dy)**2 + ((z0-z1)*dz)**2)
    
    return(s)



def main():
    case_list = [1,2,3,4,5]
    E_list = np.arange(5,75,5)
    
    ans = []
    
    for case in case_list:
        arterial_db = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_db.csv').to_numpy()
        venous_db = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_db.csv').to_numpy()
        a_outlets = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_out_pts.csv').to_numpy()
        v_outlets = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_out_pts.csv').to_numpy()


        dom = np.load('case'+str(case)+'/'+str(DX)+'_dom.npy')
        c_dom = np.load('case'+str(case)+'/'+str(DX)+'_cdom.npy')
        
        ny, nx, nz = np.shape(dom)
        
        ref_volume = (np.max(c_dom)+1)*dx*dy*dz
        
        E = 5
        
        while E <= 70:
            e = E*dx  
    
            NBRHD_A = np.empty(len(a_outlets),dtype = 'object')
            NBRHD_V = np.empty(len(v_outlets),dtype = 'object')
            
            for k in range(len(a_outlets)):
                x,y,z = a_outlets[k,1:4]
                X_max = ceil(min(x + E, nx))
                Y_max = ceil(min(y + E, ny))
                Z_max = ceil(min(z + E, nz))
                X_min = floor(max(x - E, 0))
                Y_min = floor(max(y - E, 0))
                Z_min = floor(max(z - E, 0))
                
                nbr = []
                
                for h in range(Z_min, Z_max):
                    for i in range(X_min, X_max):
                        for j in range(Y_min, Y_max):
                            
                            if(dom[i,j,h] == 0):
                                s = SoI([x,y,z], [i,j,h])  
                                if(s<e):
                                    nbr.append([i,j,h])
                
                NBRHD_A[k] = nbr
                
        
            for k in range(len(v_outlets)):
                x,y,z = v_outlets[k,1:4]
                X_max = ceil(min(x + E, nx))
                Y_max = ceil(min(y + E, ny))
                Z_max = ceil(min(z + E, nz))
                X_min = floor(max(x - E, 0))
                Y_min = floor(max(y - E, 0))
                Z_min = floor(max(z - E, 0))
                
                nbr = []
                
                for h in range(Z_min, Z_max):
                    for i in range(X_min, X_max):
                        for j in range(Y_min, Y_max):
                            # print(i,j,h)
                            if(dom[i,j,h] == 0):
                                s = SoI([x,y,z], [i,j,h]) 
                                if(s<e):
                                    nbr.append([i,j,h])
                
                NBRHD_V[k] = nbr
            
            
            # np.save('case'+str(case)+'/artery_nbrhd_'+str(E)+'.npy',NBRHD_A)
            # np.save('case'+str(case)+'/vein_nbrhd_'+str(E)+'.npy',NBRHD_V)
            
            
            
            dom_a = np.copy(dom)
            dom_v = np.copy(dom)
            
            for i in range(len(NBRHD_A)):
                for j in range(len(NBRHD_A[i])):
                    x,y,z = NBRHD_A[i][j]
                    dom_a[x,y,z] = dom_a[x,y,z] + 20
        
            for i in range(len(NBRHD_V)):
                for j in range(len(NBRHD_V[i])):
                    x,y,z = NBRHD_V[i][j]
                    dom_v[x,y,z] = dom_v[x,y,z] + 20
                    
            count_a = 0
            count_v = 0
            for h in range(nz):
                for i in range(nx):
                    for j in range(ny):
                        if(dom[i,j,h] == 0):
                            if(dom_a[i,j,h] != 0):
                                count_a = count_a + 1
                            if(dom_v[i,j,h] != 0):
                                count_v = count_v + 1
        
            vol_a = count_a*dx*dy*dz/(ref_volume)*100  # PERCENTAGE NOT COVERED BY DIRECT FLOW
            vol_v = count_v*dx*dy*dz/(ref_volume)*100 # PERCENTAFGE NOT COVERED BY DIRECT FLOW
            
            
            print('case = ', case, ' ', 'E = ',E)
        
            print('a volume covered', round(vol_a,2), '%')
            print('v volume covered', round(vol_v,2), '%')
            
            
            ans.append([case,E,vol_a,vol_v])
            
            if (vol_a < 99 or vol_v <99):
                E = E + 5
            elif(vol_a >= 99 or vol_v >= 99):
                E = E + 1
            
            if(vol_a == 100 and vol_v == 100):
                break
    
    np.save('volume_coverage.npy',ans)
    
    return(ans)


ans = main()
    
