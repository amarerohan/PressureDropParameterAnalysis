import numpy as np

import matplotlib.pyplot as plt

import scipy.sparse as sp
to

import time 

start = time.time()


row = np.load('case1/heat_solutions/38/Heat_row_38_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
col = np.load('case1/heat_solutions/38/Heat_col_38_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
data = np.load('case1/heat_solutions/38/Heat_data_38_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
X = np.load('case1/heat_solutions/38/Heat_X_38_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
n_row = np.max(row) + 1

c_dom = np.load('case1/2.5_cdom.npy')
dom = np.load('case1/2.5_dom.npy')
un_t = np.max(c_dom) + 1

A = sp.csr_matrix((data,(row,col)), shape = (n_row, n_row))


Z = sp.csr_matrix(A.multiply(X))

indices = Z.indices
indptr = Z.indptr
data = Z.data

nbr = []
ves = []
for r in range(un_t):
    index_1 = indptr[r]
    index_2 = indptr[r+1]
    
    row_data = data[index_1:index_2]
    col_index = indices[index_1:index_2]
    
    nbr.append(np.sum(row_data[np.where((col_index != r) & (col_index < un_t))]))
    ves.append(np.sum(row_data[np.where(col_index >= un_t)]))
    

            
stop = time.time()

print(round((stop - start)/60,3)) 
    





start = time.time()


row = np.load('case4/heat_solutions/66/Heat_row_66_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
col = np.load('case4/heat_solutions/66/Heat_col_66_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
data = np.load('case4/heat_solutions/66/Heat_data_66_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
X2 = np.load('case4/heat_solutions/66/Heat_X_66_a_0.001_q_1000_Tin_35_Tamb_20_hamb_10_hbt_10_.npy')
n_row = np.max(row) + 1

c_dom_2 = np.load('case4/2.5_cdom.npy')
dom2 = np.load('case4/2.5_dom.npy')
un_t2 = np.max(c_dom_2) + 1

A2 = sp.csr_matrix((data,(row,col)), shape = (n_row, n_row))


Z2 = sp.csr_matrix(A2.multiply(X2))

indices = Z2.indices
indptr = Z2.indptr
data = Z2.data

nbr2 = []
ves2 = []


for r in range(un_t2):
    index_1 = indptr[r]
    index_2 = indptr[r+1]
    
    row_data = data[index_1:index_2]
    col_index = indices[index_1:index_2]
    
    nbr2.append(np.sum(row_data[np.where((col_index != r) & (col_index < un_t))]))
    ves2.append(np.sum(row_data[np.where(col_index >= un_t)]))
    

            
stop = time.time()

print(round((stop - start)/60,3)) 



nbr_error = []
ves_error = []

nx, ny, nz = np.shape(c_dom)

N = np.zeros((nx,ny,nz), dtype = float)
V = np.zeros((nx,ny,nz), dtype = float)


for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 0):
                nbr_error.append((nbr[c_dom[i,j,k]] - nbr2[c_dom_2[i,j,k]])/nbr[c_dom[i,j,k]]*100)
                ves_error.append((ves[c_dom[i,j,k]] - ves2[c_dom_2[i,j,k]])/ves[c_dom[i,j,k]]*100)
                
                N[i,j,k] = (nbr[c_dom[i,j,k]] - nbr2[c_dom_2[i,j,k]])/nbr[c_dom[i,j,k]]*100
                V[i,j,k] = (ves[c_dom[i,j,k]] - ves2[c_dom_2[i,j,k]])/ves[c_dom[i,j,k]]*100



for z in range(1,80):

    plt.figure(figsize = (12,10))
    plt.contourf(N[1:-1,1:-1,z], np.arange(-130,135,5), cmap = 'RdBu')
    plt.colorbar()
    plt.show()
    
    
for z in range(1,80):

    plt.figure(figsize = (12,10))
    plt.contourf(V[1:-1,1:-1,z], np.arange(-100,110,10), cmap = 'RdBu')
    plt.colorbar()
    plt.show()

