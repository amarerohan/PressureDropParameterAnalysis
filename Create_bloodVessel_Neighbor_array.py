import numpy as np
import pandas as pd

case = 5
Dx = 2.5

dom = np.load('case'+str(case)+'/'+str(Dx) + '_dom.npy')
a_ele = np.load('case'+str(case)+'/a_ele.npy')
v_ele = np.load('case'+str(case)+'/v_ele.npy')

a_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_db.csv')
v_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_db.csv')

a_nbr = []
v_nbr = []

a_ele[0,:,:] = a_ele[-1,:,:] = -10
a_ele[:,0,:] = a_ele[:,-1,:] = -10
a_ele[:,:,0] = a_ele[:,:,-1] = -10

v_ele[0,:,:] = v_ele[-1,:,:] = -10
v_ele[:,0,:] = v_ele[:,-1,:] = -10
v_ele[:,:,0] = v_ele[:,:,-1] = -10


for i in range(len(a_db)):
    a_nbr.append([])
for i in range(len(v_db)):
    v_nbr.append([])

nx, ny, nz = np.shape(dom)

for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(a_ele[i,j,k] != -1 and a_ele[i,j,k] != -10):
                if(a_ele[i+1,j,k] == -1):
                    a_nbr[a_ele[i,j,k]].append([i+1,j,k])
                if(a_ele[i-1,j,k] == -1):
                    a_nbr[a_ele[i,j,k]].append([i-1,j,k])
                if(a_ele[i,j+1,k] == -1):
                    a_nbr[a_ele[i,j,k]].append([i,j+1,k])
                if(a_ele[i,j-1,k] == -1):
                    a_nbr[a_ele[i,j,k]].append([i,j-1,k])
                if(a_ele[i,j,k+1] == -1):
                    a_nbr[a_ele[i,j,k]].append([i,j,k+1])
                if(a_ele[i,j,k-1] == -1):
                    a_nbr[a_ele[i,j,k]].append([i,j,k-1])
                    
            if(v_ele[i,j,k] != -1 and v_ele[i,j,k] != -10):
                if(v_ele[i+1,j,k] == -1):
                    v_nbr[v_ele[i,j,k]].append([i+1,j,k])
                if(v_ele[i-1,j,k] == -1):
                    v_nbr[v_ele[i,j,k]].append([i-1,j,k])
                if(v_ele[i,j+1,k] == -1):
                    v_nbr[v_ele[i,j,k]].append([i,j+1,k])
                if(v_ele[i,j-1,k] == -1):
                    v_nbr[v_ele[i,j,k]].append([i,j-1,k])
                if(v_ele[i,j,k+1] == -1):
                    v_nbr[v_ele[i,j,k]].append([i,j,k+1])
                if(v_ele[i,j,k-1] == -1):
                    v_nbr[v_ele[i,j,k]].append([i,j,k-1])

np.save('case'+str(case)+'/a_nbr.npy',a_nbr, allow_pickle=True)
np.save('case'+str(case)+'/v_nbr.npy',v_nbr, allow_pickle=True)


# A = np.load('case'+str(case)+'/a_nbr.npy', allow_pickle=True)
# dom = np.load('case1/2_dom.npy')
# a_ele = np.load('case1/a_ele.npy')
# v_ele = np.load('case1/v_ele.npy')

# dom[:,:,41] = -1
# dom = dom[:,:,:42]
# a_ele[:,:,41] = -1
# v_ele[:,:,41] = -1
# a_ele = a_ele[:,:,:42]
# v_ele = v_ele[:,:,:42]

# nx, ny, nz = np.shape(dom)

# c_dom = np.zeros((nx,ny,nz), dtype = int)

# for k in range(nz):
#     for j in range(ny):
#         for i in range(nx):
#             if(a_ele[i,j,k] != -1):
#                 dom[i,j,k] = 1
#             if(v_ele[i,j,k] != -1):
#                 dom[i,j,k] = 2

# count = 0
# for k in range(nz):
#     for j in range(ny):
#         for i in range(nx):
#             if(dom[i,j,k] == 0):
#                 c_dom[i,j,k] = count
#                 count = count + 1

# np.save('case1/a_ele.npy',a_ele)
# np.save('case1/v_ele.npy',v_ele)
# np.save('case1/2_dom.npy',dom)
# np.save('case1/2_cdom.npy',c_dom)