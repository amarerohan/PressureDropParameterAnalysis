import numpy as np

dom = np.load('case1/2.5_dom.npy')
a_ele = np.load('case1/a_ele.npy')
v_ele = np.load('case1/v_ele.npy')

nx, ny, nz = np.shape(dom)

a = []
v = []

for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 1 and a_ele[i,j,k] == -1):
                a.append([i,j,k])
                dom[i,j,k] = 0
            if(dom[i,j,k] == 2 and v_ele[i,j,k] == -1):
                v.append([i,j,k])
                dom[i,j,k] = 0

c_dom = np.zeros((nx,ny,nz), dtype = int)

count = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 0):
                c_dom[i,j,k] = count
                count = count + 1

np.save('case1/2.5_dom.npy',dom)
np.save('case1/2.5_cdom.npy',c_dom)

