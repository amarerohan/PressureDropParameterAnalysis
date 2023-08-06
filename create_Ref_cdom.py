import numpy as np

case = 5

cdom = np.load('case' +str(case) + '/2.5_cdom.npy')
a_ele = np.load('case'+str(case) + '/a_ele.npy')
v_ele = np.load('case'+str(case) + '/v_ele.npy')

un_t = np.max(cdom) + 1
un_a = np.max(a_ele) + 1

new_cdom = np.copy(cdom)

nx, ny, nz = np.shape(cdom)

for z in range(nz):
    for y in range(ny):
        for x in range(nx):
            if a_ele[x,y,z] != -1:
                new_cdom[x,y,z] = un_t + a_ele[x,y,z]
            if v_ele[x,y,z] != -1:
                new_cdom[x,y,z] = un_t + un_a + v_ele[x,y,z]

np.save('case'+str(case)+'/ref_cdom.npy', new_cdom)