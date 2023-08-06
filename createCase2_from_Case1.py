import numpy as np
import pandas as pd

new_case = 5
Layer = 1

dom = np.load('case1/2.5_dom.npy')
a_ele = np.load('case1/a_ele.npy')
v_ele = np.load('case1/v_ele.npy')

a_db = pd.read_csv('case1/2.5_a_db.csv')
v_db = pd.read_csv('case1/2.5_v_db.csv')

a_ele_2 = np.copy(a_ele)
v_ele_2 = np.copy(v_ele)

dom_2 = np.copy(dom)
nx, ny, nz = np.shape(dom_2)

a_pt = pd.read_csv('case1/2.5_a_pts.csv')
v_pt = pd.read_csv('case1/2.5_v_pts.csv')



min_outlet_index = 2**Layer
max_outlet_index = min_outlet_index + 2**Layer

min_branch_ele = max_outlet_index - 2
max_branch_ele = 62 # min_branch_ele + 2**(Layer+1)

for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(a_ele_2[i,j,k] > min_branch_ele and a_ele_2[i,j,k] <= max_branch_ele):
                a_ele_2[i,j,k] = -1
                dom_2[i,j,k] = 0
            if(v_ele_2[i,j,k] > min_branch_ele and v_ele_2[i,j,k] <= max_branch_ele):
                v_ele_2[i,j,k] = -1
                dom_2[i,j,k] = 0

c_dom = np.zeros((nx,ny,nz), dtype = int)

count = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom_2[i,j,k] == 0):
                c_dom[i,j,k] = count
                count = count + 1


a_pt_2 = a_pt.iloc[:max_outlet_index,:]
v_pt_2 = v_pt.iloc[:max_outlet_index,:]

a_out = a_pt_2.iloc[min_outlet_index:,:]
v_out = v_pt_2.iloc[min_outlet_index:,:]

a_db_2 = a_db.iloc[:min_branch_ele+1,:]
v_db_2 = v_db.iloc[:min_branch_ele+1,:]



max_branch_ele = min_branch_ele + 2**(Layer+1)

un_t = np.max(c_dom) + 1
print('Layer = ', Layer)
print('un_t = ', un_t)
print('Branch Element Removed = ', min_branch_ele+1, max_branch_ele)
print('Range of Outlets = ', min_outlet_index, max_outlet_index-1)
                


np.save('case'+str(new_case)+'/a_ele.npy',a_ele_2)
np.save('case'+str(new_case)+'/v_ele.npy',v_ele_2)

np.save('case'+str(new_case)+'/2.5_dom.npy', dom_2)
np.save('case'+str(new_case)+'/2.5_cdom.npy',c_dom)

a_pt_2.to_csv('case'+str(new_case)+'/2.5_a_pts.csv', index=False)
v_pt_2.to_csv('case'+str(new_case)+'/2.5_v_pts.csv', index=False)
a_out.to_csv('case'+str(new_case)+'/2.5_a_out_pts.csv', index=False)
v_out.to_csv('case'+str(new_case)+'/2.5_v_out_pts.csv', index=False)
a_db_2.to_csv('case'+str(new_case)+'/2.5_a_db.csv', index=False)
v_db_2.to_csv('case'+str(new_case)+'/2.5_v_db.csv', index=False)

