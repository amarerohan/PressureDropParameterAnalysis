import numpy as np
import pandas as pd

case = 5
Layer = 1

dom = np.load('case' + str(case) + '/2.5_dom.npy')
a_ele = np.load('case' + str(case) + '/a_ele.npy')
v_ele = np.load('case' + str(case) + '/v_ele.npy')

a_db = pd.read_csv('case' + str(case) + '/2.5_a_db.csv')
v_db = pd.read_csv('case' + str(case) + '/2.5_v_db.csv')

a_pt = pd.read_csv('case' + str(case) + '/2.5_a_pts.csv')
v_pt = pd.read_csv('case' + str(case) + '/2.5_v_pts.csv')

a_out = pd.read_csv('case' + str(case) + '/2.5_a_out_pts.csv')
v_out = pd.read_csv('case' + str(case) + '/2.5_v_out_pts.csv')


atag = []
vtag = []

nx, ny, nz = np.shape(a_ele)

for i in range(len(a_out)):
    x, y, z = a_out.iloc[i,1:4]
    atag.append(dom[int(x),int(y),int(z)])

for i in range(len(v_out)):
    x, y, z = v_out.iloc[i,1:4]
    vtag.append(dom[int(x),int(y),int(z)])

artery_tag = 1
vein_tag = 2
tissue_tag = 0

for i in range(len(atag)):
    if atag[i] == tissue_tag:
        x = int(a_out.iloc[i,1])
        y = int(a_out.iloc[i,2])
        z = int(a_out.iloc[i,3])
        
        out_db = np.array([[x,y,z], 
                        [x+1,y,z], [x-1,y,z], [x,y+1,z], 
                        [x,y-1,z],[x,y,z+1], [x,y,z-1], 
                        [x+1,y+1,z], [x+1,y+1,z+1], [x+1,y+1,z-1], 
                        [x-1,y-1,z], [x-1,y-1,z+1], [x-1,y-1,z-1], 
                        [x+1, y-1, z], [x+1, y-1, z+1], [x+1,y-1,z-1], 
                        [x-1,y+1,z], [x-1,y+1,z-1], [x-1,y+1,z+1]                       
                        ],dtype = int)
        
        for j in range(len(out_db)):
            count = 0
            x,y,z = out_db[j,:]
            if(dom[x,y,z]) == artery_tag:
                a_out.iloc[i,1:4] = [x,y,z]
                break

for i in range(len(vtag)):
    if vtag[i] == tissue_tag:
        x = int(v_out.iloc[i,1])
        y = int(v_out.iloc[i,2])
        z = int(v_out.iloc[i,3])
        
        out_db = np.array([[x,y,z], 
                        [x+1,y,z], [x-1,y,z], [x,y+1,z], 
                        [x,y-1,z],[x,y,z+1], [x,y,z-1], 
                        [x+1,y+1,z], [x+1,y+1,z+1], [x+1,y+1,z-1], 
                        [x-1,y-1,z], [x-1,y-1,z+1], [x-1,y-1,z-1], 
                        [x+1, y-1, z], [x+1, y-1, z+1], [x+1,y-1,z-1], 
                        [x-1,y+1,z], [x-1,y+1,z-1], [x-1,y+1,z+1]                       
                        ],dtype = int)
        
        for j in range(len(out_db)):
            count = 0
            x,y,z = out_db[j,:]
            if(dom[x,y,z]) == vein_tag:
                v_out.iloc[i,1:4] = [x,y,z]
                break
        
atag2 = []
vtag2 = []

for i in range(len(a_out)):
    x, y, z = a_out.iloc[i,1:4]
    atag2.append(dom[int(x),int(y),int(z)])

for i in range(len(v_out)):
    x, y, z = v_out.iloc[i,1:4]
    vtag2.append(dom[int(x),int(y),int(z)])        
        

a_out.to_csv('case'+str(case)+'/2.5_a_out_pts.csv', index=False)
v_out.to_csv('case'+str(case)+'/2.5_v_out_pts.csv', index=False)