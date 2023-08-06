import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(15)
# font.set_style('italic')
f = 20

Xa1 = np.load('case1/X_28_a_0.001_G_0.1_spsolve.npy')
Xa2 = np.load('case1/X_28_a_0.0001_G_0.1_spsolve.npy')
Xb1 = np.load('case2/X_25_a_0.001_G_0.1_spsolve.npy')
Xb2 = np.load('case2/X_25_a_0.0001_G_0.1_spsolve.npy')

KA = np.load('KA.npy')
KV = np.load('KV.npy')

Pa_a1 = Xa1[49356:49364]
Pv_a1 = Xa1[49364:]

Pa_a2 = Xa2[49356:49364]
Pv_a2 = Xa2[49364:]

Pa_b1 = Xb1[49552:49555]
Pv_b1 = Xb1[49555:]

Pa_b2 = Xb2[49552:49555]
Pv_b2 = Xb2[49555:]


Qa_a1 = KA[0]*(Pa_a1[0] - Pa_a1[1])
Qa_a2 = KA[0]*(Pa_a2[0] - Pa_a2[1])

Qa_b1 = KA[0]*(Pa_b1[0] - Pa_b1[1])
Qa_b2 = KA[0]*(Pa_b2[0] - Pa_b2[1])

Q = np.load('case2/Q_arteries_25_a_0.001_G_3.32e-09_spsolve.npy')
V = np.load('case2/X_25_a_0.001_G_3.32e-09_spsolve.npy')
V = np.load('Heat_X_25_a_0.001_q_0_Tin_40_Tamb_0_hamb_10000_2.npy')
X = np.load('Heat_X_28_a_0.001_q_0_Tin_40_Tamb_0_.npy')
Y = np.load('Heat_X_25_a_0.001_q_0_Tin_40_Tamb_0_.npy')
Z = np.load('Heat_X_25_a_0.001_q_0_Tin_40_Tamb_0_hamb_10000_.npy')
W = np.load('Heat_X_28_a_0.001_q_0_Tin_40_Tamb_0_hamb_10000_.npy')

cdom = np.load('case1/2_cdom.npy')
nx,ny,nz = np.shape(cdom)
Tx = np.zeros((nx,ny,nz), dtype = float)
Tw = np.zeros((nx,ny,nz), dtype = float)
count = 0
dom = np.load('case1/2_dom.npy')
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 0):
                Tx[i,j,k] = X[cdom[i,j,k]]
                Tw[i,j,k] = W[cdom[i,j,k]]
                count = count + 1
                
cdom = np.load('case2/2_cdom.npy')
dom = np.load('case2/2_dom.npy')
Ty = np.zeros((nx,ny,nz), dtype = float)
Tz = np.zeros((nx,ny,nz), dtype = float)
Tv = np.zeros((nx,ny,nz), dtype = float)
count = 0
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 0):
                Ty[i,j,k] = Y[cdom[i,j,k]]
                Tz[i,j,k] = Z[cdom[i,j,k]]
                Tv[i,j,k] = V[cdom[i,j,k]]
                count = count + 1
                

for z in range(1,nz):                
    colors = 'Reds'
                 
    plt.figure(figsize=(12,10))
    plt.contourf(Tw[1:-1,1:-1,z] - Tv[1:-1,1:-1,z],np.arange(0,30,3), cmap=colors)
    plt.colorbar()
    plt.title('z = '+str(z))
    plt.show()

# plt.figure(figsize=(12,10))
# plt.contourf(Tw[:,:,20],np.arange(30,41,0.5))
# plt.colorbar()
# plt.show()
