import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, Manager
import scipy.sparse as sp
import scipy.sparse.linalg as spla

spsolve = True
mat_iterative = False
sor = False

case = 1
E = 28
Dx = 2

a_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_db.csv')
v_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_db.csv')
# a_pt = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_pt.csv').to_numpy()
# v_pt = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_pt.csv').to_numpy()
dom = np.load('case'+str(case)+'/'+str(Dx)+'_dom.npy')
c_dom = np.load('case'+str(case)+'/'+str(Dx)+'_cdom.npy')

r = np.load('case'+str(case)+'/'+str(E)+'_row.npy')
c = np.load('case'+str(case)+'/'+str(E)+'_col.npy')
d = np.load('case'+str(case)+'/'+str(E)+'_data.npy')

un_a = len(a_db) + 1 #len(a_pt)
un_v = len(v_db) + 1 #len(v_pt)
un_t = np.max(c_dom) + 1

N = np.max(r)+1

Pin = 1000 
Pout = 1

b = np.zeros((N,1), dtype = int)
b[2*un_t,0] = Pin
b[2*un_t+un_a,0] = Pout

x = np.zeros((N,1), dtype = float)
x[:] = 1.0
# x[0:un_t] = 550
# x[un_t:2*un_t] = 445

if spsolve == True:
    start = time.time()
    A = sp.csc_matrix((d,(r,c)), shape = (N,N))
    X = spla.spsolve(A,b)
    method = 'spsolve'
    stop = time.time()
    print('spsolve ', round((stop - start)/60,3), ' mins')

if sor == True:
    W = 1.85
    residue = 1
    conv = 1e-5
    res = np.ones((N,1), dtype = float)
    
    A = sp.csc_matrix((d,(r,c)), shape = (N,N))
    start_conv = time.time()
    iteration = 0
    while residue > conv:
        iter_start = time.time()
        for i in range(N):
            # loc = np.where(r == i)[0].tolist()
            r_sum = 0
            for j in np.where(r == i)[0].tolist():
                if int(c[j]) != i:
                    r_sum = r_sum + d[j]*x[int(c[j])]
                elif int(c[j]) == i:
                    a_ii = d[j]
            x[i] = (1-W)*x[i] + W*((b[i]-r_sum)/a_ii)
        iter_stop = time.time()
        
        # res_start = time.time()
        # CALCULATE RESIDUE
        # for i in range(N):
        #     # loc = np.where(r == i)[0].tolist()
        #     r_sum = 0
        #     for j in np.where(r == i)[0].tolist():
        #         r_sum = r_sum + d[j]*x[int(c[j])]
        #     res[i] = b[i] - r_sum
        
        # residue = np.linalg.norm(res)
        
        res = A.multiply(x[:,0])
        res = res.sum(axis = 1) - b
        residue = np.linalg.norm(res)
        
        # res_stop = time.time()
        # print('Residue time  = ', round((res_stop - res_start)/60,3))
        print(iteration, round(residue,5), round((iter_stop-iter_start)/60,3), ' mins')
        iteration = iteration + 1
    stop_conv = time.time()
    print('Time taken to achieve convergence = ', round((stop_conv - start_conv)/60,3), ' mins')
    X = x
    method = 'sor'

if mat_iterative == True:
    start = time.time()
    A = sp.csc_matrix((d,(r,c)), shape = (N,N))
    LU = spla.spilu(A)
    M = spla.LinearOperator(np.shape(LU), LU.solve)
    # x0 = x
    # x0[:] = 1 #(Pin+Pout)/2.0
    X = spla.lgmres(A,b, M=M, x0=x, tol=1e-15, atol=1e-15, maxiter=2000)#,tol=1e-15, atol = 1e-15)[0]
    reason = X[1]
    X = X[0]
    method = 'lgmres'
    stop = time.time()
    print('lgmres ', round((stop-start)/60,3), ' mins')


# # # # RESIDUAL AND ERROR MEASUREMENT # # # 

res = A.multiply(X)
res = res.sum(axis = 1) - b
residual = np.linalg.norm(res)
# residual = np.matmul(A.todense(),X).transpose() - b
print('residual = ', round(np.linalg.norm(residual),5))

Prs_a_comp = X[:un_t]
Prs_v_comp = X[un_t:2*un_t]

PA = X[2*un_t:2*un_t+un_a]
PV = X[2*un_t + un_a:]

QA = []
QV = []
KA = []
KV = []
myu = 1e-3
for i in range(len(a_db)):
    r = a_db.iloc[i,4]
    L = a_db.iloc[i,3]
    k = np.pi*(r)**4/(8*myu*L)
    KA.append(k)
    n1 = a_db.iloc[i,1]
    n2 = a_db.iloc[i,2]
    q = (PA[int(n1)] - PA[int(n2)])*k
    QA.append(q)

for i in range(len(v_db)):
    r = v_db.iloc[i,4]
    L = v_db.iloc[i,3]
    k = np.pi*(r)**4/(8*myu*L)
    KV.append(k)
    n1 = v_db.iloc[i,1]
    n2 = v_db.iloc[i,2]
    q = (PV[int(n1)] - PV[int(n2)])*k
    QV.append(q)


print('Flow Error = ', round(QA[0] + QV[0],5))


nx,ny,nz = np.shape(dom)

pa = np.zeros((nx,ny,nz), dtype = float)
pv = np.zeros((nx,ny,nz), dtype = float)

# pa[1:-1,1:-1,1:-1] = 0
# pv[1:-1,1:-1,1:-1] = 0

pa[:,:,:] = 0.0
pv[:,:,:] = 0.0


for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 0):
                h = c_dom[i,j,k]
                pa[i,j,k] = Prs_a_comp[h]
                pv[i,j,k] = Prs_v_comp[h]


'''
# # # SAVE SOLUTIONS # # # # 
solution = 'case'+str(case)+'/X_' + str(E) + method +'.npy'
prs_acomp = 'case'+str(case)+'/prs_acomp_' + str(E) +  method +'.npy'
prs_vcomp = 'case'+str(case)+'/prs_vcomp_' + str(E) +  method +'.npy'
prs_a = 'case'+str(case)+'/prs_arteries_' + str(E) +  method +'.npy'
prs_v = 'case'+str(case)+'/prs_veins_' + str(E) + method + '.npy'
qa = 'case'+str(case)+'/Q_arteries_' + str(E) +  method +'.npy'
qv = 'case'+str(case)+'/Q_veins_' + str(E) +  method +'.npy'

np.save(solution,X)
np.save(prs_acomp,pa)
np.save(prs_vcomp,pv)
np.save(prs_a, PA)
np.save(prs_v, PV)
np.save(qa, QA)
np.save(qv, QV)
'''


# t1 = time.time()
# Y = spla.gmres(A,b,M=M)
# t2 = time.time()
# print((t2-t1)/60)