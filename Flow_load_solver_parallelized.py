import numpy as np
import pandas as pd
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


case = 1
E = 38
Dx = 2

a = 1e-3
gamma_a = 0.1

smaller_domain = True

a_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_db.csv')
v_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_db.csv')
dom = np.load('case'+str(case)+'/'+str(Dx)+'_dom.npy')
c_dom = np.load('case'+str(case)+'/'+str(Dx)+'_cdom.npy')

r_a_comp = np.load('case'+str(case)+'/'+ 'row_arterial_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')
c_a_comp = np.load('case'+str(case)+'/'+ 'col_arterial_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')
d_a_comp = np.load('case'+str(case)+'/'+ 'data_arterial_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')

r_v_comp = np.load('case'+str(case)+'/'+ 'row_venous_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')
c_v_comp = np.load('case'+str(case)+'/'+ 'col_venous_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')
d_v_comp = np.load('case'+str(case)+'/'+ 'data_venous_compartment_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')

r_vasc = np.load('case'+str(case)+'/'+ 'row_vasculature_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')
c_vasc = np.load('case'+str(case)+'/'+ 'col_vasculature_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')
d_vasc = np.load('case'+str(case)+'/'+ 'data_vasculature_a_' + str(a) + '_G_' + str(gamma_a) + '_E_' + str(E) + '_.npy')

r = np.concatenate((r_a_comp, r_v_comp, r_vasc))
c = np.concatenate((c_a_comp, c_v_comp, c_vasc))
d = np.concatenate((d_a_comp, d_v_comp, d_vasc))

if smaller_domain == True:
    E0 = 5
    r0 = np.load('case'+str(case)+'/'+str(E0)+'_row.npy')
    c0 = np.load('case'+str(case)+'/'+str(E0)+'_col.npy')
    d0 = np.load('case'+str(case)+'/'+str(E0)+'_data.npy')
    N0 = int(np.max(r0)+1)
    
    A0 = sp.csc_matrix((d0,(r0,c0)), shape = (N0,N0))

un_a = len(a_db) + 1 
un_v = len(v_db) + 1 
un_t = np.max(c_dom) + 1

N = np.max(r)+1

Pin = 1000 
Pout = 1

b = np.zeros((N,1), dtype = int)
b[2*un_t,0] = Pin
b[2*un_t+un_a,0] = Pout

x = np.zeros((N,1), dtype = float)
x[:] = 1.0

max_iter = 1000
tol = 1E-10
start = time.time()
A = sp.csc_matrix((d,(r,c)), shape = (N,N))

if smaller_domain == True:
    LU = spla.spilu(A0) 
if smaller_domain == False:        
    LU = spla.spilu(A)

M = spla.LinearOperator(np.shape(LU), LU.solve)    
x0 = np.zeros((N,1), dtype = float)
x0[:] = 1.0
X = spla.lgmres(A,b,M=M,x0=x0, maxiter = max_iter, tol = tol, inner_m = 200, atol = tol)
reason = X[1]
X = X[0]
method = 'lgmres'
stop = time.time()
print('lgmres ', round((stop-start)/60,3), ' mins', ' reason = ', reason)



# # # # RESIDUAL AND ERROR MEASUREMENT # # # 

res = A.multiply(X)
res = res.sum(axis = 1) - b
residual = np.linalg.norm(res)


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
print('Flow going in = ', QA[0])
print('Flow going out = ', QV[0])


nx,ny,nz = np.shape(dom)

pa = np.zeros((nx,ny,nz), dtype = float)
pv = np.zeros((nx,ny,nz), dtype = float)



pa[:,:,:] = 0.0
pv[:,:,:] = 0.0


for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if(dom[i,j,k] == 0):
                h = c_dom[i,j,k]
                pa[i,j,k] = Prs_a_comp[h]
                pv[i,j,k] = Prs_v_comp[h]



# # # SAVE SOLUTIONS # # # # 
solution = 'case'+str(case)+'/X_' + str(E) + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '_2.npy'
prs_acomp = 'case'+str(case)+'/prs_acomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '_2.npy'
prs_vcomp = 'case'+str(case)+'/prs_vcomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '_2.npy'
prs_a = 'case'+str(case)+'/prs_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '_2.npy'
prs_v = 'case'+str(case)+'/prs_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '_2.npy'
qa = 'case'+str(case)+'/Q_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '_2.npy'
qv = 'case'+str(case)+'/Q_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '_2.npy'

np.save(solution,X)
np.save(prs_acomp,pa)
np.save(prs_vcomp,pv)
np.save(prs_a, PA)
np.save(prs_v, PV)
np.save(qa, QA)
np.save(qv, QV)


