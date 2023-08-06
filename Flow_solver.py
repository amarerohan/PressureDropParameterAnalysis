import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pandas as pd

case = 1
E = 2
Dx = 2

a_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_db.csv')
v_db = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_db.csv')
a_pt = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_a_pt.csv').to_numpy()
v_pt = pd.read_csv('case'+str(case)+'/'+str(Dx)+'_v_pt.csv').to_numpy()
dom = np.load('case'+str(case)+'/'+str(Dx)+'_dom.npy')
c_dom = np.load('case'+str(case)+'/'+str(Dx)+'_cdom.npy')

r = np.load('case'+str(case)+'/'+str(E)+'_row.npy')
c = np.load('case'+str(case)+'/'+str(E)+'_col.npy')
d = np.load('case'+str(case)+'/'+str(E)+'_data.npy')

un_a = len(a_pt)
un_v = len(v_pt)
un_t = np.max(c_dom) + 1

N = np.max(r)+1

A = sp.csc_matrix((d,(r,c)), shape = (N,N))

Pin = 1000 
Pout = 1

b = np.zeros((N,1), dtype = int)
b[2*un_t,0] = Pin
b[2*un_t+un_a,0] = Pout

X = spla.spsolve(A,b)

'''
Zx = np.matmul(A.todense(),X).transpose()

LU = spla.spilu(A)
M = spla.LinearOperator(np.shape(LU), LU.solve)
Y = spla.gmres(A,b,M=M,tol=1e-10)[0]
'''