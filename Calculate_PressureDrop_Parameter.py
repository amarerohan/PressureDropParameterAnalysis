import numpy as np
import pandas as pd
decimal = 10

case = 5
DX = 2.5
dx = dy = dz = 0.001*DX


Q = 0.000992545
P_in = 1000
P_out = 1
myu = 1E-3

gamma_a = 1E-1
gamma_v = 1E-1

num_a_t = 1

a_db = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_db.csv')
v_db = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_db.csv')
a_out = pd.read_csv('case'+str(case)+'/'+str(DX)+'_a_out_pts.csv')
v_out = pd.read_csv('case'+str(case)+'/'+str(DX)+'_v_out_pts.csv')
c_dom = np.load('case'+str(case)+'/'+str(DX)+'_cdom.npy')

vol = (np.max(c_dom)+1)*dx*dy*dz
a = 1E-3

KA = np.empty(len(a_db), dtype = float)
KV = np.empty(len(v_db), dtype = float)
RA = np.empty(len(a_db), dtype = float)
RV = np.empty(len(v_db), dtype = float)

KA[:] = KV[:] = 0.0
RA[:] = RV[:] = 0.0

for i in range(len(a_db)):
    r = a_db.iloc[i,4]
    L = a_db.iloc[i,3]
    RA[i] = (8*myu*L)/(np.pi*(r)**4)
    KA[i] = (np.pi*(r)**4)/(8*myu*L)

for i in range(len(v_db)):
    r = v_db.iloc[i,4]
    L = v_db.iloc[i,3]
    RV[i] = (8*myu*L)/(np.pi*(r)**4)
    KV[i] = (np.pi*(r)**4)/(8*myu*L)

# np.save('KA.npy',KA)
# np.save('KV.npy',KV)


def parallel(G1, G2):
    G = G1 + G2
    return (G)

def series(G1, G2):
    if G1 == 0:
        G = G2
    if G2 == 0:
        G = G1
    if (G1!=0 and G2!=0):
        G = (1/G1 + 1/G2)**(-1)
    return(G)

# # # ARTERIAL TREE # # # 
Layer = int(np.log(len(a_out))/np.log(2))
G_eq = []
for i in range(Layer+1):
    G_eq.append([])

if gamma_a > 0:
    for i in range(len(KA)-2**Layer, len(KA)):
        k = series(KA[i],num_a_t*gamma_a/myu) #; print(KA[i], gamma_a/myu)
        G_eq[Layer].append([i,k]) # ; print(k)

if gamma_a <= 0:
    for i in range(len(KA)-2**Layer, len(KA)):
        G_eq[Layer].append([i,KA[i]])

G_eq_terminal = []
for i in reversed(range(1,Layer + 1)):
    ref_Geq = np.array(G_eq[i])
    
    # PARALLEL
    for j in range(2**(i-1)):
        index = 2**(i-1) + j
        ele = a_db.loc[a_db.iloc[:,1] == index]
        G1 = ref_Geq[ref_Geq[:,0] == int(ele.iloc[0,0])][0][1]
        G2 = ref_Geq[ref_Geq[:,0] == int(ele.iloc[1,0])][0][1]
        g = parallel(G1,G2)
        G_eq[i-1].append([index-1,round(g,decimal)])# ; print(index,j)
        
        if i==Layer:
            G_eq_terminal.append(g)
        
        
        
    ref_Geq = np.array(G_eq[i-1])
    
    # SERIES 
    for j in range(len(ref_Geq)):
        ele = a_db.loc[a_db.iloc[:,0] == ref_Geq[j,0]]
        G1 = ref_Geq[ref_Geq[:,0] == int(ele.iloc[0,0])][0][1]
        G2 = KA[int(ele.iloc[0,0])]
        g = series(G1,G2)
        G_eq[i-1][j][1] = round(g,decimal)
        
Eq_Ga = G_eq[0][0][1]
Eq_Ra = round(1/Eq_Ga, 3)

# CALCULATE CONDUCTIVITY OF VENOUS TREE


G_eq_v = []
for i in range(Layer+1):
    G_eq_v.append([])
    
if gamma_v > 0:
    for i in range(len(KV)-2**Layer, len(KV)):
        k = series(KV[i],num_a_t*gamma_v/myu) #; print(KA[i], gamma_a/myu)
        G_eq_v[Layer].append([i,k]) # ; print(k)

if gamma_v <= 0:
    for i in range(len(KV)-2**Layer, len(KV)):
        G_eq_v[Layer].append([i,KV[i]])

G_eq_terminal = []
for i in reversed(range(1,Layer + 1)):
    ref_Geq = np.array(G_eq_v[i])
    
    # PARALLEL
    for j in range(2**(i-1)):
        index = 2**(i-1) + j
        ele = v_db.loc[v_db.iloc[:,1] == index]
        G1 = ref_Geq[ref_Geq[:,0] == int(ele.iloc[0,0])][0][1]
        G2 = ref_Geq[ref_Geq[:,0] == int(ele.iloc[1,0])][0][1]
        g = parallel(G1,G2)
        G_eq_v[i-1].append([index-1,round(g,decimal)])# ; print(index,j)
        
        if i==Layer:
            G_eq_terminal.append(g)
        
        
        
    ref_Geq = np.array(G_eq_v[i-1])
    
    # SERIES 
    for j in range(len(ref_Geq)):
        ele = v_db.loc[v_db.iloc[:,0] == ref_Geq[j,0]]
        G1 = ref_Geq[ref_Geq[:,0] == int(ele.iloc[0,0])][0][1]
        G2 = KV[int(ele.iloc[0,0])]
        g = series(G1,G2)
        G_eq_v[i-1][j][1] = round(g,decimal)
        
Eq_Gv = G_eq_v[0][0][1]
Eq_Rv = round(1/Eq_Gv, 3)

phi = Eq_Ra/Eq_Rv

Pa_terminal = P_in - Q*Eq_Ra
Pv_terminal = P_out + Q*Eq_Rv
dPt = 1/(a*vol)*Q

dPv = (Pa_terminal - Pv_terminal - dPt)/(1+phi)
dPa = (Pa_terminal - Pv_terminal - dPt)*phi/(1+phi)

Ga = Q/(2**Layer)*myu/dPa
Gv = Q/(2**Layer)*myu/dPv



print(Layer, '\n' , round(dPa,3), round(dPt,3), round(dPv,3), format(Ga,'.2E'), format(Gv,'.2E'))
print('Pressure drop across arterial Vascular = ', Q*Eq_Ra)
print('Pressure drop across venous vasculature = ', Q*Eq_Rv)

# X = np.load('case1/X_28.npy')
