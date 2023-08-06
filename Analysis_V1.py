import numpy as np
import matplotlib.pyplot as plt

case = 2
dx = 2.5
E = 5

un_t = 506086

Pin = 1000 
Pout = 1

solution = 'case'+str(case)+'/flow_solutions/X_' + str(E) + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_acomp = 'case'+str(case)+'/flow_solutions/prs_acomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_vcomp = 'case'+str(case)+'/flow_solutions/prs_vcomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_a = 'case'+str(case)+'/flow_solutions/prs_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_v = 'case'+str(case)+'/flow_solutions/prs_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qa = 'case'+str(case)+'/flow_solutions/Q_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qv = 'case'+str(case)+'/flow_solutions/Q_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'

X = np.load(solution)
pa_comp = np.load(prs_acomp)
pv_comp = np.load(prs_vcomp)
PA = np.load(prs_a)
PV = np.load(prs_v)
QA = np.load(qa)
QV = np.load(qv)

del(solution, prs_acomp, prs_vcomp, prs_a, prs_v, qa, qv)

min_pa = round(np.min(X[0:un_t]),3)
max_pa = round(np.max(X[0:un_t]),3)

min_pv = round(np.min(X[un_t:2*un_t]),3)
max_pv = round(np.max(X[un_t:2*un_t]),3)

z = 20
plt.figure(figsize = (12,10))

plt.contourf(pa_comp[1:-1,1:-1,z], np.arange(min_pa, max_pa, 0.005))
plt.colorbar()

plt.figure(figsize = (12,10))
plt.contourf(pv_comp[1:-1,1:-1,z], np.arange(min_pv, max_pv, 0.005))
plt.colorbar()




case = 1
dx = 2.5
E = 5

un_t = 504406

Pin = 1000 
Pout = 1

solution = 'case'+str(case)+'/flow_solutions/X_' + str(E) + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_acomp = 'case'+str(case)+'/flow_solutions/prs_acomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_vcomp = 'case'+str(case)+'/flow_solutions/prs_vcomp_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_a = 'case'+str(case)+'/flow_solutions/prs_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
prs_v = 'case'+str(case)+'/flow_solutions/prs_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qa = 'case'+str(case)+'/flow_solutions/Q_arteries_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'
qv = 'case'+str(case)+'/flow_solutions/Q_veins_' + str(E)  + '_Pin_'+str(Pin) + '_Pout_' + str(Pout)+ '.npy'

X1 = np.load(solution)
pa_comp1 = np.load(prs_acomp)
pv_comp1 = np.load(prs_vcomp)
PA1 = np.load(prs_a)
PV1 = np.load(prs_v)
QA1 = np.load(qa)
QV1 = np.load(qv)

del(solution, prs_acomp, prs_vcomp, prs_a, prs_v, qa, qv)

min_pa1 = round(np.min(X1[0:un_t]),3)
max_pa1 = round(np.max(X1[0:un_t]),3)

min_pv1 = round(np.min(X1[un_t:2*un_t]),3)
max_pv1 = round(np.max(X1[un_t:2*un_t]),3)

z = 20
plt.figure(figsize = (12,10))

plt.contourf(pa_comp1[1:-1,1:-1,z], np.arange(min_pa1, max_pa1, 0.005))
plt.colorbar()

plt.figure(figsize = (12,10))
plt.contourf(pv_comp1[1:-1,1:-1,z], np.arange(min_pv1, max_pv1, 0.005))
plt.colorbar()


