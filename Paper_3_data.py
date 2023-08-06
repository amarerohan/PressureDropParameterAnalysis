import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(40)

c1 = 1
c2 = 2
c3 = 3
c4 = 4
c5 = 5

e1 = 10
e2 = 10
e3 = 10
e4 = 10
e5 = 10

pa1 = np.load('case'+str(c1)+'/flow_solutions/'+str(e1)+'/prs_acomp_' + str(e1) +'_Pin_1000_Pout_1.npy')
pv1 = np.load('case'+str(c1)+'/flow_solutions/'+str(e1)+'/prs_vcomp_' + str(e1) +'_Pin_1000_Pout_1.npy')

pa2 = np.load('case'+str(c2)+'/flow_solutions/'+str(e2)+'/prs_acomp_' + str(e2) +'_Pin_1000_Pout_1.npy')
pv2 = np.load('case'+str(c2)+'/flow_solutions/'+str(e2)+'/prs_vcomp_' + str(e2) +'_Pin_1000_Pout_1.npy')

pa3 = np.load('case'+str(c3)+'/flow_solutions/'+str(e3)+'/prs_acomp_' + str(e3) +'_Pin_1000_Pout_1.npy')
pv3 = np.load('case'+str(c3)+'/flow_solutions/'+str(e3)+'/prs_vcomp_' + str(e3) +'_Pin_1000_Pout_1.npy')

pa4 = np.load('case'+str(c4)+'/flow_solutions/'+str(e4)+'/prs_acomp_' + str(e4) +'_Pin_1000_Pout_1.npy')
pv4 = np.load('case'+str(c4)+'/flow_solutions/'+str(e4)+'/prs_vcomp_' + str(e4) +'_Pin_1000_Pout_1.npy')

pa5 = np.load('case'+str(c5)+'/flow_solutions/'+str(e5)+'/prs_acomp_' + str(e5) +'_Pin_1000_Pout_1.npy')
pv5 = np.load('case'+str(c5)+'/flow_solutions/'+str(e5)+'/prs_vcomp_' + str(e5) +'_Pin_1000_Pout_1.npy')

qa1 = np.load('case'+str(c1)+'/flow_solutions/'+str(e1)+'/Q_arteries_10_Pin_1000_Pout_1.npy')
qv1 = np.load('case'+str(c1)+'/flow_solutions/'+str(e1)+'/Q_veins_10_Pin_1000_Pout_1.npy')
x1 = np.load('case'+str(c1)+'/flow_solutions/'+str(e1)+'/X_10_Pin_1000_Pout_1.npy')

qa2 = np.load('case'+str(c2)+'/flow_solutions/'+str(e2)+'/Q_arteries_10_Pin_1000_Pout_1.npy')
qv2 = np.load('case'+str(c2)+'/flow_solutions/'+str(e2)+'/Q_veins_10_Pin_1000_Pout_1.npy')
x2 = np.load('case'+str(c2)+'/flow_solutions/'+str(e2)+'/X_10_Pin_1000_Pout_1.npy')

qa3 = np.load('case'+str(c3)+'/flow_solutions/'+str(e3)+'/Q_arteries_10_Pin_1000_Pout_1.npy')
qv3 = np.load('case'+str(c3)+'/flow_solutions/'+str(e3)+'/Q_veins_10_Pin_1000_Pout_1.npy')
x3 = np.load('case'+str(c3)+'/flow_solutions/'+str(e3)+'/X_10_Pin_1000_Pout_1.npy')

qa4 = np.load('case'+str(c4)+'/flow_solutions/'+str(e4)+'/Q_arteries_10_Pin_1000_Pout_1.npy')
qv4 = np.load('case'+str(c4)+'/flow_solutions/'+str(e4)+'/Q_veins_10_Pin_1000_Pout_1.npy')
x4 = np.load('case'+str(c4)+'/flow_solutions/'+str(e4)+'/X_10_Pin_1000_Pout_1.npy')

qa5 = np.load('case'+str(c5)+'/flow_solutions/'+str(e5)+'/Q_arteries_10_Pin_1000_Pout_1.npy')
qv5 = np.load('case'+str(c5)+'/flow_solutions/'+str(e5)+'/Q_veins_10_Pin_1000_Pout_1.npy')
x5 = np.load('case'+str(c5)+'/flow_solutions/'+str(e5)+'/X_10_Pin_1000_Pout_1.npy')

p1 = x1[-2*len(qa1)-2:]
p2 = x2[-2*len(qa2)-2:]
p3 = x3[-2*len(qa3)-2:]
p4 = x4[-2*len(qa4)-2:]
p5 = x5[-2*len(qa5)-2:]


N1 = int(len(x1[:-len(p1)])/2)
N2 = int(len(x2[:-len(p2)])/2)
N3 = int(len(x3[:-len(p3)])/2)
N4 = int(len(x4[:-len(p4)])/2)
N5 = int(len(x5[:-len(p5)])/2)

print(np.average(x1[:N1]))
print(np.average(x2[:N2]))
print(np.average(x3[:N3]))
print(np.average(x4[:N4]))
print(np.average(x5[:N5]))

print(np.max(x1[:N1]))
print(np.max(x2[:N2]))
print(np.max(x3[:N3]))
print(np.max(x4[:N4]))
print(np.max(x5[:N5]))

print(np.min(x1[:N1]))
print(np.min(x2[:N2]))
print(np.min(x3[:N3]))
print(np.min(x4[:N4]))
print(np.min(x5[:N5]))


print(np.average(x1[N1:2*N1]))
print(np.average(x2[N2:2*N2]))
print(np.average(x3[N3:2*N3]))
print(np.average(x4[N4:2*N4]))
print(np.average(x5[N5:2*N5]))

print(np.max(x1[N1:N2*1]))
print(np.max(x2[N2:2*N2]))
print(np.max(x3[N3:2*N3]))
print(np.max(x4[N4:2*N4]))
print(np.max(x5[N5:2*N5]))

print(np.min(x1[N1:2*N1]))
print(np.min(x2[N2:2*N2]))
print(np.min(x3[N3:2*N3]))
print(np.min(x4[N4:2*N4]))
print(np.min(x5[N5:2*N5]))


