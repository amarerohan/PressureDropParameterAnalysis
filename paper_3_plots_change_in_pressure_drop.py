import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import FontProperties


colormaps = cm.get_cmap('Set1',8)
markerset = ['o','*','v','^','s','X']

xa = np.arange(0,6,1)
xv = np.arange(10,16,1)

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(40)

c = 2

e = 10

g_ref = 3.78E-10

pa0 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_arteries_10_Pin_1000_Pout_1.npy')
pv0 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_veins_10_Pin_1000_Pout_1.npy')
qa0 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_arteries_10_Pin_1000_Pout_1.npy')
qv0 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_veins_10_Pin_1000_Pout_1.npy')

g = g_ref*1.01
pa1 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
pv1 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qa1 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qv1 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')

g = g_ref*1.02
pa2 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
pv2 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qa2 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qv2 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')

g = g_ref*1.05
pa3 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
pv3 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qa3 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qv3 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')

g = g_ref*0.99
pa4 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
pv4 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qa4 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qv4 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')

g = g_ref*0.98
pa5 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
pv5 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qa5 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qv5 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')

g = g_ref*0.95
pa6 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
pv6 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/prs_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qa6 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_arteries_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')
qv6 = np.load('case'+str(c)+'/flow_solutions/'+str(e)+'/Q_veins_10_Pin_1000_Pout_1_Ka_1e-05_Ga_'+str(g)+'.npy')

a0 = []
a0.append((pa0[0]))
a0.append((pa0[1]))
a0.append(np.average(pa0[2:4]))
# a0.append(np.average(pa0[4:8]))
# a0.append(np.average(pa0[8:16]))
# a0.append(np.average(pa0[16:]))

v0 = []
# v0.append(np.average(pv0[16:]))
# v0.append(np.average(pv0[8:16]))
# v0.append(np.average(pv0[4:8]))
v0.append(np.average(pv0[2:4]))
v0.append((pv0[1]))
v0.append((pv0[0]))




a1 = []
a1.append((pa1[0]))
a1.append((pa1[1]))
a1.append(np.average(pa1[2:4]))
# a1.append(np.average(pa1[4:8]))
# a1.append(np.average(pa1[8:16]))
# a1.append(np.average(pa1[16:]))

v1 = []
# v1.append(np.average(pv1[16:]))
# v1.append(np.average(pv1[8:16]))
# v1.append(np.average(pv1[4:8]))
v1.append(np.average(pv1[2:4]))
v1.append((pv1[1]))
v1.append((pv1[0]))



a2 = []
a2.append((pa2[0]))
a2.append((pa2[1]))
a2.append(np.average(pa2[2:4]))
a2.append(np.average(pa2[4:8]))
a2.append(np.average(pa2[8:16]))
a2.append(np.average(pa2[16:]))

v2 = []
v2.append(np.average(pv2[16:]))
v2.append(np.average(pv2[8:16]))
v2.append(np.average(pv2[4:8]))
v2.append(np.average(pv2[2:4]))
v2.append((pv2[1]))
v2.append((pv2[0]))





a3 = []
a3.append((pa3[0]))
a3.append((pa3[1]))
a3.append(np.average(pa3[2:4]))
a3.append(np.average(pa3[4:8]))
a3.append(np.average(pa3[8:16]))
a3.append(np.average(pa3[16:]))

v3 = []
v3.append(np.average(pv3[16:]))
v3.append(np.average(pv3[8:16]))
v3.append(np.average(pv3[4:8]))
v3.append(np.average(pv3[2:4]))
v3.append((pv3[1]))
v3.append((pv3[0]))






a4 = []
a4.append((pa4[0]))
a4.append((pa4[1]))
a4.append(np.average(pa4[2:4]))
a4.append(np.average(pa4[4:8]))
a4.append(np.average(pa4[8:16]))
a4.append(np.average(pa4[16:]))

v4 = []
v4.append(np.average(pv4[16:]))
v4.append(np.average(pv4[8:16]))
v4.append(np.average(pv4[4:8]))
v4.append(np.average(pv4[2:4]))
v4.append((pv4[1]))
v4.append((pv4[0]))



a5 = []
a5.append((pa5[0]))
a5.append((pa5[1]))
a5.append(np.average(pa5[2:4]))
a5.append(np.average(pa5[4:8]))
a5.append(np.average(pa5[8:16]))
a5.append(np.average(pa5[16:]))

v5 = []
v5.append(np.average(pv5[16:]))
v5.append(np.average(pv5[8:16]))
v5.append(np.average(pv5[4:8]))
v5.append(np.average(pv5[2:4]))
v5.append((pv5[1]))
v5.append((pv5[0]))



a6 = []
a6.append((pa6[0]))
a6.append((pa6[1]))
a6.append(np.average(pa6[2:4]))
a6.append(np.average(pa6[4:8]))
a6.append(np.average(pa6[8:16]))
a6.append(np.average(pa6[16:]))

v6 = []
v6.append(np.average(pv6[16:]))
v6.append(np.average(pv6[8:16]))
v6.append(np.average(pv6[4:8]))
v6.append(np.average(pv6[2:4]))
v6.append((pv6[1]))
v6.append((pv6[0]))

plt.figure(figsize = (12,8), dpi=300)

color = colormaps(0)
marker = markerset[0]
plt.plot(xa,a0, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'a0')
#plt.plot(xv,v0, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'v0')

color = colormaps(1)
marker = markerset[1]
plt.plot(xa,a1, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'a0')
#plt.plot(xv,v1, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'v0')

color = colormaps(2)
marker = markerset[2]
plt.plot(xa,a2, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'a0')
# plt.plot(xv,v2, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'v0')

color = colormaps(3)
marker = markerset[3]
plt.plot(xa,a3, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'a0')
# plt.plot(xv,v3, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'v0')

# color = colormaps(4)
# marker = markerset[4]
# plt.plot(xa,a4, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'a0')
# # plt.plot(xv,v4, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'v0')

# color = colormaps(5)
# marker = markerset[5]
# plt.plot(xa,a5, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'a0')
# # plt.plot(xv,v5, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'v0')

# color = colormaps(6)
# marker = markerset[6]
# plt.plot(xa,a6, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'a0')
# # plt.plot(xv,v6, color = color, marker = marker, markersize = 10, alpha = 1.0, label = 'v0')

plt.show()


colormaps = cm.get_cmap('Set1',4)


x = [-5,-2,-1,0,1,2,5]
n6 = [0.636595202,0.250238287,0.123744208,0,-0.122369272,-0.243363608,-0.595347133]
n5 = [0.415659305,0.163028051,0.080891781,0,-0.079647292,-0.158050095,-0.388280548]
n4 = [0.225454758,0.088369202,0.044184601,0,-0.043051662,-0.086103325,-0.210726558]

font.set_size(20)
plt.figure(figsize = (10,5), dpi = 300)
plt.plot(x,n6, color = colormaps(0), marker = markerset[0], markersize = 10, label = 'Nodes 5')
plt.plot(x,n5, color = colormaps(1), marker = markerset[1], markersize = 10, label = 'Nodes 4')
plt.plot(x,n4, color = colormaps(2), marker = markerset[2], markersize = 10, label = 'Nodes 3')
plt.grid(True)
plt.legend(prop = font, fontsize = 18)
plt.xticks(np.arange(-5,6,1),font=font, fontsize = 20)
plt.yticks(np.arange(-0.6,0.8,0.2),font=font, fontsize = 20)
plt.xlabel('Percentage change in $\gamma$', font = font, fontsize = 20)
plt.ylabel('Percentage difference in Pressure', font = font, fontsize = 20)
plt.savefig('Pressure_drop_variation.jpg', dpi=300, bbox_inches='tight')
plt.show()





    