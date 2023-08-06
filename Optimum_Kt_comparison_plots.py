import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sb
sb.set(rc={'figure.figsize':(11.7,8.27)})
import pandas as pd

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(60)


# dT2 = np.load('Temperature_error_5_1_E71_Kt0.5.npy')
dT3 = np.load('Temperature_error_3_1.npy')
dT4 = np.load('Temperature_error_5_1_E71_Kt1.25.npy')

th3 = np.load('Theta_error_3_1.npy')
th4 = np.load('Theta_error_5_1_E71_Kt1.25.npy')

dT_max = max(np.max(np.abs(dT3)), np.max(np.abs(dT4)))
th_max = max(np.max(np.abs(th3)), np.max(np.abs(th4)))

n = 20
dT_array = np.arange(-dT_max, dT_max + 2*dT_max/n, dT_max/n)
th_array = np.arange(-th_max, th_max + 2*th_max/n, th_max/n)

plt.figure(figsize = (25,25), dpi = 100)
plt.subplot(2,2,1)
plt.contourf(dT3[0:-1,0:-1,80], dT_array, cmap = 'RdBu')
plt.axis('off')

plt.subplot(2,2,2)
plt.contourf(dT4[0:-1,0:-1,80], dT_array, cmap = 'RdBu')
plt.axis('off')

plt.subplot(2,2,3)
plt.contourf(th3[0:-1,0:-1,80], th_array, cmap = 'RdBu')
plt.axis('off')

plt.subplot(2,2,4)
plt.contourf(th4[0:-1,0:-1,80], th_array, cmap = 'RdBu')
plt.axis('off')


