import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties 
import seaborn as sb
import pandas as pd
import time as time

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(60)


E_array = [0,38,49,53,66,71]

c = 5

c_dom = np.load('case'+str(c)+'/2.5_cdom.npy')
E = 71 #E_array[c]
n1 = np.load('case_'+str(c)+'_E_'+str(E)+'_nbr_spread.npy')

c = 5
E = 100 #E_array[c]
n2 = np.load('case_'+str(c)+'_E_'+str(E)+'_nbr_spread.npy')


en2 = n2-n1

n_voxels = len(np.where((en2 != 0)))
N = np.max(c_dom)+1

volume = (N + n_voxels)/N*100
print(volume)