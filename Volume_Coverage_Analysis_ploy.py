import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib
from matplotlib.font_manager import FontProperties


font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(15)
f = 20

colormaps = cm.get_cmap('Set1',5)
markerset = list(Line2D.markers.keys())
lines_array = list(matplotlib.lines.lineStyles.keys())


db = np.load('volume_coverage.npy', allow_pickle=True)
db = pd.DataFrame(db)

c1 = db.loc[(db.iloc[:,0] == 1)]
c2 = db.loc[(db.iloc[:,0] == 2)]
c3 = db.loc[(db.iloc[:,0] == 3)]
c4 = db.loc[(db.iloc[:,0] == 4)]
c5 = db.loc[(db.iloc[:,0] == 5)]

plt.figure(figsize = (8,4), dpi = 300)
plt.plot(c1.iloc[:,1], c1.iloc[:,2], color = colormaps(0), marker = markerset[0], markersize = 12)
plt.plot(c2.iloc[:,1], c2.iloc[:,2], color = colormaps(1), marker = markerset[0], markersize = 12)
plt.plot(c3.iloc[:,1], c3.iloc[:,2], color = colormaps(2), marker = markerset[0], markersize = 12)
plt.plot(c4.iloc[:,1], c4.iloc[:,2], color = colormaps(3), marker = markerset[0], markersize = 12)
plt.plot(c5.iloc[:,1], c5.iloc[:,2], color = colormaps(4), marker = markerset[0], markersize = 12)


plt.plot(c1.iloc[:,1], c1.iloc[:,3], color = colormaps(0), marker = markerset[12], markersize = 6, linestyle = '--')
plt.plot(c2.iloc[:,1], c2.iloc[:,3], color = colormaps(1), marker = markerset[12], markersize = 6, linestyle = '--')
plt.plot(c3.iloc[:,1], c3.iloc[:,3], color = colormaps(2), marker = markerset[12], markersize = 6, linestyle = '--')
plt.plot(c4.iloc[:,1], c4.iloc[:,3], color = colormaps(3), marker = markerset[12], markersize = 6, linestyle = '--')
plt.plot(c5.iloc[:,1], c5.iloc[:,3], color = colormaps(4), marker = markerset[12], markersize = 6, linestyle = '--')




plt.grid(True)


# db2 = np.load('volume_coverage_beocat.npy', allow_pickle=True)
# db2 = pd.DataFrame(db2)