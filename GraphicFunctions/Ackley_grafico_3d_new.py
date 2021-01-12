# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:01:53 2019

@author: rrs
"""

from math import *
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
#if using a Jupyter notebook, include: %matplotlib inline

#Abrir gráfico em nova janela
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
step = 4

xx = np.arange(-32,32,step)
yy = np.arange(-32,32,step)
N=len(xx)
NN = N*N
zz =np.zeros((N,N))
X,Y = np.meshgrid(xx,yy)
x=np.zeros(2)
for i in range(N):
    for j in range(N):
        x[0]=xx[i]; x[1]=yy[j];
        W=-20*exp(-0.2*sqrt(1/len(x)*sum([i**2 for i in x]))) - \
           exp(1/len(x)*sum([cos(2*np.pi*i) for i in x])) + 20 + exp(1)
        print(W)
        zz[i,j]=W*1.0;

#Ackley
#-20*exp(-0.2*sqrt(1/len(x)*sum([i**2 for i in x]))) - \
#           exp(1/len(x)*sum([cos(2*np.pi*i)

Z = zz.copy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.grid(False) # esconder linhas de grade
#mycmap = plt.get_cmap('gist_earth')

ax.azim = -140
ax.elev = 30
ax.view_init(ax.elev, ax.azim)

mycmap = plt.get_cmap('jet')
#ax.set_title('gist_earth color map')
#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
surf1 = ax.plot_surface(X, Y, Z, cmap=mycmap)
#surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)

cset = ax.contour(X, Y, Z, zdir='z', offset=-3, cmap=mycmap)
plt.xlabel('X',fontsize=14, rotation=-150)
plt.ylabel('Y',fontsize=14, rotation=-30)
ax.set_zlabel('Z',fontsize=14)
plt.show()

