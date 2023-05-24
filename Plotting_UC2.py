# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:43:59 2023

@author: J. G. Hoffer
"""


import warnings
warnings.filterwarnings('ignore')

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

x11 = np.array([-1,-3,-4])
x12 = np.array([1,3,4])
z1 = np.array([-5,-4,-1])

x21 = np.array([-1,-3,-4])
x22 = z1
z21 = np.array([0,0.5,0.9])
z22 = np.array([0,-0.5,0.4])

#%% process 1 
np.random.seed(0)
def plotter(x11, x12, z1, x21, x22, z21, z22, sig, target11, target12, target21, target22, path):
#%%
    num_plot = 100
    
    x11_plot = np.linspace(-5,1,num_plot).reshape(-1,1)
    x12_plot = np.linspace(2,6,num_plot).reshape(-1,1)
    
    def proc1(x11, x12):
    
        X11, X12= np.meshgrid(x11, x12)
        X12[X12<0] = 0
        Z1 = np.log(X12)*X11
        return Z1
    
    # process 2
    def proc2(x21, x22):
    
        X21, X22 = np.meshgrid(x21, x22)
        Z21 = np.sin(X21*X22/10)
        Z22 = np.cos(X21*X22/10)
    
        return Z21, Z22
    
    def proc1_(x11, x12):
        x12[x12<0] = 0
        Z1 = np.log((x12))*x11
        return Z1
    
    # process 2
    def proc2_(x21, x22):
    
        Z21 = np.sin(x21*x22/10)
        Z22 = np.cos(x21*x22/10)
    
        return Z21, Z22
    
    #%%
    
    def proc1_unc(x11, x12, sig):
        num_carlo = 100
        fs = []
        for i in range(0,num_carlo):
            np.random.seed(i)
            x12 = x12.reshape(-1,1)
            x11 = x11.reshape(-1,1)+ np.random.normal(0, sig, len(x11)).reshape(-1,1)

            Z1 = proc1(x11, x12)
            fs.append(Z1)
    
        f_mean = np.array(fs).mean(axis = 0)
        f_std = np.array(fs).std(axis = 0)
    
        return f_mean, f_std
    
    # process 2
    def proc2_unc(x21, x22, sig):
        num_carlo = 100
        f1s = []
        f2s = []
        for i in range(0,num_carlo):
            np.random.seed(i)
            x22 = x22.reshape(-1,1)
            x21 = x21.reshape(-1,1)+ np.random.normal(0, sig, len(x21)).reshape(-1,1)
            
            Z21, Z22 = proc2(x21.reshape(-1,1), x22)
    
            f1s.append(Z21)
            f2s.append(Z22)
            
        f1_mean = np.array(f1s).mean(axis = 0)
        f2_mean = np.array(f2s).mean(axis = 0)
    
        f1_std = np.array(f1s).std(axis = 0)
        f2_std = np.array(f2s).std(axis = 0)
        
        return f1_mean, f2_mean, f1_std, f2_std
    
    
    #%%
    Z1_plot = proc1(x11_plot, x12_plot)
    Z1_plot_mean, Z1_plot_std = proc1_unc(x11_plot, x12_plot, sig)
    
    X11_plot, X12_plot = np.meshgrid(x11_plot, x12_plot)
    
    
    target_out = proc1(target11,target12) 
    
    
    #%%

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf1 = ax.plot_surface(X11_plot, X12_plot, Z1_plot
                 ,cmap='viridis' 
                 ,facecolors = cm.jet(MinMaxScaler().fit_transform(Z1_plot_std))
                 ,alpha = .2
                 )

    ax.scatter(x11, x12, z1, s = 10, color = 'black')
    
    ax.set_xlabel(r'$x^{(1)}_{1}$')
    ax.set_ylabel(r'$x^{(1)}_{2}$')
    ax.set_zlabel(r'$y^{(1)}$')
    ax.view_init(elev=10., azim=100)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.35)
    m = cm.ScalarMappable(cmap=cm.jet, norm = norm)
    m.set_array(Z1_plot_std)
    fig.colorbar(m
                 ,location = 'left'
                 , shrink = 0.5)
    plt.savefig(path+'_proc_11.pdf', bbox_inches='tight')
     
    
    #%% looking for a target
    
    x21_target = None
    x22_target = None
    
    target1, target2 = proc2_(x21_target,x22_target)
    
    #%
    
    x21_plot = np.linspace(-5,0,num_plot).reshape(-1,1)
    x22_plot = np.linspace(Z1_plot.min(), Z1_plot.max(),num_plot)
    
    X21_plot, X22_plot = np.meshgrid(x21_plot, x22_plot)
    
    Z21_plot, Z22_plot = proc2(x21_plot, x22_plot)
    Z21_plot_mean, Z22_plot_mean, Z21_plot_std, Z22_plot_std = proc2_unc(x21_plot, x22_plot, sig)
    
    #%
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax = fig.add_subplot(projection='3d')
    surf21 = ax.plot_surface(X21_plot, X22_plot, Z21_plot
                 ,facecolors = cm.jet(MinMaxScaler().fit_transform(Z21_plot_std))
                 , alpha = .2
                 )
    
    
    ax.scatter(x21_target, x22_target, target1, s = 10, color = 'red')
    ax.scatter(x21, x22, z21, s = 10, color = 'black')
    ax.set_ylabel(r'$x^{(2)}_{1} = y^{(1)}$')
    ax.set_xlabel(r'$x^{(2)}_{2}$')
    ax.set_zlabel(r'$y^{(2)}_{1}$')
    ax.view_init(elev=20., azim=320)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.15)
    m = cm.ScalarMappable(cmap=cm.jet, norm = norm)
    m.set_array(Z21_plot_std)
    fig.colorbar(m
                 ,location = 'left'
                 , shrink = 0.5)
    plt.savefig(path+'_proc_21.pdf', bbox_inches='tight')
    #%%
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf22 = ax.plot_surface(X21_plot, X22_plot, Z22_plot
                 ,cmap='viridis' 
                 ,facecolors = cm.jet(MinMaxScaler().fit_transform(Z22_plot_std))
                 , alpha = .2
                 )
    
    ax.scatter(x21_target, x22_target, target2, s = 10, color = 'red')
    ax.scatter(x21, x22, z22, s = 10, color = 'black')
    
    ax.set_ylabel(r'$x^{(2)}_{1} = y^{(1)}$')
    ax.set_xlabel(r'$x^{(2)}_{2}$')
    ax.set_zlabel(r'$y^{(2)}_{2}$')
    ax.view_init(elev=10., azim=170)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.15)
    m = cm.ScalarMappable(cmap=cm.jet, norm = norm)
    m.set_array(Z22_plot_std)
    fig.colorbar(m
                 ,location = 'left'
                 , shrink = 0.5)
    plt.savefig(path+'_proc_22.pdf', bbox_inches='tight')
    















