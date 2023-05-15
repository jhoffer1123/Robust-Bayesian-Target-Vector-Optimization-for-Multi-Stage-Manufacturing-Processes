# -*- coding: utf-8 -*-
"""
Created on Mon May 15 06:54:19 2023

@author: J. G. Hoffer
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import direct, Bounds
import warnings
warnings.filterwarnings('ignore')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, ConstantKernel
from Acquisition_function_UC2 import robust_L2_EI_CASC
from BO_UC2 import BO_robust_cascaded_L2_EI, BO_robust_joint_L2_EI

#%% data

# process 1
def proc1(x11, x12):
    X11, X12= np.meshgrid(x11, x12)
    Z1 = np.log(X12)*X11
    return Z1

def proc1_(x11, x12):
    Z1 = np.log(abs(x12))*x11
    return Z1

def proc1_unc(x11, x12, sig):
    num_carlo = 10000
    fs = []
    for _ in range(0,num_carlo):
        x11 = x11.reshape(-1,1)+ np.random.normal(0, sig, len(x11)).reshape(-1,1)
        Z1 = proc1_(x11, x12)
        fs.append(Z1)
    f_mean = np.array(fs).mean(axis = 0)
    f_std = np.array(fs).std(axis = 0)
    return f_mean, f_std

# process 2
def proc2(x21, x22):
    X21, X22 = np.meshgrid(x21, x22)
    Z21 = np.sin(X21*X22/10)
    Z22 = np.cos(X21*X22/10)
    return Z21, Z22

def proc2_(x21, x22):
    Z21 = np.sin(x21*x22/10)
    Z22 = np.cos(x21*x22/10)
    return Z21, Z22

def proc2_unc(x21, x22, sig):
    num_carlo = 10000
    f1s = []
    f2s = []
    for _ in range(0,num_carlo):
        x21 = x21.reshape(-1,1)+ np.random.normal(0, sig, len(x21)).reshape(-1,1)
        Z21, Z22 = proc2_(x21.reshape(-1,1), x22)
        f1s.append(Z21)
        f2s.append(Z22)
    f1_mean = np.array(f1s).mean(axis = 0)
    f2_mean = np.array(f2s).mean(axis = 0)
    f1_std = np.array(f1s).std(axis = 0)
    f2_std = np.array(f2s).std(axis = 0)
    return f1_mean, f2_mean, f1_std, f2_std

#%% UC visualization

num_plot = 10

x11_plot = np.linspace(-5,1,num_plot).reshape(-1,1)
x12_plot = np.linspace(1,6,num_plot).reshape(-1,1)
Z1_plot = proc1(x11_plot, x12_plot)
X11_plot, X12_plot = np.meshgrid(x11_plot, x12_plot)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X11_plot, X12_plot, Z1_plot
              ,cmap='jet' 
              )
ax.set_xlabel(r'$x^{(1)}_{1}$')
ax.set_ylabel(r'$x^{(1)}_{2}$')
ax.set_zlabel(r'$y^{(1)}$')
x21_plot = np.linspace(-5,0,num_plot).reshape(-1,1)
x22_plot = np.linspace(Z1_plot.min(), Z1_plot.max(),num_plot)
X21_plot, X22_plot = np.meshgrid(x21_plot, x22_plot)
Z21_plot, Z22_plot = proc2(x21_plot, x22_plot)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X21_plot, X22_plot, Z21_plot
              ,cmap='jet' 
              )
ax.set_xlabel(r'$x^{(2)}_{2}$')
ax.set_ylabel(r'$x^{(2)}_{1}$')
ax.set_zlabel(r'$y^{(2)}_{1}$')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X21_plot, X22_plot, Z22_plot
              ,cmap='jet' 
              )
ax.set_xlabel(r'$x^{(2)}_{2}$')
ax.set_ylabel(r'$x^{(2)}_{1}$')
ax.set_zlabel(r'$y^{(2)}_{2}$')


#%% start conditions

sig = 0.01
x11_init = np.array([-5,1]).reshape(-1,1)
x12_init = np.array([2,6]).reshape(-1,1)
z11_init, z11_init_std = proc1_unc(x11_init, x12_init, sig)
x21_init = np.array([-5,0]).reshape(-1,1)
x22_init = np.array([Z1_plot.min(),Z1_plot.max()]).reshape(-1,1)
z21_init, z22_init, z21_init_std, z22_init_std = proc2_unc(x21_init, z11_init, sig = z11_init_std.ravel())
x21_target = -1
x22_target = -7
target21, target22 = proc2_(x21_target,x22_target)


#%% preparation for CASCADED
m_x1_train = np.concatenate((x11_init, x12_init), axis = 1)
m_y1_train = z11_init
sig_y1_train = z11_init_std
m_x2_train = np.concatenate((x21_init, x22_init), axis = 1)
m_y2_train = np.concatenate((z21_init, z22_init), axis = 1)
sig_y2_train = np.concatenate((z21_init_std, z22_init_std), axis = 1)
bound1 = Bounds(m_x1_train.min(axis = 0),m_x1_train.max(axis = 0))
bound2 = Bounds(m_x2_train.min(axis = 0),m_x2_train.max(axis = 0))
target22_ = np.array([target21,target22])
#%% cascaded
num_BO = None

kernel_m2 = None
m_model2 = GaussianProcessRegressor(kernel = kernel_m2)
kernel_sig2 = None
sig_model2 = GaussianProcessRegressor(kernel = kernel_sig2)
kernel_m2 = None
m_model1 = GaussianProcessRegressor(kernel = kernel_m2)


xi = None
num_MC = None
E_minss = []
robust_cascaded = BO_robust_cascaded_L2_EI(num_BO, sig, m_x2_train, sig_y1_train, m_y2_train, sig_y2_train, m_model2, sig_model2, target22_, xi, num_MC, bound1, 
                          bound2, m_x1_train, m_y1_train, m_model1, None,
                          proc1_unc, proc2_unc, target22)

#% joint

kernel_m = None
m_model = GaussianProcessRegressor(kernel = kernel_m)
m_x1_train = np.concatenate((x11_init, x12_init), axis = 1)
m_x_train = np.concatenate((m_x1_train, x21_init), axis = 1)
m_y_train = m_y2_train
sig_y_train = sig_y2_train
bound = Bounds(m_x_train.min(axis = 0), m_x_train.max(axis = 0))
target22_ = np.array([target21,target22])
robust_joint = BO_robust_joint_L2_EI(num_BO, bound, sig, m_x_train, m_y_train, sig_y_train, m_model, target22_, xi, num_MC, proc1_unc, proc2_unc, target22_)



