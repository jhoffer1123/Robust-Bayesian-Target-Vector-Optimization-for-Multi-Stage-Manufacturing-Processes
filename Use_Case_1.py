# -*- coding: utf-8 -*-
"""
Created on Mon May 15 06:54:12 2023

@author: J. G. Hoffer
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import direct
from Acquisition_function_UC1 import robust_L2_EI, L2_EI
from BO_UC1 import BO_loop
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

#%% data

def f1(x):
    return np.cos(x).reshape(-1,1) + ((x**2) /4).reshape(-1,1)

def f2(x):
    return np.cos(x**2).reshape(-1,1) + ((x**2) /4).reshape(-1,1)

def f(x):
    return np.concatenate((f1(x),f2(x)), axis = 1)

def y(x, sig):
    num_carlo = 10000
    f1s = []
    f2s = []
    for _ in range(0,num_carlo):
        x_noise = x+ np.random.normal(0, sig, len(x))
        f1s.append(f1(x_noise))
        f2s.append(f2(x_noise))
    f1_mean = np.array(f1s).mean(axis = 0)
    f2_mean = np.array(f2s).mean(axis = 0)
    f1_std = np.array(f1s).std(axis = 0)
    f2_std = np.array(f2s).std(axis = 0)
    return np.concatenate((f1_mean, f2_mean), axis = 1), np.concatenate((f1_std, f2_std), axis = 1)

num_plot_points = 300
start = 0.5
end = 4
sig=.1
x_plot = np.linspace(start,end,num_plot_points)
x_plot_noise = np.linspace(start,end,num_plot_points)+ np.random.normal(0, sig, num_plot_points)
f_mean_plot, _ = y(x_plot,0)
f_mean, f_std = y(x_plot,sig)

#%% start conditions

x_init = np.array([start,end])
y_init, std_init = y(x_init, sig)
y_init, _ = y(x_init, 0)
sig_x_train = np.array([[sig],[sig]])
m_x_train = x_init.reshape(-1,1)
m_y_train = y_init
sig_y_train = std_init
x_target = 2
target, target_stds = y(np.array([x_target]), 0)
E_target = (target_stds**2).mean()
E = ((target- f_mean_plot)**2 + f_std**2).mean(axis = 1)
E_min = E.min()
x_E_min = x_plot[E.argmin()]
MSE = ((target- f_mean_plot)**2).mean(axis = 1)
bounds = Bounds([start],[end])

#%% UC visualization

fontsize = 16
fig, ax1 = plt.subplots(figsize = (8,5))
ax2 = ax1.twinx()
ax2.plot(x_plot,E, label = r'$E$', color = 'grey', linestyle = '--')
ax2.plot(x_plot,MSE, label = r'$squared-error$', color = 'grey', linestyle = ':')
ax1.plot(x_plot,f_mean_plot[:,0], label = r'$m_{y_1}$', color = 'blue')
ax1.plot(x_plot,f_mean_plot[:,1], label = r'$m_{y_2}$', color = 'magenta')
ax2.axvline(x=2 , color = 'black', alpha = .4, linestyle = '-')
ax2.text(2, 1.3,  r'$m_{x_{target}}$')
ax2.axvline(x=x_E_min, color = 'black', alpha = .4, linestyle = '-')
plt.text(1.3, 1.3, r'$E_{min}$')
ax1.plot([start,end], [target[0,0], target[0,0]], label = r'$y_1^*$', color = 'blue', linestyle = '-.')
ax1.plot([start,end], [target[0,1], target[0,1]], label = r'$y_2^*$', color = 'magenta', linestyle = '-.')
ax2.set_yscale('log')
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'lower right')
ax1.set_xlabel(r'$x$', fontsize = fontsize)
ax1.set_ylabel(r'$ y_1, y_2 $', fontsize = fontsize)
ax2.set_ylabel(r'$E,~ squared-error$', fontsize = fontsize)

#%% Evaluation

alpha = None
num_it = None
kernel_robust = None
m_model_robust = None
sig_model_robust = None
kernel_nonrobust = None
m_model_nonrobust = None
sig_model_nonrobust = None
xi = None

robust_L2_EI_results = BO_loop(bounds, m_x_train, sig_x_train, m_y_train, sig_y_train, m_model_robust, sig_model_robust, target, xi, robust_L2_EI, num_it, num_MC = 20)
L2_EI_results = BO_loop(bounds, m_x_train, sig_x_train, m_y_train, sig_y_train, m_model_nonrobust, sig_model_nonrobust, target, xi, L2_EI, num_it, num_MC = 0, robust = False)

#%% Visualization

fig, ax = plt.subplots(figsize = (8,4))
ax.plot(robust_L2_EI_results[-1], color = 'green')
ax.plot(robust_L2_EI_results[-1], label = r'$robust$', color = 'green', marker = 'o', alpha = .5, markersize = 10)
ax.plot(L2_EI_results[-1], color = 'red')
ax.plot(L2_EI_results[-1], label = r'$non-robust$', color = 'red', marker = '^', alpha = .5, markersize = 10)
ax.set_ylabel(r'$E_{min}$', fontsize = fontsize)
ax.set_xlabel(r'$optimization~ iteration$', fontsize = fontsize)
ax.set_yscale('log')
default_x_ticks = range(1, len(L2_EI_results[-1]),2)
ax.plot([0,num_it-1],[E_min,E_min], label = r'$E_{min}$', color = 'black', alpha = .4, linestyle = '--')
ax.set_xticks(default_x_ticks)
ax.legend()


























