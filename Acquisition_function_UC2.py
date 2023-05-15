# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:40:13 2023

@author: J. G. Hoffer
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ncx2

#%% cascaded ACQ
def robust_L2_EI_CASC(m_x, sig_xp, m_x_train, sig_x_train, m_y_train, sig_y_train, m_model, sig_model, target, xi, num_MC, proc):
    m_model.fit(m_x_train, m_y_train)
    if proc==2:
        sig_model.fit(m_x_train[:,1].reshape(-1,1), sig_x_train)   
        sig_xpp = sig_model.predict(np.array([[m_x[1]]]))
    bar = []
    bar_train = []
    for i in range(0,num_MC):
        if proc==2:
            x = m_x
            x_train = m_x_train
            x[0] = m_x[0]+ np.random.normal(0, sig_xp, 1)
            x[1] = m_x[1]+ np.random.normal(0, abs(sig_xpp).ravel(), 1)
            x_train[:,0] = m_x_train[:,0] + np.random.normal(0, sig_xp, len(m_x_train))
            x_train[:,1] = m_x_train[:,1] + np.random.normal(0, sig_x_train.ravel(), len(m_x_train))
        if proc==1:
            x_train = m_x_train
            x = m_x + np.random.normal(0, sig_xp, 1)
            x_train[:,0] = m_x_train[:,0] + np.random.normal(0, sig_xp, len(m_x_train))
            x_train[:,1] = m_x_train[:,1] + np.random.normal(0, sig_xp, len(m_x_train))
        y_j = m_model.predict(x.reshape(1,-1), return_std = False)
        bar.append(y_j)
        y_j_train = m_model.predict(x_train, return_std = False)
        bar_train.append(y_j_train)
    bar_arr = np.array(bar)
    bar_train_arr = np.array(bar_train)
    sig_al = bar_arr.std(axis = 0)
    sig_al_train = bar_train_arr.std(axis = 0)
    means, sig_ep = m_model.predict(m_x.reshape(1,-1), return_std = True)
    scaler = MinMaxScaler()
    scaler.fit(sig_al_train.mean(axis=1)[:,None])
    sig_ep = scaler.transform(sig_ep[:,None])
    means_train, _ = m_model.predict(m_x_train, return_std = True)
    k = target.shape[-1]
    gamma2 = sig_ep/2
    nc = ((target-means)**2).sum(axis=1) / (gamma2**2) 
    E_min = ((means_train-target)**2 + sig_al_train**2).min()
    e_min = ((E_min - xi - sig_al**2) / (sig_ep**2)).min()
    improvement = e_min*ncx2.cdf(e_min, k, nc) - ncx2.cdf(e_min, k+2, nc) + nc* ncx2.cdf(e_min, k+4, nc)
    return -improvement.mean()  

#%% joint ACQ
def robust_L2_EI_JOINT(m_x, sig_xp, m_x_train, m_y_train, sig_y_train, m_model, target, xi, num_MC):
    m_model.fit(m_x_train, m_y_train)
    bar = []
    bar_train = []
    for i in range(0,num_MC):
        x = m_x
        x_train = m_x_train
        x[0] = m_x[0]+ np.random.normal(0, sig_xp, 1)
        x[1] = m_x[1]+ np.random.normal(0, sig_xp, 1)
        x[2] = m_x[2]+ np.random.normal(0, sig_xp, 1)
        x_train[:,0] = m_x_train[:,0] + np.random.normal(0, sig_xp, len(m_x_train))
        x_train[:,1] = m_x_train[:,1] + np.random.normal(0, sig_xp, len(m_x_train))
        x_train[:,2] = m_x_train[:,2] + np.random.normal(0, sig_xp, len(m_x_train))
        y_j = m_model.predict(x.reshape(1,-1), return_std = False)
        bar.append(y_j)
        y_j_train = m_model.predict(x_train, return_std = False)
        bar_train.append(y_j_train)
    bar_arr = np.array(bar)
    bar_train_arr = np.array(bar_train)
    sig_al = bar_arr.std(axis = 0)
    sig_al_train = bar_train_arr.std(axis = 0)
    means, sig_ep = m_model.predict(m_x.reshape(1,-1), return_std = True)
    scaler = MinMaxScaler()
    scaler.fit(sig_al_train.mean(axis=1)[:,None])
    sig_ep = scaler.transform(sig_ep[:,None])
    means_train, _ = m_model.predict(m_x_train, return_std = True)
    k = target.shape[-1]
    gamma2 = sig_ep/2
    nc = ((target-means)**2).sum(axis=1) / (gamma2**2) 
    E_min = ((means_train-target)**2 + sig_al_train**2).min()
    e_min = ((E_min - xi - sig_al**2) / (sig_ep**2)).min()
    improvement = e_min*ncx2.cdf(e_min, k, nc) - ncx2.cdf(e_min, k+2, nc) + nc* ncx2.cdf(e_min, k+4, nc)
    return -improvement.mean()
