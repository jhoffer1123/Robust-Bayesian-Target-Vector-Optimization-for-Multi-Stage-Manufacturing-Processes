# -*- coding: utf-8 -*-
"""
Created on Mon May 15 06:53:57 2023

@author: J. G. Hoffer
"""


import numpy as np
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import ncx2


def robust_L2_EI(m_x, sig_xp, m_x_train, sig_x_train, m_y_train, m_model, sig_model, target, xi, num_MC):
    
    #%% fit surrogate model
    m_model.fit(m_x_train, m_y_train)
    
    #%% Monte Carlo uncertainty propagation
    bar = []
    sig = []
    bar_train = []
    sig_train = []
    
    sig_x = sig_xp
    for _ in range(0,num_MC):
        x = m_x + np.random.normal(0, sig_x, 1)
        x_train = m_x_train.ravel() + np.random.normal(0, sig_x_train.ravel(), len(m_x_train))
        #%% surrogate model predicitons for uncertainty propagation
        #possible candidates
        y_j = m_model.predict(x[:,None], return_std = False)
        bar.append(y_j)
        # training data
        y_j_train = m_model.predict(x_train[:,None], return_std = False)
        bar_train.append(y_j_train)
    #%% making arrays  
    bar_arr = np.array(bar)
    bar_train_arr = np.array(bar_train)
    #%% calculation of mean and std, i.e., aleatoric uncertainty
    sig_al = bar_arr.std(axis = 0)
    
    
    
    sig_al_train = bar_train_arr.std(axis = 0)
    means, sig_ep = m_model.predict(m_x[:,None], return_std = True)
    #print(sig_al_train.mean(axis=1))
    scaler = MinMaxScaler(
        #feature_range=(sig_al_train.min(), sig_al_train.max())
        )
    scaler.fit(sig_al_train.mean(axis=1)[:,None])
    sig_ep = scaler.transform(sig_ep[:,None])
    
    means_train, _ = m_model.predict(m_x_train, return_std = True)
    k = target.shape[-1]
    gamma2 = sig_ep/2
    #%%    
    # nc.... lambda = noncentrality parameter
    nc = ((target-means)**2).sum(axis=1) / (gamma2**2) 
    #%%
    E_min = ((means_train-target)**2 + sig_al_train**2).min()
    e_min = ((E_min - xi - sig_al**2) / (sig_ep**2)).min()
    #%%
    improvement = e_min*ncx2.cdf(e_min, k, nc) - ncx2.cdf(e_min, k+2, nc) + nc* ncx2.cdf(e_min, k+4, nc)

    return -improvement.mean()



#%%

def L2_EI(m_x, sig_xp, m_x_train, sig_x_train, m_y_train, m_model, sig_model, target, xi, num_MC):
    
    #%% fit surrogate model
    m_model.fit(m_x_train, m_y_train)
    #%% calculation of mean and std, i.e., aleatoric uncertainty
    means, sig_ep = m_model.predict(m_x[:,None], return_std = True)
    means_train, _ = m_model.predict(m_x_train, return_std = True)
    k = target.shape[-1]
    gamma2 = sig_ep/2
    #%%    
    # nc.... lambda = noncentrality parameter
    nc = ((target-means)**2).sum(axis=1) / (gamma2**2) 
    #%%
    E_min = ((means_train-target)**2 ).min()
    e_min = ((E_min - xi ) / (sig_ep**2)).min()
    #%% improvement
    improvement = e_min*ncx2.cdf(e_min, k, nc) - ncx2.cdf(e_min, k+2, nc) + nc* ncx2.cdf(e_min, k+4, nc)

    return -improvement.mean()





























