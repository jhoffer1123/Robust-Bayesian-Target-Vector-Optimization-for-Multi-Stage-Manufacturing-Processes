# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:19:01 2023

@author: J. G. Hoffer
"""

import numpy as np
from scipy.optimize import direct
from Acquisition_function_UC2 import robust_L2_EI_CASC
from Acquisition_function_UC2 import robust_L2_EI_JOINT

def BO_robust_cascaded_L2_EI(num_BO, sig, m_x2_train, sig_y1_train, m_y2_train, sig_y2_train, m_model2, sig_model2, target22_, xi, num_MC, bound1, 
                          bound2, m_x1_train, m_y1_train, m_model1, sig_model1,
                          proc1_unc, proc2_unc, target22,
                          ):
    m_x1_mins = []
    m_y1_mins = []
    sig_y1_mins = []
    m_x2_mins = []
    m_y2_mins = []
    sig_y2_mins = []
    E_minss = []
    for bo in range(0,num_BO):
        print('iteration: '+str(bo))
        sig_xp=np.array([sig])
        res2 = direct(robust_L2_EI_CASC, bounds = bound2, args=[sig_xp, m_x2_train, sig_y1_train, m_y2_train, sig_y2_train, m_model2, sig_model2, target22_, xi, num_MC, 2]
                      , maxiter = 100
                      )
        target_proc1 = np.array([res2.x[1]])
        res1 = direct(robust_L2_EI_CASC, bounds = bound1, args=[sig_xp, m_x1_train, None, m_y1_train, sig_y1_train, m_model1, sig_model1, target_proc1, xi, num_MC, 1]
                      , maxiter = 100
                      )
        m_x1_next = res1.x.reshape(-1,1)
        m_y1_next, sig_y1_next = proc1_unc(m_x1_next[0], m_x1_next[1], sig)
        m_x22_next = m_y1_next
        m_x21_next = np.array([res2.x[0]])
        m_y21_next, m_y22_next, sig_y21_next, sig_y22_next = proc2_unc(m_x21_next, m_x22_next, sig = sig_y1_next.ravel())
        m_x2_next = np.concatenate((m_x21_next.reshape(-1,1), m_x22_next), axis = 1)
        m_y2_next = np.concatenate((m_y21_next, m_y22_next), axis = 1)
        m_x1_next = np.concatenate((m_x1_next[0].reshape(-1,1), m_x1_next[1].reshape(-1,1)), axis = 1)
        m_x2_train = np.concatenate((m_x2_train, m_x2_next))
        m_y2_train = np.concatenate((m_y2_train, m_y2_next))
        m_x1_train = np.concatenate((m_x1_train, m_x1_next))
        m_y1_train = np.concatenate((m_y1_train, m_y1_next))
        sig_y1_train = np.concatenate((sig_y1_train, sig_y1_next))
        sig_y2_train = np.concatenate((sig_y2_train, np.concatenate((sig_y21_next, sig_y22_next), axis = 1)))
        E_min = ((m_y2_train-target22)**2 + sig_y2_train**2).mean(axis = 1)
        E_argmin = E_min.argmin()
        m_x1_mins.append(m_x1_train[E_argmin])
        m_y1_mins.append(m_y1_train[E_argmin])
        sig_y1_mins.append(sig_y1_train[E_argmin])
        m_x2_mins.append(m_x2_train[E_argmin])
        m_y2_mins.append(m_y2_train[E_argmin])
        sig_y2_mins.append(sig_y2_train[E_argmin])
        E_minss.append(E_min.min())
    return E_minss

#%%

def BO_robust_joint_L2_EI(num_BO, bound, sig, m_x_train, m_y_train, sig_y_train, m_model, target22_, xi, num_MC, proc1_unc, proc2_unc, target22,
                          ):
    E_minss = []
    m_x_mins = []
    m_y_mins = []
    sig_y_mins = []
    for bo in range(0,num_BO):
        print('iteration: '+str(bo))
        res = direct(robust_L2_EI_JOINT, bounds = bound, args=[sig, m_x_train, m_y_train, sig_y_train, m_model, target22_, xi, num_MC]
                     , maxiter = 100
                     )
        m_x_next = res.x.reshape(-1,1)
        m_y1_next, sig_y1_next = proc1_unc(m_x_next[0], m_x_next[1], sig)
        m_x22_next = m_y1_next
        m_y21_next, m_y22_next, sig_y21_next, sig_y22_next = proc2_unc(m_x_next[2], m_x22_next, sig=sig_y1_next.ravel())
        m_x_train = np.concatenate((m_x_train, m_x_next.reshape(1,-1)))
        m_y_next = np.concatenate((m_y21_next, m_y22_next), axis = 1)
        sig_y_next = np.concatenate((sig_y21_next, sig_y22_next), axis = 1)
        m_y_train = np.concatenate((m_y_train,m_y_next))
        sig_y_train = np.concatenate((sig_y_train, sig_y_next))
        E_min = ((m_y_train-target22)**2 + sig_y_train**2).mean(axis = 1)
        E_argmin = E_min.argmin()
        m_x_mins.append(m_x_train[E_argmin])
        m_y_mins.append(m_y_train[E_argmin])
        sig_y_mins.append(sig_y_train[E_argmin])
        E_minss.append(E_min.min())
    return E_minss




















