# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:33:24 2023

@author: J. G. Hoffer
"""

import numpy as np
from scipy.optimize import direct, Bounds

def BO_loop(bounds, m_x_train, sig_x_train, m_y_train, sig_y_train, m_model, sig_model, target, xi, ACQ, num_it, num_MC, robust = True):
    E_minss = []
    m_x_mins = []
    m_y_mins = []
    sig_y_mins = []
    for i in range(0,num_it):
        
        print('BO iteration: '+str(i))
        sig_xp=np.array([0.1])
        res = direct(ACQ, bounds = bounds
                     , args = [sig_xp, m_x_train, sig_x_train, m_y_train, m_model, sig_model, target, xi, num_MC]
                     , maxiter = 100
                     )
        
        m_x_next = res.x.reshape(-1,1)
        _, sig_y_next = y(m_x_next, sig)
        m_y_next, _ = y(m_x_next, 0)
        m_x_train = np.concatenate((m_x_train, m_x_next))
        sig_x_train = np.concatenate((sig_x_train, sig_xp[:,None]))
        m_y_train = np.concatenate((m_y_train, m_y_next))
        sig_y_train = np.concatenate((sig_y_train, sig_y_next))
        E_min = ((m_y_train-target)**2 + sig_y_train**2).mean(axis = 1)
        E_min_arg = E_min
        
        # if robust == True:
        #     E_min_arg = E_min
        # if robust == False:
        #     E_min_arg = ((m_y_train-target)**2).mean(axis = 1)

        
        E_argmin = E_min_arg.argmin()
        print(E_argmin)
        
        m_x_mins.append(m_x_train[E_argmin])
        m_y_mins.append(m_y_train[E_argmin])
        sig_y_mins.append(sig_y_train[E_argmin])
        
        E_minss.append(E_min.min())
        
    return m_x_train, m_y_train, E_min, np.array(m_x_mins), np.array(m_y_mins), np.array(sig_y_mins), np.array(E_minss)
