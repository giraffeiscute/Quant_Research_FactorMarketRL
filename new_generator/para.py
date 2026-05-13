# -*- coding: utf-8 -*-
#para.py
'''
FACTOR PARA
Global:
    time_corr in [0.95,0.995]
    trend in [-0.02,0.02]
    noisy in [0.01,0.05]
    macro_trend controlled by std(macro_trend) in [0.1,0.5]
Style:
    time_corr in [0.7,0.95]
    noisy in [0.02,0.1]
Sector:
    time_corr in [0.5,0.9]
    noisy in [0.05,0.2]
    jump_prob ~ 0.1
    jump_scale ~ Normal(0,0.5)
    
EXPOSURE PARA
Global:
    avg ~ 1.0
    fluct in [0.05,0.2]
    time_corr in [0.99,0.999]
    noisy ~ 0.01
    LowRank_dim ~ 1
    LowRank_fluc ~ 0.01
Style:
    avg ~ 0
    fluct ~ 1
    time_corr in [0.9,0.97]
    noisy in [0.05,0.1]
    LowRank_dim in [2,5]
    LowRank_fluc in [0.05,0.2]
Sector:
    pct in [0.1,0.3]
    avg ~ 1
    fluct in [0.1,0.3]
    time_corr in [0.9,0.98]
    noisy in [0.03,0.08]
    LowRank_dim in [1,2]
    LowRank_fluc in [0.02,0.05]
    
OBSERVABLE PARA
Types:
    'Linea': beta(t)F(t)
    'Convo': exp(tau/tau0)beta(t)F(t-tau)
    'Satur': tanh(beta(t)F(t))
    'Squar': (beta(t)F(t))^2
    'Noisy': rng.random((num_stock,time_scale))
Weight(dim_latent=3):
    global factor   [1.0,0.3,0.5]
    style factor    [0.5,1.0,0.4]
    sector factor   [0.2,0.4,1.0]

STOCHASTIC NOISY PARA
Global:
    noisy_time_corr in [0.97,0.995]
    noisy_noisy in [0.05,0.2]
Style:
    noisy_time_corr in [0.9,0.97]
    noisy_noisy in [0.1,0.3]
Sector:
    noisy_time_corr in [0.7,0.9]
    noisy_noisy in [0.2,0.5]
'''
import numpy as np

def price_sequ(retur):#assuming the maximum of return is 0.01 at each time step
    retur=0.1*retur/np.max(np.abs(retur))
    cum_retur=np.cumsum(retur,axis=0)
    price=np.exp(cum_retur)
    return price

u_return=np.array([1.0,0.8,0.2])/np.linalg.norm(np.array([1.0,0.8,0.2])) 
type_return='Linea'
u_volat1=np.array([1.0,0.3,0.3])/np.linalg.norm(np.array([1.0,0.3,0.3]))
type_volat1='Squar'
u_volat2=np.array([1.0,0.2,0.2])/np.linalg.norm(np.array([1.0,0.2,0.2]))
type_volat2='Convo'
tau_volat2=20
u_moment=np.array([0.3,1.0,0.2])/np.linalg.norm(np.array([0.3,1.0,0.2]))
type_moment='Convo'
tau_moment=50
u_secstr=np.array([0.2,0.4,1.0])/np.linalg.norm(np.array([0.2,0.4,1.0]))
type_secstr='Linea'
u_nonlin=np.array([0.6,0.8,0.3])/np.linalg.norm(np.array([0.6,0.8,0.3]))
type_nonlin='Satur'
