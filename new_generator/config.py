# -*- coding: utf-8 -*-
import numpy as np
from para import *
import os

fold_dir=r'YOUR DIRECTION'
os.makedirs(fold_dir,exist_ok=True)

num_stock=100
time_scale=500
master_seed=42

is_alpha=True
alpha_fluct=0.5

factor_list=[]
f={'type':'Global','time_corr':0.995,'trend':0.02,'noisy':0.01,\
   'beta_avg':0.9,'beta_fluct':0.05,'is_sec':False,'sec_pct':None}
factor_list.append(f)
f={'type':'Style','time_corr':0.8,'trend':0.01,'noisy':0.05,\
   'beta_avg':0.1,'beta_fluct':0.9,'is_sec':False,'sec_pct':None}
factor_list.append(f)
f={'type':'Sector','time_corr':0.7,'trend':-0.01,'noisy':0.2,\
   'beta_avg':0.8,'beta_fluct':0.3,'is_sec':True,'sec_pct':0.2}
factor_list.append(f)

#'Linea','Convo','Squar','Satur',the first one is return
obs_list=[]
obs={'type':'Linea','u_vec':u_return}
obs_list.append(obs)
obs={'type':'Convo','u_vec':u_moment}
obs_list.append(obs)
obs={'type':'Linea','u_vec':None}
obs_list.append(obs)