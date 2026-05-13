# -*- coding: utf-8 -*-
#main.py
import numpy as np
import os
from model import Market
from para import *
import pandas as pd

num_stock=100
time_scale=500
master_seed=42
market=Market(num_stock,time_scale,is_exposure_dyna=False,master_seed=master_seed)
'''
WRITE YOUR FACTORS DESIGN BELOW, LIKE
#time_corr, trend, noisy, beta_avg, beta_fluct

market.factor_design('Style', 0.85,  0.00,  0.05,   0.1,   0.9)
market.factor_design('Style', 0.85,  0.00,  0.05,   0.1,   0.9)
'''
market.factor_design('Style', 0.85,  0.00,  0.05,   0.1,   0.9)
market.LatentBuild()
'''
WRITE YOUR OBSERVABLES DESIGN BELOW, LIKE
#time_corr, trend, noisy, beta_avg, beta_fluct
market.obs_build('Linea',u=u_return)

'''
market.obs_build('Linea',u=u_return)

data_obs=np.array([obs.sequ for obs in market.obs_list])
data_factor=np.array([fac.sequ for fac in market.factor_list])
data_beta=np.array([beta.static for beta in market.exposure_list])

fold_dir=r'YOUR DIRECTION'
os.makedirs(fold_dir,exist_ok=True)
def data_3D_parquet_save(data,fold_dir):
    num_features,time_scale,num_stock=data.shape
    total_rows=time_scale*num_stock
    result=np.zeros((total_rows,2+num_features),dtype=float)
    col_idx=np.repeat(np.arange(num_stock),time_scale)
    row_idx=np.tile(np.arange(time_scale),num_stock)    
    result[:,0]=col_idx
    result[:,1]=row_idx
    data_reshaped=data.reshape(num_features,-1).T
    result[:,2:]=data_reshaped
    columns=['stock_index','time_index']+[f'feature_{i}' for i in range(num_features)]
    df=pd.DataFrame(result, columns=columns)
    df.to_parquet(os.path.join(fold_dir,'data_obs.parquet'),compression='snappy',index=False)    
    return df
def data_2D_parquet_save(data,fold_dir,name):
    num_features,index=data.shape
    result=np.zeros((index,1+num_features),dtype=float)
    idx=np.arange(index)
    result[:,0]=idx
    data_reshaped=data.reshape(num_features,-1).T
    result[:,1:]=data_reshaped
    columns=['index']+[f'feature_{i}' for i in range(num_features)]
    df=pd.DataFrame(result, columns=columns)
    df.to_parquet(os.path.join(fold_dir,f'data_{name}.parquet'),compression='snappy',index=False)    
    return df
data_3D_parquet_save(data_obs,fold_dir)
data_2D_parquet_save(data_beta,fold_dir,'beta')
data_2D_parquet_save(data_factor,fold_dir,'fac')
np.save(os.path.join(fold_dir,'data_beta.npy'),data_beta.T)