# -*- coding: utf-8 -*-
#main.py
import numpy as np
from model import Market
from para import price_sequ
from config import *
import pandas as pd
import os

market=Market(num_stock,time_scale,is_exposure_dyna=False,master_seed=master_seed)
if is_alpha:
    market.alpha_design(alpha_fluct)
for f in factor_list:
    market.factor_design(fac_type=f['type'],fac_time_corr=f['time_corr'],fac_trend=f['trend'],
                         fac_noisy=f['noisy'],beta_avg=f['beta_avg'],beta_fluct=f['beta_fluct'],
                         beta_is_sector=f['is_sec'],beta_sec_pct=f['sec_pct'])
market.LatentBuild()
for obs in obs_list:
    market.obs_build(obs_type=obs['type'],u=obs['u_vec'])
price=price_sequ(market.obs_list[0].sequ)
data_obs=np.array([price]+[obs.sequ for obs in market.obs_list])
data_factor=np.array([fac.sequ for fac in market.factor_list])
data_beta=np.array([beta.static for beta in market.exposure_list])
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