# -*- coding: utf-8 -*-
#model.py
import numpy as np

class Factor:
    def __init__(self,fac_type,time_scale,time_corr,trend,noisy,is_norm=True,\
                 is_jump=False,jump_prob=None,jump_scale=None,\
                 is_macro_trend=False,macro_trend=None,\
                 is_stochastic_noisy=False,noisy_time_corr=None,noisy_noisy=None,\
                 rng=None,seed=None):
        self.fac_type=fac_type
        self.time_scale=time_scale
        self.burn_in=int(5.0/(1-time_corr))
        self.time_total=self.time_scale+self.burn_in        
        self.time_corr=time_corr
        self.trend=trend
        self.noisy=noisy
        self.is_norm=is_norm       
        self.is_jump=is_jump
        if self.is_jump==True:
            self.jump_prob=jump_prob
            self.jump_scale=jump_scale            
        self.is_macro_trend=is_macro_trend
        if self.is_macro_trend==True:
            self.macro_trend=macro_trend           
        self.is_stochastic_noisy=is_stochastic_noisy
        if self.is_stochastic_noisy==True:
            self.noisy_time_corr=noisy_time_corr
            self.noisy_noisy=noisy_noisy
            self.noisy_init=noisy
            self.noisy_sequ=[noisy]
        if rng is None:
            self.rng=np.random.default_rng(seed=seed)
        else:
            self.rng=rng
    def noisy_update(self):
        temp=self.noisy_time_corr*(np.log(self.noisy**2)-np.log(self.noisy_init**2))\
            +np.log(self.noisy_init**2)+self.noisy_noisy*self.rng.standard_normal()
        self.noisy=np.sqrt(np.exp(temp))
        self.noisy_sequ.append(self.noisy)
    def time_sequ(self):
        sequ_init=np.zeros((self.time_total))
        sequ_init[0]=0
        for i in range(self.time_total-1):
            new_data=self.time_corr*sequ_init[i]+self.trend*(1-self.time_corr)\
                    +self.noisy*np.sqrt(1-self.time_corr**2)*self.rng.standard_normal()
            if self.is_stochastic_noisy==True:
                self.noisy_update()
            if self.is_jump==True and self.rng.random()<self.jump_prob:
                jump_val=self.rng.normal(0,self.jump_scale)
                new_data=new_data+jump_val
            sequ_init[i+1]=new_data
        self.sequ=sequ_init[self.burn_in:]
        if self.is_stochastic_noisy==True:
            self.noisy_sequ=self.noisy_sequ[self.burn_in:]
        if self.is_norm==True:
            mean_val=np.mean(self.sequ)
            std_val=np.std(self.sequ)+1e-9
            self.sequ=(self.sequ-mean_val)/std_val
        if self.is_macro_trend==True:
            Trend=self.macro_trend*np.linspace(1/self.time_scale,1,num=self.time_scale)
            self.sequ=self.sequ+Trend
class Exposure:
    def __init__(self,num_stock,avg,fluct,is_sector=False,sec_pct=None,rng=None,seed=None):
        self.num_stock=num_stock
        self.avg=avg
        self.fluct=fluct
        self.is_sector=is_sector
        if self.is_sector==True:
            self.sec_pct=sec_pct
        if rng is None:
            self.rng=np.random.default_rng(seed=seed)
        else:
            self.rng=rng
    def initial(self):
        if self.is_sector==False:
            self.static=self.rng.normal(self.avg,self.fluct,self.num_stock)
        else:
            in_sec_num=int(self.num_stock*self.sec_pct)
            distri=self.rng.normal(self.avg,self.fluct,in_sec_num)
            self.sec_index=self.rng.choice(self.num_stock,in_sec_num,replace=False)
            self.static=np.zeros(self.num_stock)
            self.static[self.sec_index]=distri
    def dynamics(self,time_scale,time_corr,noisy,is_LowRank=False,LowRank_dim=None,LowRank_fluc=None):
        self.time_scale=time_scale
        self.time_corr=time_corr
        self.noisy=noisy
        self.is_LowRank=is_LowRank
        if self.is_LowRank==True:
            self.LowRank_dim=LowRank_dim
            self.LowRank_fluc=LowRank_fluc 
        self.sequ=np.zeros((self.time_scale,self.num_stock))
        self.sequ[0]=np.copy(self.static)
        if self.is_LowRank==True:
            if self.is_sector==True:
                U,_=np.linalg.qr(self.rng.normal(0,1,(len(self.sec_index),self.LowRank_dim)))
                Up,_=np.linalg.qr(self.rng.normal(0,1,(len(self.sec_index),self.LowRank_dim)))
            else:
                U,_=np.linalg.qr(self.rng.normal(0,1,(self.num_stock,self.LowRank_dim)))
                Up,_=np.linalg.qr(self.rng.normal(0,1,(self.num_stock,self.LowRank_dim)))
        for i in range(self.time_scale-1):
            old_distri=self.sequ[i]
            if self.is_sector==True:
                old_distri=np.copy(old_distri[self.sec_index])
            new_distri=self.time_corr*old_distri\
                      +self.noisy*np.sqrt(1-time_corr**2)*self.rng.normal(0,1,len(old_distri))
            if self.is_LowRank==True:
                LowRank=self.LowRank_fluc*np.einsum('ij,mj,m->i',U,U,old_distri)\
                       +self.noisy*np.einsum('ij,j->i',Up,self.rng.normal(0,1,self.LowRank_dim))
                new_distri=new_distri+LowRank
            mean_val=np.mean(new_distri)
            std_val=np.std(new_distri)+1e-9
            new_distri=self.fluct*(new_distri-mean_val)/std_val+self.avg
            if self.is_sector==True:
                new_distri_temp=np.copy(new_distri)
                new_distri=np.zeros(self.num_stock)
                new_distri[self.sec_index]=new_distri_temp
            self.sequ[i+1]=new_distri
class Observable:
    def __init__(self,factor_list,exposure_list,weight_list,obs_type,\
                 is_exposure_dyna,tau_length=20,tau_bench=10,\
                 rng=None,seed=None):
        self.factor_list=factor_list
        self.exposure_list=exposure_list
        self.weight_list=weight_list
        self.obs_type=obs_type
        self.is_exposure_dyna=is_exposure_dyna
        self.time_scale=self.factor_list[0].time_scale
        self.num_stock=self.exposure_list[0].num_stock
        if self.obs_type=='Convo':
            self.tau_length=tau_length
            self.tau_bench=tau_bench
        if rng is None:
            self.rng=np.random.default_rng(seed=seed)
        else:
            self.rng=rng

        if self.obs_type=="Noisy":
            self.sequ=self.rng.random((self.time_scale,self.num_stock))
        else:
            self.sequ=np.zeros((self.time_scale,self.num_stock))
            if self.is_exposure_dyna==False:
                for index in range(len(self.factor_list)):
                    if self.obs_type=='Linea':
                        step=np.einsum('t,s->ts',self.factor_list[index].sequ,\
                                                 self.exposure_list[index].static)
                        self.sequ=self.sequ+weight_list[index]*np.copy(step)
                    elif self.obs_type=='Convo':
                        K_tau=np.exp(np.linspace(0,-self.tau_length/self.tau_bench,self.tau_length+1))
                        K_tau=K_tau/np.sum(K_tau)
                        beta=self.exposure_list[index].static
                        F=self.factor_list[index].sequ
                        Fp=np.concatenate([np.zeros(self.tau_length),F])\
                            [np.arange(self.time_scale)[:,None]\
                            -np.arange(self.tau_length+1)[None,:]+(self.tau_length)]
                        step=np.einsum('p,s,tp->ts',K_tau,beta,Fp)
                        self.sequ=self.sequ+weight_list[index]*np.copy(step)
                    elif self.obs_type=='Satur':
                        step=np.einsum('t,s->ts',self.factor_list[index].sequ,\
                                                 self.exposure_list[index].static)
                        self.sequ=self.sequ+weight_list[index]*np.tanh(step)
                    elif self.obs_type=='Squar':
                        step=np.einsum('t,s->ts',self.factor_list[index].sequ,\
                                                 self.exposure_list[index].static)
                        self.sequ=self.sequ+weight_list[index]*step**2
                    else:
                        print('Wrong type')
                        break
            else:
                for index in range(len(self.factor_list)):
                    if self.obs_type=='Linea':
                        step=np.einsum('t,ts->ts',self.factor_list[index].sequ,\
                                                  self.exposure_list[index].sequ)
                        self.sequ=self.sequ+weight_list[index]*np.copy(step)
                    elif self.obs_type=='Convo':
                        K_tau=np.exp(np.linspace(0,-self.tau_length/self.tau_bench,self.tau_length+1))
                        K_tau=K_tau/np.sum(K_tau)
                        beta=self.exposure_list[index].sequ
                        F=self.factor_list[index].sequ
                        Fp=np.concatenate([np.zeros(self.tau_length),F])\
                            [np.arange(self.time_scale)[:,None]\
                            -np.arange(self.tau_length+1)[None,:]+(self.tau_length)]
                        step=np.einsum('p,ts,tp->ts',K_tau,beta,Fp)
                        self.sequ=self.sequ+weight_list[index]*np.copy(step)
                    elif self.obs_type=='Satur':
                        step=np.einsum('t,ts->ts',self.factor_list[index].sequ,\
                                                  self.exposure_list[index].sequ)
                        self.sequ=self.sequ+weight_list[index]*np.tanh(step)
                    elif self.obs_type=='Squar':
                        step=np.einsum('t,ts->ts',self.factor_list[index].sequ,\
                                                  self.exposure_list[index].sequ)
                        self.sequ=self.sequ+weight_list[index]*step**2
                    else:
                        print('Wrong type')
                        break
class Market:
    def __init__(self,num_stock,time_scale,is_exposure_dyna=False,master_seed=None):
        self.num_stock=num_stock
        self.time_scale=time_scale
        self.rng=np.random.default_rng(seed=master_seed)
        self.is_exposure_dyna=is_exposure_dyna
        self.is_factor_finish=False
        self.factor_list=[]
        self.exposure_list=[]
        self.obs_list=[]
    def alpha_design(self,alpha_fluct,alpha_rng='global',alpha_seed=None,\
                      alpha_time_corr=None,alpha_noisy=None,is_LowRank=False,LowRank_dim=None,LowRank_fluc=None):
        if self.is_factor_finish==False:
            if alpha_rng=='global':
                alpha_rng=self.rng
            pseudo_factor=Factor('Global',self.time_scale,0.5,0,0.5)
            pseudo_factor.sequ=np.ones((self.time_scale))
            self.factor_list.append(pseudo_factor)
            exposure=Exposure(self.num_stock,0,alpha_fluct,rng=alpha_rng,seed=alpha_seed)
            exposure.initial()
            if self.is_exposure_dyna==True:
                exposure.dynamics(self.time_scale,alpha_time_corr,alpha_noisy,is_LowRank=is_LowRank,\
                                  LowRank_dim=LowRank_dim,LowRank_fluc=LowRank_fluc)
            self.exposure_list.append(exposure)
        else:
            print('Initialization is Finished.')
    def factor_design(self,fac_type,fac_time_corr,fac_trend,fac_noisy,beta_avg,beta_fluct,\
                      fac_is_norm=True,fac_is_jump=False,fac_jump_prob=None,fac_jump_scale=None,\
                      fac_is_macro_trend=False,fac_macro_trend=None,fac_is_stochastic_noisy=False,\
                      fac_noisy_time_corr=None,fac_noisy_noisy=None,fac_rng='global',fac_seed=None,\
                      beta_is_sector=False,beta_sec_pct=None,beta_rng='global',beta_seed=None,\
                      beta_time_corr=None,beta_noisy=None,is_LowRank=False,LowRank_dim=None,LowRank_fluc=None):
        if self.is_factor_finish==False:
            if fac_rng=='global':
                fac_rng=self.rng
            if beta_rng=='global':
                beta_rng=self.rng
            factor=Factor(fac_type,self.time_scale,fac_time_corr,fac_trend,fac_noisy,is_norm=fac_is_norm,\
                          is_jump=fac_is_jump,jump_prob=fac_jump_prob,jump_scale=fac_jump_scale,\
                          is_macro_trend=fac_is_macro_trend,macro_trend=fac_macro_trend,\
                          is_stochastic_noisy=fac_is_stochastic_noisy,noisy_time_corr=fac_noisy_time_corr,\
                          noisy_noisy=fac_noisy_noisy,rng=fac_rng,seed=fac_seed)
            factor.time_sequ()
            self.factor_list.append(factor)
            exposure=Exposure(self.num_stock,beta_avg,beta_fluct,is_sector=beta_is_sector,\
                              sec_pct=beta_sec_pct,rng=beta_rng,seed=beta_seed)
            exposure.initial()
            if self.is_exposure_dyna==True:
                exposure.dynamics(self.time_scale,beta_time_corr,beta_noisy,is_LowRank=is_LowRank,\
                                  LowRank_dim=LowRank_dim,LowRank_fluc=LowRank_fluc)
            self.exposure_list.append(exposure)
        else:
            print('Initialization is Finished.')
    def vector_build(self,g,s,noisy,dim):
        g_norm=g/np.linalg.norm(g)
        g_noisy=self.rng.normal(0,1,dim)
        g_noisy=noisy*g_noisy/np.linalg.norm(g_noisy)
        g=(g_norm+g_noisy)/np.linalg.norm(g_norm+g_noisy)
        vec=(s+noisy*self.rng.normal(0,1))*g
        return vec 
    def LatentBuild(self,dim_latent=3,s_Global=3.0,s_Style=2.0,s_Sector=1.0,\
                    g_Global=np.array([1.0,0.3,0.5]),\
                    g_Style=np.array([0.5,1.0,0.4]),\
                    g_Sector=np.array([0.2,0.4,1.0]),noisy=0.1):
        self.is_factor_finish=True
        self.num_factor=len(self.factor_list)
        self.dim_latent=dim_latent
        self.Latent=np.zeros((self.num_factor,self.dim_latent))
        for index in range(len(self.factor_list)):
            factor=self.factor_list[index]
            if factor.fac_type=='Global':
                vec=self.vector_build(g_Global,s_Global,noisy,self.dim_latent)
            elif factor.fac_type=='Style':
                vec=self.vector_build(g_Style,s_Style,noisy,self.dim_latent)
            else:
                vec=self.vector_build(g_Sector,s_Sector,noisy,self.dim_latent)
            self.Latent[index]=np.copy(vec)
        self.ulist=[]
    def obs_build(self,obs_type,u=None,tau_length=20,tau_bench=10,rng='global',seed=None):
        if self.is_factor_finish==False:
            print('Latent Space has not been Built.')
        else:
            if rng=='global':
                rng_local=self.rng
            else:
                rng_local=np.random.default_rng(seed=seed)
            if u is None:
                vec=rng_local.normal(0,1,self.dim_latent)
                u=vec/np.linalg.norm(vec)
            self.ulist.append(u)
            weight_list=self.Latent@u
            obs=Observable(self.factor_list,self.exposure_list,weight_list,\
                           obs_type,self.is_exposure_dyna,tau_length=tau_length,tau_bench=tau_bench,\
                           rng=rng_local)
            self.obs_list.append(obs)