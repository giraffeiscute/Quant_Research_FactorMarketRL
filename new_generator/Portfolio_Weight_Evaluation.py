# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

class DataEval:
    def __init__(self,data_weights,data_beta):
        self.data=data_weights
        if data_beta.ndim==1:
            data_beta=data_beta[:,np.newaxis]
        self.beta=data_beta        
        self.num_stock,self.num_factor=np.shape(self.beta)
        self.num_time=np.shape(self.data)[0]        
        self.weights=self.data[:,:-1]
        self.cash=self.data[:,-1]        
        self.delta=np.diff(self.weights,axis=0)
        self.turnover_time=np.sum(np.abs(self.delta),axis=1)        
        self.weight_avg=np.mean(self.weights,axis=0)
        self.turnover_avg=np.mean(np.abs(self.delta),axis=0)
        self.turnover_std=np.std(np.abs(self.delta),axis=0)        
        self.market_bias=np.mean(self.weight_avg)
        self.cash_avg=np.mean(self.cash)
        self.HHI=np.sum(self.weight_avg**2)        
        self.weights_corr=np.zeros((self.num_factor))
        self.turnover_corr=np.zeros((self.num_factor))        
        for idx in range(self.num_factor):
            beta_idx=self.beta[:,idx]
            self.weights_corr[idx]=np.corrcoef(self.weight_avg,beta_idx)[0,1]
            self.turnover_corr[idx]=np.corrcoef(self.turnover_avg,beta_idx)[0,1]       
        Cov_ss=np.cov(self.weights.T)
        eig_vals,eig_vecs=np.linalg.eigh(Cov_ss)
        eig_vals=np.maximum(eig_vals,1e-12)        
        self.eig_vals=eig_vals[::-1]
        self.mode_strength=self.eig_vals[0]/np.sum(self.eig_vals)        
        prob=self.eig_vals/np.sum(self.eig_vals)
        entropy=-np.sum(prob*np.log(prob))
        self.effective_rank=np.exp(entropy)        
        X=self.beta
        y=self.weight_avg
        coef=np.linalg.lstsq(X,y,rcond=None)[0]
        y_pred=X@coef
        ss_res=np.sum((y-y_pred)**2)
        ss_tot=np.sum((y-np.mean(y))**2)
        self.factor_R_square=1-ss_res/ss_tot        
    def result_write(self,write_path):
        doc_path=os.path.join(write_path,'weight_data_evaluation.txt')        
        with open(doc_path,'a',encoding='utf-8') as f:
            f.write('='*60+'\n')
            f.write('Portfolio Weight Evaluation\n')
            f.write('='*60+'\n')            
            for idx in range(self.num_factor):
                f.write(f'Factor-{idx} Weight Corr: {self.weights_corr[idx]:.6f}\n')
                f.write(f'Factor-{idx} Turnover Corr: {self.turnover_corr[idx]:.6f}\n')            
            f.write('-'*60+'\n')
            f.write(f'Factor Regression R^2: {self.factor_R_square:.6f}\n')
            f.write(f'Market Bias: {self.market_bias:.6f}\n')
            f.write(f'Cash Average: {self.cash_avg:.6f}\n')
            f.write(f'Concentration HHI: {self.HHI:.6f}\n')
            f.write(f'Dominant Mode Strength: {self.mode_strength:.6f}\n')
            f.write(f'Effective Rank: {self.effective_rank:.6f}\n')
            f.write('-'*60+'\n')    
    def result_plot(self,figure_path,beta_idx):
        beta=self.beta[:,beta_idx]        
        fig=plt.figure(figsize=(14,12))
        fig.suptitle(f'Factor-{beta_idx} Portfolio Analysis',fontsize=16,y=1.02)        
        ax1=plt.subplot(221)
        ax1.scatter(beta,self.weight_avg,alpha=0.5,s=10,label='Data Points')
        ax1.set_title(f'Exposure vs Weights (Corr={self.weights_corr[beta_idx]:.4f})')
        ax1.set_xlabel('Exposure Values')
        ax1.set_ylabel('Time-Averaged Weights')
        ax1.legend()        
        ax2=plt.subplot(222)
        ax2.scatter(beta,self.turnover_avg,alpha=0.5,s=10,label='Data Points')
        ax2.set_title(f'Exposure vs Turnover (Corr={self.turnover_corr[beta_idx]:.4f})')
        ax2.set_xlabel('Exposure Values')
        ax2.set_ylabel('Time-Averaged Absolute Turnover')
        ax2.legend()        
        ax3=plt.subplot(223)
        ax3.plot(self.eig_vals,'o-',alpha=0.7)
        ax3.set_title(f'Eigen Spectrum (EffRank={self.effective_rank:.2f})')
        ax3.set_xlabel('Mode Index')
        ax3.set_ylabel('Eigenvalue')        
        ax4=plt.subplot(224)
        ax4.hist(self.turnover_time,bins=30,density=True,\
                 alpha=0.6,color='g',edgecolor='black')
        mu,sigma=stats.norm.fit(self.turnover_time)
        xmin,xmax=ax4.get_xlim()
        x=np.linspace(xmin,xmax,100)
        p=stats.norm.pdf(x,mu,sigma)
        ax4.plot(x,p,'r--',linewidth=2,label=f'N({mu:.3f}, {sigma:.3f})')
        ax4.set_title('Portfolio Turnover Distribution')
        ax4.set_xlabel('Turnover')
        ax4.set_ylabel('Density')
        ax4.legend()        
        plt.savefig(os.path.join(figure_path,f'{beta_idx}-factor.png'),\
                    bbox_inches='tight')
        plt.show()    
    def result_plot_total(self,figure_path):
        for idx in range(self.num_factor):
            self.result_plot(figure_path,idx)

if __name__ == "__main__":

    eval_dir=r'YourDataDirection'
    save_dir=r'YourDataDirection'
    data_weights=np.load(os.path.join(eval_dir,'model_output.npy'))
    data_beta=np.load(os.path.join(eval_dir,'data_beta.npy'))
    Results=DataEval(data_weights,data_beta)
    Results.result_write(save_dir)
    Results.result_plot_total(save_dir)


















