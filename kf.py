#!/usr/bin/env python
##############################################################
# Author:    Yasuko Matsubara 
# Email:     yasuko@sanken.osaka-u.ac.jp
# URL:       https://www.dm.sanken.osaka-u.ac.jp/~yasuko/
# Date:      2020-01-01
#------------------------------------------------------------#
# Copyright (C) 2020 Yasuko Matsubara & Yasushi Sakurai
# OrbitMap is freely available for non-commercial purposes
##############################################################
import numpy as np
import tool as tl
import pylab as pl
import sys
from pykalman import KalmanFilter
import pickle
try:
    from pykalman import KalmanFilter
except:
    tl.error("can not find pykalman - please see https://pykalman.github.io/")

#-----------------------------#
ITERf=10 #2
ITERe=20 #10
#-----------------------------#
DBG=tl.NO


class KF:
    def __init__(self, data, k):
        if(DBG): tl.comment("KF (Kalman Filter)")
        (self.n,self.d)=np.shape(data)
        # data (observations)
        self.data=data
        self.k=k
        self.initParams()
    def initParams(self):
        self.kf=_initParams(self)
    def em(self, ITER=ITERf, iter_each=ITERe):
        self.kf=_em(self,ITER)
    def plot(self, fn='', pblock=False):
        _plot(self, fn, pblock)
    def printParams(self, fn=''):
        tl.comment("print/save params [%s]"%(fn))
        _printParams(self, fn)
        tl.save_obj(self, fn)
    def getParams(self):
        return _getParams(self)
    def gen(self, n=-1):
        return _gen(self, n)
    def LH(self, data):
        try: return self.kf.loglikelihood(data)
        except: return -tl.INF

#----------------------------------#
#     private functions 
#----------------------------------#
def _initParams(model):
    d=model.d; n=model.n; k=model.k
    if(DBG): tl.msg("init params")
    if(DBG): tl.msg("d: %d, n: %d, k:%d"%(d, n, k))
    # init KF params
    A=np.random.random((k,k))
    C=np.random.random((d,k))
    A0=np.random.random(k)*tl.ZERO
    C0=np.random.random(d)*tl.ZERO
    # set KF params
    kf = KalmanFilter(
            transition_matrices=A,  transition_offsets=A0,
            observation_matrices=C, observation_offsets=C0,
            #transition_covariance  = np.eye(k)*tl.ZERO, #((k,k)), 
            #observation_covariance = np.eye(d)*tl.ZERO, #((k,k)), 
            #n_dim_state=k, n_dim_obs=d,
            em_vars=[
                'transition_matrices', 
                'transition_covariance', 
                'transition_offsets', 
                'observation_matrices',
                'observation_covariance',
                'observation_offsets', 
                'initial_state_mean', 
                'initial_state_covariance'])
    return kf

def _em(model, ITER=ITERf, iter_each=ITERe):
    observations= tl.dcopy(model.data)
    if(np.sum(np.isnan(observations))>0):
        tl.warning("kf.py --- em: isnan->0")
    observations[np.isnan(observations)]=0
    observations+=tl.ZERO*np.random.rand(model.n, model.d)
    kf=model.kf
    if(DBG): tl.msg("kf.em: d=%d, n=%d, k=%d"%(model.d, model.n, model.k))
    if(DBG): tl.msg("start EM algorithm ...")
    lhs= np.zeros(ITER)
    for i in range(len(lhs)):
        kf = kf.em(X=observations, n_iter=iter_each)
        lhs[i] = model.LH(observations)
        #tl.msg("iter:%d/%d, LH= %f"%(i+1,ITER,lhs[i])) 
        if(DBG): tl.msg("iter:%d/%d (%d)"%(i+1,ITER, iter_each)) 
        if(i>1 and lhs[i-1]>lhs[i]):
            break
    if(DBG): tl.msg("end")
    return kf

def _filter(model):
    kf=model.kf
    observations=model.data
    # Estimate the hidden states using observations up to and including
    # time t for t in [0...n_timesteps-1].  This method outputs the mean and
    # covariance characterizing the Multivariate Normal distribution for
    #   P(x_t | z_{1:t})
    f_state = kf.filter(observations)[0]
    f_obs=np.dot(kf.observation_matrices, f_state.T).T
    return (f_state, f_obs)

def _smoother(model):
    kf=model.kf
    observations=model.data
    # Estimate the hidden states using all observations.  These estimates
    # will be 'smoother' (and are to be preferred) to those produced by
    # simply filtering as they are made with later observations in mind.
    # Probabilistically, this method produces the mean and covariance
    # characterizing,
    #    P(x_t | z_{1:n_timesteps})
    s_state = kf.smooth(observations)[0]
    s_obs=np.dot(kf.observation_matrices, s_state.T).T
    return (s_state, s_obs)


def _gen(model,n=-1):
    kf=model.kf
    if(n==-1): n=model.n 
    k=model.k
    # Estimate the state without using any observations.  This will let us see how
    # good we could do if we ran blind.
    g_state=np.zeros((n, k))
    for t in range(n - 1):
        if t == 0:
            g_state[t] = kf.initial_state_mean
        g_state[t + 1] = (
        np.dot(kf.transition_matrices, g_state[t])
        + kf.transition_offsets# %[t]
        )
    g_obs=np.dot(kf.observation_matrices, g_state.T).T+kf.observation_offsets
    return (g_state, g_obs)

def _plot(model, fn, pblock):
    observations=model.data
    fig=pl.figure(figsize=(8, 6))
    pl.subplot(311)
    pl_obs=pl.plot(observations, '-', label='org')
    pl.title("Original sequence")
    #
    (g_state, g_obs)=_gen(model)
    (f_state, f_obs)=_filter(model)
    #(s_state, s_obs)=_smoother(model)
    #
    pl.subplot(312)
    pl_sta=pl.plot(f_state, label='states(filter)')
    pl.title("States (filter)")
    pl.subplot(313)
    #pl_obs=pl.plot(observations, '--', label='org')
    #tl.resetCol()
    pl_gen=pl.plot(f_obs, '-', label='gen(filter)')
    #pl.title("Generated (filter)")
    pl.title("Generated (filter), rmse=%f"%(tl.RMSE(observations, f_obs)))
    #
    #pl.xlabel('time')
    fig.set_tight_layout(True)
    if(fn!=''): tl.savefig(fn+'_fit','pdf')
    pl.show(block=pblock)
 

def _getParams(model):
    A=model.kf.transition_matrices
    b=model.kf.transition_offsets
    C=model.kf.observation_matrices
    d=model.kf.observation_offsets
    Q=model.kf.transition_covariance
    R=model.kf.observation_covariance
    sm0=model.kf.initial_state_mean
    sv0=model.kf.initial_state_covariance
    return (A,b,C,d,Q,R,sm0,sv0)

def _printParams(model, fn):
    if(fn!=''):
        f = open(fn+'_base', 'w')
        sys.stdout = f
    print("d= %d" %model.d)
    print("n= %d" %model.n)
    print("k= %d" %model.k)
    # full parameter set
    (A,b,C,d,Q,R,sm0,sv0)=model.getParams()
    print("A (trans):")
    tl.printM(A)
    print("Q (trans_v):")
    tl.printM(Q)
    print("C (obs):")
    tl.printM(C)
    print("R (obs_v):")
    tl.printM(R)
    print("sm0 (init_m):")
    tl.printM(sm0)
    print("sv0 (init_v):")
    tl.printM(sv0)
    if(fn!=''):
        sys.stdout = sys.__stdout__
        f.close()


def _example1():
    # create observations
    n=300
    d=1
    x = np.linspace(0, 3 * np.pi, n)
    data=np.zeros((n,d))
    observations = 20 * (np.sin(x) + 0.005 * np.random.random(n))
    data[:,0]=observations
    return (data) 
 
#----------------------------------#
#     main 
#----------------------------------#
if __name__ == "__main__":
    from kf import KF
    tl.comment("kf.py -- example ")
    tl.msg("create synthetic sequence (1-dim)")
    (data)=_example1()
    fn='../output/tmp/kf_sample'
    
   
    k=2 #4 
    tl.msg("k=%d: # of latent variables"%(k)) 
    tl.msg("init params")
    mykf=KF(data,k)
    tl.msg("em algorithm -- START")
    tic = tl.time.clock()
    mykf.em(ITER=2, iter_each=10)
    toc = tl.time.clock(); fittime= toc-tic;
    tl.msg("em algorithm -- END (%f sec.)"%(fittime))
    tl.msg("plot/save models")
    mykf.plot(fn)
    mykf.printParams(fn)



