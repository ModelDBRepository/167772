# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:44:54 2013

@authors: Patricio Orio & Danilo Pezo
Current clamp simulation of the stochastic Hodgkin & Huxley model
Explicit Markov Chain modeling with a modified Gillespie's algorithm
optimized for few transitions (i.e. few channels or slow channels) 
USAGE:
Mode 1: Run the script as stand-alone. In the last lines (after the
 if __name__=='__main__' line) you can modify the parameters of the call to simulate
Mode 2: Import as module and use simulate

simulate(nsim=30,Tstop=15,dt=0.005,Idel=0,Idur=-1,Iamp=0,NNa=600,NK=180,recording=0)
    nsim: Number of simultaneous simulations to be run
        There are nsim simulations simultaneously being calculated. 
        Thus everything is a vector of length nsim (voltage, rates, etc)
    Tstop: Total time to simulate (ms)
    dt: time step (ms). 
    Idel, Idur, Iamp: Delay, duration and amplitude of currrent stimulus
        if Idur=-1, then Idur=Tstop    (ms, ms, microA/cm2)
    NNa, NK: Numbers of channels to be simulated
    recording: to record or not to record voltage
    
    RETURNS:
    vrec: array with voltage trajectories calculated (shape is (Tstop/dt,nsim))
    tvec: vector of time points
    firetime: array with times of firing for each simulation. 0 for simulations that didn't fire
    NaNs: Number of simulations that ended with a NaN in the voltage
    time: real time spent in the simulation     

"""

import numpy as np
import time as tm
import matplotlib.pyplot as plt

#Model parameters
gK=36  # mS/cm2    default:36
gNa=120  # mS/cm2  default:120
gL=0.3  # ms/cm2   default:0.3

EK=-77    #mV
EL=-54.4  #mV
ENa=50    #mV



threshold=0  #Threshold for action potential detection

#Narat and Krat will return the actual transition rates (for every possible
# transition) at any given time, given the voltage and the number of channels
# in each state

def Narat(v,Nastates):
    am=0.1*(v+40)/(1-np.exp(-(v+40)/10))
    bm=4*np.exp(-(v+65)/18)
    bh=1/(1+np.exp(-(v+35)/10))
    ah=0.07*np.exp(-(v+65)/20)
    return np.array((3*am*Nastates[0,:],
        bm*Nastates[1,:],
        2*am*Nastates[1,:],
        2*bm*Nastates[2,:],
        am*Nastates[2,:],
        3*bm*Nastates[3,:],
        ah*Nastates[0,:],
        bh*Nastates[4,:],
        ah*Nastates[1,:],
        bh*Nastates[5,:],
        ah*Nastates[2,:],
        bh*Nastates[6,:],
        ah*Nastates[3,:],
        bh*Nastates[7,:],
        3*am*Nastates[4,:],
        bm*Nastates[5,:],
        2*am*Nastates[5,:],
        2*bm*Nastates[6,:],
        am*Nastates[6,:],
        3*bm*Nastates[7,:]))

def Krat(v,Kstates):
    an=0.01*(v+55)/(1-np.exp(-(v+55)/10))
    bn=0.125*np.exp(-(v+65)/80)
    return np.array((4*an*Kstates[0,:],
        bn*Kstates[1,:],
        3*an*Kstates[1,:],
        2*bn*Kstates[2,:],
        2*an*Kstates[2,:],
        3*bn*Kstates[3,:],
        an*Kstates[3,:],
        4*bn*Kstates[4,:]))

""" Na_trans represent the 20 possible transitions for Na channels, as follows:
First row trasitions (m particles)
    Forward            Backward
 0: m0h0 --> m1h0   1: m1h0 --> m0h0
 2: m1h0 --> m2h0   3: m2h0 --> m1h0
 4: m2h0 --> m3h0   5: m3h0 --> m2h0
Vertical transitions (h particle)
 6: m0h0 --> m0h1   7: m0h1 --> m0h0
 8: m1h0 --> m1h1   9: m1h1 --> m1h0
10: m2h0 --> m2h1  11: m2h1 --> m2h0
12: m3h0 --> m3h1  13: m3h1 --> m3h0
First row trasitions (m particles)
14: m0h1 --> m1h1  15: m1h1 --> m0h1
16: m1h1 --> m2h1  17: m2h1 --> m1h1
18: m2h1 --> m3h1  19: m3h1 --> m2h1
"""       
Na_trans=np.array([[-1, 1, 0, 0, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, -1, 1, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0],
            [0, -1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, -1, 0, 0],
            [0, 0, -1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0],
            [0, 0, 0, -1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, -1],
            [0, 0, 0, 0, -1, 1, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 0],
            [0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 0, -1, 1],
            [0, 0, 0, 0, 0, 0, 1, -1]]).T

""" K_trans represents the 8 possible transitions for K channels, as follows:
    Forward            Backward
 0: n0 --> n1   1: n1 --> n0
 2: n1 --> n2   3: n2 --> n1
 4: n2 --> n3   5: n3 --> n2
 6: n3 --> n4   7: n4 --> n3
"""

K_trans=np.array([[-1, 1, 0, 0, 0],
        [1, -1, 0, 0, 0],
        [0, -1, 1, 0, 0],
        [0, 1, -1, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, 1, -1, 0],
        [0, 0, 0, -1, 1],
        [0, 0, 0, 1, -1]]).T


def simulate(nsim=30,Tstop=15,dt=0.005,
             Idel=0,Idur=-1,Iamp=0,NNa=600,NK=180,recording=0):
    """
    Stochastic simulation of HH model using Markov Chains with modified Gillespie's method
    nsim: Number of simultaneous simulations to be run
        There are nsim simulations simultaneously being calculated. 
        Thus everything is a vector of length nsim (voltage, rates, etc)
    Tstop: Total time to simulate (ms)
    dt: time step (ms). 
    Idel, Idur, Iamp: Delay, duration and amplitude of currrent stimulus
        if Idur=-1, then Idur=Tstop    (ms, ms, microA/cm2)
    NNa, NK: Numbers of channels to be simulated
    recording: to record or not to record voltage
    
    RETURNS:
    vrec: array with voltage trajectories calculated (shape is (Tstop/dt,nsim))
    tvec: vector of time points
    firetime: array with times of firing for each simulation. 0 for simulations that didn't fire
    NaNs: Number of simulations that ended with a NaN in the voltage
    time: real time spent in the simulation     
    """
    points = np.around(Tstop/dt) #Number of time steps to perform

    if Idur==-1:
        Idur=Tstop
    # Just an auxiliary variable to be used when doing transitions
    xx = np.zeros((nsim),dtype=int)
    
    NNaNs=0; p=0
    t0=tm.time()
    
    if recording:
        vrec=np.zeros((points,nsim))
    
    # calculate the initial conditions at -65 mV, including the initial 
    # distribution of channels in each state (arrays Nastates and Kstates). It's deterministic
    # All of these are vectors of length nsim.
    v=-65*np.ones(nsim)
    an=0.01*(v+55)/(1-np.exp(-(v+55)/10))
    bn=0.125*np.exp(-(v+65)/80)
    am=0.1*(v+40)/(1-np.exp(-(v+40)/10))
    bm=4*np.exp(-(v+65)/18)
    bh=1/(1+np.exp(-(v+35)/10))
    ah=0.07*np.exp(-(v+65)/20)
    M=am/bm
    H=ah/bh
    Nastatesum=(1+H)*(1+M)**3
    m=np.array([np.ones(nsim),3*M,3*M**2,M**3,H,3*M*H,3*M**2*H,M**3*H])/np.outer(np.ones((8,1)),Nastatesum)
    
    Nastates=np.round(m*NNa)
    Nastates[np.argmax(Nastates,axis=0)]+=NNa-np.sum(Nastates,axis=0)            
    
    
    N=an/bn
    Kstatesum=(1+N)**4        
    n=np.array([np.ones(nsim),4*N,6*N**2,4*N**3,N**4])/np.outer(np.ones((5,1)),Kstatesum)
    Kstates=np.round(n*NK)
    Kstates[np.argmax(Kstates,axis=0)]+=NK-np.sum(Kstates,axis=0)
        
    # pick random numbers for the next Na or K transition (next_RNa, next_RK), 
    # and set the time of the previous transition to 0 (prev_ev??).
    # All of these are vectors of length nsim.
    next_RNa=-np.log(np.random.uniform(size=nsim))
    prev_evNa=np.zeros_like(v)    
    next_RK=-np.log(np.random.uniform(size=nsim))
    prev_evK=np.zeros_like(v)
    
    # firetime will store the time of AP (if any) for each simulation
    # firing will tell whether voltage>threshold (an AP is on the way) 
    firetime=np.zeros(nsim)
    firing=np.zeros(nsim)
    
    # Here begins the simulation loop
    while p<points:
        if recording:   
            vrec[p,:]=v # only v is recorded
        p=p+1
        # check whether an action potential has occurred
        if (np.any(np.logical_and((firing==0),(v>=threshold)))):
            ind=(np.logical_and((firing==0),(v>=threshold))).nonzero()
            firetime[ind]=p*dt
            firing[ind]=1
        # calculate the rates at time t
        an=0.01*(v+55)/(1-np.exp(-(v+55)/10))
        bn=0.125*np.exp(-(v+65)/80)
        am=0.1*(v+40)/(1-np.exp(-(v+40)/10))
        bm=4*np.exp(-(v+65)/18)
        bh=1/(1+np.exp(-(v+35)/10))
        ah=0.07*np.exp(-(v+65)/20)
        
        # calculate the membrane current, however the voltage is not yet advanced in time
        Iapp=Iamp*((p*dt)>Idel and (p*dt)<(Idel+Idur))
        Imemb=gL*(v-EL)+gNa*Nastates[7,:]*(v-ENa)/NNa+gK*Kstates[4,:]*(v-EK)/NK-Iapp
    
        # calculate the rates and the times of the next transitions 
        # (in plural because there are nsim times for nsim next transitions)
        Krates=Krat(v,Kstates)        
        next_evK=prev_evK+next_RK/np.sum(Krates,axis=0)        
        Narates=Narat(v,Nastates)
        next_evNa=prev_evNa+next_RNa/np.sum(Narates,axis=0)
        
        while np.any(p*dt>=next_evNa):
            """ Perform the transitions.
            ii are the indices of those simulations (out of the nsim) that are
            'suffering' a transition so everything within this while loop will
            occur only to those simulations with index ii
            """
            ii=np.where(p*dt>=next_evNa)[0]
            
            # build a cummulative sum of the rates, normalized to 1
            dist=np.cumsum(Narates[:,ii]/(np.ones((20,1))*np.sum(Narates[:,ii],axis=0)),axis=0)
            
            # for each cumm dist, see where does a random number fall. 
            # The xx vector stores which transitions were selected so they
            # tell which columns of the Na_trans matrix have to be added
            for a,ind in zip(ii,range(len(ii))):
                xx[a]=np.where(np.random.uniform()<dist[:,ind])[0][0]
    
            # the transitions are executed. 
            Nastates[:,ii]=Nastates[:,ii]+Na_trans[:,xx[ii]]
                    
            # calculate new random numbers and new transition times 
            # (only for the 'ii' simulations). So if any of the new transition times
            # fall again within the present time step, the 'while' loop operates again. 
            next_RNa[ii]=-np.log(np.random.uniform(size=len(ii)))
            Narates=Narat(v,Nastates)
            prev_evNa[ii]=next_evNa[ii]
            next_evNa[ii]=prev_evNa[ii]+next_RNa[ii]/np.sum(Narates[:,ii],axis=0)
        
        while np.any(p*dt>=next_evK):
            # The procedure repeats for K channels
            ii=np.where(p*dt>=next_evK)[0]
            
            dist=np.cumsum(Krates[:,ii]/(np.ones((8,1))*np.sum(Krates[:,ii],axis=0)),axis=0)
            for a,ind in zip(ii,range(len(ii))):
                xx[a]=np.where(np.random.uniform()<dist[:,ind])[0][0]
    
            Kstates[:,ii]=Kstates[:,ii]+K_trans[:,xx[ii]]
            
            prev_evK[ii]=next_evK[ii]
            
            Krates=Krat(v,Kstates)
            next_RK[ii]=-np.log(np.random.uniform(size=len(ii)))
            next_evK[ii]=prev_evK[ii]+next_RK[ii]/np.sum(Krates[:,ii],axis=0)
                
        # advance voltage, with the current that was calculated before any change
        # was done to the states, so it is still the current at time t
        # membrane capacitance is always 1 microF/cm2
        v=v-dt*Imemb
    
    NNaNs=np.sum(np.logical_or(np.isinf(v),np.isnan(v)))
    
    if recording:
        tvec=np.arange(0,Tstop,dt)
        return vrec,tvec,firetime,NNaNs,tm.time()-t0
    else:
        return firetime,NNaNs,tm.time()-t0

if __name__ == "__main__":
    nsim=10
    rec=1
    
    vrec,trec,firetime,NNaNs,time=simulate(nsim=nsim,Iamp=2,recording=rec)
    
    Eff=np.size((firetime!=0).nonzero())
    meanFT=np.mean(firetime[firetime!=0])
    if (np.size((firetime!=0).nonzero())>1):
        varFT=np.var(firetime[firetime!=0])
    else:
        varFT=0
    print "fired",Eff,"times of",nsim,"simulations"
    print "mean firing time",meanFT,"Variance of Firing time",varFT    
    
    plt.figure(1)
    plt.clf()
    plt.plot(trec,vrec)
    plt.show()