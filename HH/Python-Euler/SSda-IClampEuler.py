# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:44:54 2013

@authors: Patricio Orio & Danilo Pezo
Current clamp simulation of the stochastic Hodgkin & Huxley model
Diffusion approximation method (Orio & Soudry, 2012) combined with a 
Stochastic Shielding approach (Shcmandt & Galan, 2012)
i.e., stochastic terms for transitions not involving the conducting states
are neglected

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

def simulate(nsim=30,Tstop=15,dt=0.005,
             Idel=0,Idur=-1,Iamp=0,NNa=600,NK=180,recording=0):
    """
    Stochastic simulation of HH model using unbound diffusion approximation
    (Orio & Soudry, PLoS One 2012) with stochastic shielding (Schmand and Galan
    2012).     
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

    NNaNs=0; p=0
    t0=tm.time()
    
    if recording:
        vrec=np.zeros((points,nsim))
    
    # calculate the initial conditions at -65 mV, including the initial 
    # distribution of channels in each state (arrays m and n). It's deterministic
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
    
    N=an/bn
    Kstatesum=(1+N)**4        
    n=np.array([np.ones(nsim),4*N,6*N**2,4*N**3,N**4])/np.outer(np.ones((5,1)),Kstatesum)
        
    
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
        Imemb=gK*n[4,]*(v-EK)+gNa*m[7,]*(v-ENa)+gL*(v-EL)-Iapp
        
        # Deterministic part of sodium channel transitions
        trans_m=np.array([-(3*am+ah)*m[0,]+bm*m[1,]+bh*m[4,],
                3*am*m[0,]-(2*am+bm+ah)*m[1,]+2*bm*m[2,]+bh*m[5,],
                2*am*m[1,]-(2*bm+am+ah)*m[2,]+3*bm*m[3,]+bh*m[6,],
                am*m[2,]-(3*bm+ah)*m[3,]+bh*m[7,],
                -(3*am+bh)*m[4,]+bm*m[5,]+ah*m[0,],
                3*am*m[4,]-(2*am+bm+bh)*m[5,]+2*bm*m[6,]+ah*m[1,],
                2*am*m[5,]-(2*bm+am+bh)*m[6,]+3*bm*m[7,]+ah*m[2,],
                am*m[6,]-(3*bm+bh)*m[7,]+ah*m[3,]])
        # random part of sodium channel transitions
        # Absolute values of current state variables are used just to avoid
        # imaginary roots
        ma=abs(m)
        R=np.random.normal(0,1,(2,nsim))*np.sqrt((dt/NNa)*
                        np.array([ah*ma[3,]+bh*ma[7,],am*ma[6,]+3*bm*ma[7,]]))
        Wtm=np.array([np.zeros(nsim),np.zeros(nsim),np.zeros(nsim),R[0],
                      np.zeros(nsim),np.zeros(nsim),R[1],-R[1]-R[0]])
        #advance sodium states
        m=m+dt*trans_m+Wtm
        #fullfil normalization in the easiest way
        #The assumption is that by doing this at each time step, the deviations
        #are neglectable
        m[0,:]=np.ones((1,nsim))-np.sum(m[1:,:],axis=0)
        
        # Deterministic part of potassium channel transitions        
        trans_n=np.array([-4*an*n[0,]+bn*n[1,],-(bn+3*an)*n[1,]+4*an*n[0,]+2*bn*n[2,],
                          -(2*bn+2*an)*n[2,]+3*an*n[1,]+3*bn*n[3,],-(3*bn+an)*n[3,]+2*an*n[2,]+4*bn*n[4,],
                           -4*bn*n[4,]+an*n[3,]])
        # random part of sodium channel transitions
        # Absolute values of current state variables are used just to avoid
        # imaginary roots
        na=abs(n)
        R=np.random.normal(0,1,(1,nsim))*np.sqrt((dt/NK)*np.array([an*na[3,]+4*bn*na[4,]]))
        Wtn=np.array([np.zeros(nsim),np.zeros(nsim),np.zeros(nsim),R[0],-R[0]])
        #advance potassium states        
        n=n+dt*trans_n+Wtn
        #normalization
        n[0,:]=np.ones((1,nsim))-np.sum(n[1:,:],axis=0)
                
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