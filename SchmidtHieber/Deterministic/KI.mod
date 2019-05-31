TITLE Kdr potassium channels
 
COMMENT
Hodgkin Huxley potassium channel

Used in Pezo, Soudry and Orio (2014) Front Comp Neurosci 

ENDCOMMENT
 
UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}
 
? interface
NEURON {
    THREADSAFE
    SUFFIX KI                                                                                         :   KIn   
    USEION k READ ek WRITE ik                                                            :   local K may affect ion flux, and ion fluc may affect local K
    RANGE gkbar, gk, scale_a                                                                                  :   functions of position  
    GLOBAL ninf, ntau
}
 
PARAMETER {
    gkbar = .004 (mho/cm2)	 <0,1e9>   :   so these parameter are viewed and can be changed in GUI 
    scale_a = 1.0
}
 
STATE {
    n                                :  n for activation, h for inactivation 
}
 
ASSIGNED {                                   
    v (mV)                          :  variables given values outside the mod file
    celsius (degC)
    ek (mV)
    
    gk (mho/cm2)               :  variables that appear on left hand side of assigment statements within the mod file
    ik (milliamp/cm2)
    ninf
    ntau (ms)
}
 
 
? currents
BREAKPOINT {                                      : this block is responsible for making all variables consistent at time t 
    SOLVE states METHOD cnexp
    gk = gkbar*n*n*n*n
    ik = gk*(v - ek)      
}
 
 
INITIAL {
    rates(v)
    n = ninf
}

DERIVATIVE states {  
    rates(v)
    n' = (ninf-n)/ntau
}
 
? rates
PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                                                    :Call once from HOC to initialize inf at resting v.
    LOCAL  alpha, beta, sum

UNITSOFF
    alpha = scale_a*.01*vtrap(-(v+55),10) 
    beta = scale_a*.125*exp(-(v+65)/80)
    sum = alpha + beta
    ntau = 1/sum
    ninf = alpha/sum
}
 
FUNCTION vtrap(x,y) {            :Traps for 0 in denominator of rate eqns., based on three terms of infinite series expansion of exp
    if (fabs(x/y) < 1e-6) {
        vtrap = y*(1 - x/y/2)
    } else {
        vtrap = x/(exp(x/y) - 1)
    }
}
 
UNITSON
