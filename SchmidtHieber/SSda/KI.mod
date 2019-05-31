TITLE Kdr channels with DA and Stochastic Shielding app
 
COMMENT

Hodgkin Huxley potassium channels

Stochastic equations with diffusion aproximation and Stochastic Shielding approximation (hhSSda)
Equations as in Orio & Soudry (2012) PLoS One with the approximation
    proposed by Schmandt & Galan (2012) Phys Rev Lett 109:118101
Stochastic terms for transitions that do not connect the conducting states are neglected.
Thus the only transitions that keep random terms are i4<-->o and  c3<-->o
Variables are unbound and real square roots are ensured by applying absolute
    values to variables, but only in random terms
    
Implemented for Pezo, Soudry and Orio (2014) Front Comp Neurosci 


ENDCOMMENT
 
UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}
 
NEURON {
        SUFFIX KI
        USEION k READ ek WRITE ik
        RANGE gnabar, gkbar, NNa, NK, scale_a, n0, se
        GLOBAL gu_K
 	THREADSAFE : assigned GLOBALs will be per thread
}
 
PARAMETER {
        se = -1
        gkbar = .004 (S/cm2)	<0,1e9>
        gu_K = 20e-12	(S)
        scale_a = 1.0
}
 
STATE {	
	n1
	n2
	n3
	n4
}
 
ASSIGNED {
	NK
	area	(micron2)
	v (mV)
	celsius (degC)
	ek (mV)
	dt (ms)
	ik (mA/cm2)
	an	(/ms)
	bn	(/ms)
	stsum
	N
	R	(/ms)
        n0    : treated as assigned as it doesn't obey diff eq
}
 
BREAKPOINT {
    SOLVE states METHOD cnexp 
	ik = gkbar*n4*(v - ek)   
}
 
 
INITIAL {
	rates(v)
	NK = floor((1e-8)*gkbar*area/gu_K + 0.5)	
    if (se>=0) {set_seed(se)}
	
	N=an/bn
	stsum=(1+N)^4
	n0=1/stsum
	n1=4*N/stsum
	n2=6*N^2/stsum
	n3=4*N^3/stsum
	n4=N^4/stsum

	rates(v)
}

DERIVATIVE states {  
	rates(v)
	
	n1' = (-3*an-bn)*n1 + 4*an*n0 + 2*bn*n2
	n2' = (-2*an-2*bn)*n2 + 3*an*n1 + 3*bn*n3
        n3' = (-an-3*bn)*n3 + 2*an*n2 + 4*bn*n4 + R
	n4' = (-4*bn)*n4 + an*n3 - R
	n0 = 1-n1-n2-n3-n4
}
 
PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.

UNITSOFF
    
    :"n" activation 
    an = scale_a*.01*vtrap(-(v+55),10) 
    bn = scale_a*.125*exp(-(v+65)/80)

	R = normrand(0,1/sqrt(NK*dt))*sqrt(fabs(an*n3+4*bn*n4))
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON
