TITLE Kdr potassium channel with SSmc
COMMENT
Hodgkin-Huxley potassium channels 

Stochastic model using Markov Chain modeling with Stochastic Shielding as proposed by 
Schmandt & Galan (2012) Phys Rev Lett 109:118101
Thus the only stochastic transitions are n3<-->n4

Stochastic transitions are solved with Gillespie's method with a modification for low channel numbers (or few transitions).
Deterministic transitions are left to Neuron's numerical method
(note that we obtained comparable results with Euler method in Python)

Written for Pezo, Soudry and Orio (2014) Front. Comp Neurosci

ENDCOMMENT
 
UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}
 
NEURON {
        SUFFIX KI
        USEION k READ ek WRITE ik
        RANGE gnabar, gkbar, NNa, NK, se, scale_a
    RANGE 	K_4_3,K_3_4,prev_evK,next_evK,nextRK,sumrtK,n0 
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
	K_4_3	(/ms)
	K_3_4	(/ms)
	prev_evK	(ms)
	next_evK	(ms)
	nextRK
	sumrtK		(/ms)
	ev
        n0    : treated as assigned as it doesn't obey diff eq
}
 
BREAKPOINT {
    SOLVE states METHOD cnexp
	ik = gkbar*n4*(v - ek)/NK
}
 
 
INITIAL {
    LOCAL N
	rates(v)
	NK = floor((1e-8)*gkbar*area/gu_K + 0.5)	
    if (se>=0) {set_seed(se)}
	N=an/bn
	stsum=(1+N)^4
	n0=NK/stsum
	n1=NK*4*N/stsum
	n2=NK*6*N^2/stsum
	n3=NK*4*N^3/stsum
	n4=NK*N^4/stsum

	rates(v)
    nextRK=log(scop_random())
	prev_evK=0
}

DERIVATIVE states {  
	rates(v)
	
	n1' = (-3*an-bn)*n1 + 4*an*n0 + 2*bn*n2
	n2' = (-2*an-2*bn)*n2 + 3*an*n1 + 3*bn*n3
    n3' = (-3*bn)*n3 + 2*an*n2
:	n4' = (-4*bn)*n4 + an*n3 - R

	next_evK = prev_evK - nextRK/sumrtK
	while (t>= next_evK){
		transK()
        	prev_evK = next_evK
		next_evK = prev_evK - nextRK/sumrtK
	}

	n0 = NK-n1-n2-n3-n4
}
 
PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.

UNITSOFF
    
    :"n" activation 
    an = scale_a*.01*vtrap(-(v+55),10) 
    bn = scale_a*.125*exp(-(v+65)/80)
	K_3_4 = an*n3
	K_4_3 = 4*bn*n4
	sumrtK = K_3_4 + K_4_3


}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON

PROCEDURE transK() {
	rates(v)
	sumrtK = K_3_4 + K_4_3
	ev = scop_random()

	if (ev <= K_3_4 / sumrtK) {
		n3=n3-1
		n4=n4+1
	} else {
		n3=n3+1
		n4=n4-1
	}	
	if (n3>NK){n3=NK}
	if (n4>NK){n4=NK}
	if (n3<0){n3=0}
	if (n4<0){n4=0}
	nextRK = log(scop_random())
}
