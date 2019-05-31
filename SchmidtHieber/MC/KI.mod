TITLE Kdr potassium channels with Markov Chain modelling
 
COMMENT

Stochastic model using Markov Chain modeling.
Gillespie's method with a modification for low channel numbers (or few transitions)

Used in Pezo, Soudry and Orio (2014) Front Comp Neurosci 

ENDCOMMENT
 
UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
	(S) = (siemens)
}
 
NEURON {
        SUFFIX KI
        USEION k READ ek WRITE ik
        RANGE gkbar, NK, n0, scale_a, se
        RANGE Kst, Krt
        GLOBAL gu_K
 		THREADSAFE : assigned GLOBALs will be per thread
}
 
PARAMETER {
        se = -1
        gkbar = .004 (S/cm2)	<0,1e9>
        gu_K = 20e-12	(S)
        scale_a = 1.0
}
 
STATE {mock}
 
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
   	Kst[5]
	Krt[8]		(/ms)
	sumrtK		(/ms)
	cumsumK[8]	(/ms)
	nextRK
	next_evK	(ms)
	prev_ev		(ms)
	ev			(/ms)

}
 
BREAKPOINT {
	SOLVE mula METHOD euler
	ik = gkbar*Kst[4]*(v - ek)/NK   
}
 
 
INITIAL {
	rates(v)
	NK = floor((1e-8)*gkbar*area/gu_K + 0.5)	
	if (se>=0) {set_seed(se)}
	N=an/bn
	stsum=(1+N)^4
	Kst[0]=floor(NK/stsum+0.5)
	Kst[1]=floor(NK*4*N/stsum+0.5)
	Kst[2]=floor(NK*6*N^2/stsum+0.5)
	Kst[3]=floor(NK*4*N^3/stsum+0.5)
	Kst[4]=floor(NK*N^4/stsum+0.5)

	nextRK = log(scop_random())
	prev_ev=0
}

DERIVATIVE mula {  
    rates(v)
	next_evK = prev_ev - nextRK/sumrtK
	while (t>= next_evK){
		transK()
		rates(v)
        prev_ev = next_evK
	    next_evK = prev_ev - nextRK/sumrtK
	}
	mock'=0
}
 

PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.

	UNITSOFF    
    :"n" activation 
    an = scale_a*.01*vtrap(-(v+55),10) 
    bn = scale_a*.125*exp(-(v+65)/80)

	Krt[0]=4*an*Kst[0]
	Krt[1]=bn*Kst[1]
	Krt[2]=3*an*Kst[1]
	Krt[3]=2*bn*Kst[2]
	Krt[4]=2*an*Kst[2]
	Krt[5]=3*bn*Kst[3]
	Krt[6]=an*Kst[3]
	Krt[7]=4*bn*Kst[4]
	sumrtK=0
	FROM ii=0 TO 7 {
		sumrtK = sumrtK + Krt[ii]
	}

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
	sumrtK=0
    UNITSOFF
	FROM ii=0 TO 7 {
		sumrtK = sumrtK + Krt[ii]
		cumsumK[ii] = sumrtK
	}
	FROM ii=0 TO 7 {cumsumK[ii] = cumsumK[ii] / sumrtK}
    UNITSON
	ev = scop_random()*1(/ms)
	if (ev <= cumsumK[0]) {
		Kst[0]=Kst[0]-1
		Kst[1]=Kst[1]+1
	}
	if (cumsumK[0] < ev && ev <= cumsumK[1]) {
		Kst[0]=Kst[0]+1
		Kst[1]=Kst[1]-1
	}	
	if (cumsumK[1] < ev && ev <= cumsumK[2]) {
		Kst[1]=Kst[1]-1
		Kst[2]=Kst[2]+1
	}
	if (cumsumK[2] < ev && ev <= cumsumK[3]) {
		Kst[1]=Kst[1]+1
		Kst[2]=Kst[2]-1
	}	
	if (cumsumK[3] < ev && ev <= cumsumK[4]) {
		Kst[2]=Kst[2]-1
		Kst[3]=Kst[3]+1
	}
	if (cumsumK[4] < ev && ev <= cumsumK[5]) {
		Kst[2]=Kst[2]+1
		Kst[3]=Kst[3]-1
	}
	if (cumsumK[5] < ev && ev <= cumsumK[6]) {
		Kst[3]=Kst[3]-1
		Kst[4]=Kst[4]+1
	}
	if (cumsumK[6] < ev && ev <= cumsumK[7]) {
		Kst[3]=Kst[3]+1
		Kst[4]=Kst[4]-1
	}
	nextRK = log(scop_random())
}

