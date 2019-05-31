COMMENT
Eight state kinetic sodium channel gating scheme
Modified from k3st.mod, chapter 9.9 (example 9.7)
of the NEURON book
12 August 2008, Christoph Schmidt-Hieber
Schmidt-Hieber C, Bischofberger J. (2010) J Neurosci 30:10233-42

Stochastic model using Markov Chain modeling.
Gillespie's method with a modification for low channel numbers (or few transitions)

Used in Pezo, Soudry and Orio (2014) Front Comp Neurosci 

Possible Sodium channel transitions are indexed as follows:
First row trasitions (m particles)
    Forward         Backward
  0: i1 --> i2   1: i2 --> i1
  2: i2 --> i3   3: i3 --> i2
  4: i3 --> i4   5: i4 --> i3
Vertical transitions (h particle)
  6: i1 --> c1   7: c1 --> i1
  8: i2 --> c2   9: c2 --> i2
 10: i3 --> c3  11: c3 --> i3
 12: i4 --> o   13: o  --> i4
First row trasitions (m particles)
 14: c1 --> c2  15: c2 --> c1
 16: c2 --> c3  17: c3 --> c2
 18: c3 --> o   19: o  --> c3


ENDCOMMENT
NEURON {
    SUFFIX na8st
    USEION na READ ena WRITE ina
    GLOBAL vShift, vShift_inact, maxrate, gu_Na
    RANGE vShift_inact_local
    RANGE g, gbar, NNa, i0, se
    RANGE a1_0, a1_1, b1_0, b1_1, a2_0, a2_1
    RANGE b2_0, b2_1, a3_0, a3_1, b3_0, b3_1
    RANGE bh_0, bh_1, bh_2, ah_0, ah_1, ah_2
    RANGE i1,i2,i3,i4,c1,c2,c3,o,Nart
}

UNITS { (mV) = (millivolt) }

: initialize parameters

PARAMETER {
    se = -1  : seed to be used. se=-1 means no seed is set
    gbar = 0.018     (mho/cm2)
    gu_Na = 20e-12  (mho)

    a1_0 = 4.584982656184167e+01 (/ms)
    a1_1 = 2.393541665657613e-02 (/mV) 
    
    b1_0 = 1.440952344322651e-02 (/ms)
    b1_1 = 8.847609128769419e-02 (/mV)

    a2_0 = 1.980838207143563e+01 (/ms)
    a2_1 = 2.217709530008501e-02 (/mV) 
    
    b2_0 = 5.650174488683913e-01 (/ms)
    b2_1 = 6.108403283302217e-02 (/mV)

    a3_0 = 7.181189201089192e+01 (/ms)
    a3_1 = 6.593790601261940e-02 (/mV) 
    
    b3_0 = 7.531178253431512e-01 (/ms)
    b3_1 = 3.647978133116471e-02 (/mV)

    bh_0 = 2.830146966213825e+00 (/ms)
    bh_1 = 2.890045633775495e-01 
    bh_2 = 6.960300544163878e-02 (/mV)

    ah_0 = 5.757824421450554e-01 (/ms)
    ah_1 = 1.628407420157048e+02  
    ah_2 = 2.680107016756367e-02 (/mV)

    vShift = 12            (mV)  : shift to the right to account for Donnan potentials
                                 : 12 mV for cclamp, 0 for oo-patch vclamp simulations
    vShift_inact = 10      (mV)  : global additional shift to the right for inactivation
                                 : 10 mV for cclamp, 0 for oo-patch vclamp simulations
    vShift_inact_local = 0 (mV)  : additional shift to the right for inactivation, used as local range variable
    maxrate = 8.00e+03     (/ms) : limiting value for reaction rates
                                 : See Patlak, 1991
}

ASSIGNED {
    v    (mV)
    ena  (mV)
    g    (mho/cm2)
    ina  (milliamp/cm2)
    a1   (/ms)
    b1   (/ms)
    a2   (/ms)
    b2   (/ms)
    a3   (/ms)
    b3   (/ms)
    ah   (/ms)
    bh   (/ms)
    stsum
    R[10]   (/ms)
    dt   (ms)
    NNa
    area    (micron2)
   	i1
    i2
    i3
    i4
    c1
    c2
    c3
    o
	Nart[20]	(/ms)
	sumrtNa		(/ms)
	cumsumNa[20](/ms)
	next_evNa	(ms)
	prev_ev		(ms)
	nextRNa
	ev			(/ms)

}

STATE { mock }

BREAKPOINT {
	SOLVE mula METHOD euler
    g = gbar*o/NNa
    ina = g*(v - ena)
}

INITIAL {
	rates(v)
	NNa = floor((1e-8)*gbar*area/gu_Na + 0.5)
    if (se>=0) {set_seed(se)}
   
	: calculate initial states
    stsum=(1+ah/bh)*(1+(1+(1+a3/b3)*a2/b2)*a1/b1)
    i1=floor(NNa/stsum+0.5)
    i2=floor(NNa*(a1/b1)/stsum+0.5)
    i3=floor(NNa*(a1*a2/(b1*b2))/stsum+0.5)
    i4=floor(NNa*(a1*a2*a3/(b1*b2*b3))/stsum+0.5)
    c1=floor(NNa*(ah/bh)/stsum+0.5)
    c2=floor(NNa*(a1*ah/(b1*bh))/stsum+0.5)
    c3=floor(NNa*(a1*a2*ah/(b1*b2*bh))/stsum+0.5)
    o=floor(NNa*(a1*a2*a3*ah/(b1*b2*b3*bh))/stsum+0.5)

    :calculate the random number (log) that will be used in the first transition 
    :time and set the last transition to t=0
	nextRNa = log(scop_random())
	prev_ev=0
}


DERIVATIVE mula {  
    :recalculate rates
    rates(v)
    :recalculate time of next event with the already existing random value (nextRNa)
	next_evNa = prev_ev - nextRNa/sumrtNa  
	while (t>= next_evNa){
        transNa()
	    rates(v)
        prev_ev = next_evNa
        :calculate again next transition in case it falls within the current time step
       	next_evNa = prev_ev - nextRNa/sumrtNa  
    }

	mock'=0
}

: FUNCTION_TABLE tau1(v(mV)) (ms)
: FUNCTION_TABLE tau2(v(mV)) (ms)

UNITSOFF
PROCEDURE rates(v(millivolt)) {
    LOCAL vS
    vS = v-vShift

    a1 = a1_0*exp( a1_1*vS)
    a1 = a1*maxrate / (a1+maxrate)
    b1 = b1_0*exp(-b1_1*vS)
    b1 = b1*maxrate / (b1+maxrate)
    
    a2 = a2_0*exp( a2_1*vS)
    a2 = a2*maxrate / (a2+maxrate)
    b2 = b2_0*exp(-b2_1*vS)
    b2 = b2*maxrate / (b2+maxrate)
    
    a3 = a3_0*exp( a3_1*vS)
    a3 = a3*maxrate / (a3+maxrate)
    b3 = b3_0*exp(-b3_1*vS)
    b3 = b3*maxrate / (b3+maxrate)
    
    bh = bh_0/(1+bh_1*exp(-bh_2*(vS-vShift_inact-vShift_inact_local)))
    bh = bh*maxrate / (bh+maxrate)
    ah = ah_0/(1+ah_1*exp( ah_2*(vS-vShift_inact-vShift_inact_local)))
    ah = ah*maxrate / (ah+maxrate)

    :Nart[i] is the effective rate for the ith transition
	Nart[0]=a1*i1
	Nart[1]=b1*i2
	Nart[2]=a2*i2
	Nart[3]=b2*i3
	Nart[4]=a3*i3
	Nart[5]=b3*i4
	Nart[6]=ah*i1
	Nart[7]=bh*c1
	Nart[8]=ah*i2
	Nart[9]=bh*c2
	Nart[10]=ah*i3
	Nart[11]=bh*c3
	Nart[12]=ah*i4
	Nart[13]=bh*o
	Nart[14]=a1*c1
	Nart[15]=b1*c2
	Nart[16]=a2*c2
	Nart[17]=b2*c3
	Nart[18]=a3*c3
	Nart[19]=b3*o
	sumrtNa=0
	FROM ii=0 TO 19 {
		sumrtNa = sumrtNa + Nart[ii]
	}
	UNITSON
    
}
UNITSON

PROCEDURE transNa() {
	:perform a transition on sodium channels

    :calculate a cummulative sum of effective transition rates
    UNITSOFF
    sumrtNa=0
    FROM ii=0 TO 19 {
	  sumrtNa = sumrtNa + Nart[ii]
	  cumsumNa[ii] = sumrtNa
    }

    :normalize the cummulative sum to 1
	FROM ii=0 TO 19 {cumsumNa[ii] = cumsumNa[ii] / sumrtNa}
    UNITSON

    :draw a random number [0,1] and select a transition depending on 
    :where it falls within the cummulative sum of transition rates
	ev = scop_random()*1(/ms)
	if (ev <= cumsumNa[0]) {
		i1=i1-1
		i2=i2+1
	}
	if (cumsumNa[0] < ev && ev <= cumsumNa[1]) {
		i1=i1+1
		i2=i2-1
	}	
	if (cumsumNa[1] < ev && ev <= cumsumNa[2]) {
		i2=i2-1
		i3=i3+1
	}
	if (cumsumNa[2] < ev && ev <= cumsumNa[3]) {
		i2=i2+1
		i3=i3-1
	}	
	if (cumsumNa[3] < ev && ev <= cumsumNa[4]) {
		i3=i3-1
		i4=i4+1
	}
	if (cumsumNa[4] < ev && ev <= cumsumNa[5]) {
		i3=i3+1
		i4=i4-1
	}
	if (cumsumNa[5] < ev && ev <= cumsumNa[6]) {
		i1=i1-1
		c1=c1+1
	}
	if (cumsumNa[6] < ev && ev <= cumsumNa[7]) {
		i1=i1+1
		c1=c1-1
	}
	if (cumsumNa[7] < ev && ev <= cumsumNa[8]) {
		i2=i2-1
		c2=c2+1
	}
	if (cumsumNa[8] < ev && ev <= cumsumNa[9]) {
		i2=i2+1
		c2=c2-1
	}
	if (cumsumNa[9] < ev && ev <= cumsumNa[10]) {
		i3=i3-1
		c3=c3+1
	}
	if (cumsumNa[10] < ev && ev <= cumsumNa[11]) {
		i3=i3+1
		c3=c3-1
	}				
	if (cumsumNa[11] < ev && ev <= cumsumNa[12]) {
		i4=i4-1
		o=o+1
	}
	if (cumsumNa[12] < ev && ev <= cumsumNa[13]) {
		i4=i4+1
		o=o-1
	}
	if (cumsumNa[13] < ev && ev <= cumsumNa[14]) {
		c1=c1-1
		c2=c2+1
	}
	if (cumsumNa[14] < ev && ev <= cumsumNa[15]) {
		c1=c1+1
		c2=c2-1
	}
	if (cumsumNa[15] < ev && ev <= cumsumNa[16]) {
		c2=c2-1
		c3=c3+1
	}
	if (cumsumNa[16] < ev && ev <= cumsumNa[17]) {
		c2=c2+1
		c3=c3-1
	}
	if (cumsumNa[17] < ev && ev <= cumsumNa[18]) {
		c3=c3-1
		o=o+1
	}
	if (cumsumNa[18] < ev && ev <= cumsumNa[19]) {
		c3=c3+1
		o=o-1
	}
    :finally, calculate a random number used to determine the next transition time
    :logarithm is applied immediately
	nextRNa = log(scop_random())
}
