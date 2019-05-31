COMMENT
Eight state kinetic sodium channel gating scheme
Modified from k3st.mod, chapter 9.9 (example 9.7)
of the NEURON book
12 August 2008, Christoph Schmidt-Hieber

**** Converted to DERIVATIVE and added DA stochastics by Patricio Orio, 2014 ****
Stochastic Hodgkin and Huxley model using Markov Chain modeling with Stochastic Shielding as proposed by 
Schmandt & Galan (2012) Phys Rev Lett 109:118101
Thus the only stochastic transitions are i4<-->o and  c3<-->o

Stochastic transitions are solved with Gillespie's method with a modification for low channel numbers (or few transitions).
Deterministic transitions are left to Neuron's numerical method
(note that we obtained comparable results with Euler method in Python)
Written for Pezo, Soudry and Orio (2014) Front. Comp Neurosci
ENDCOMMENT

NEURON {
    SUFFIX na8st
    USEION na READ ena WRITE ina
    GLOBAL vShift, vShift_inact, maxrate, gu_Na
    RANGE vShift_inact_local
    RANGE g, gbar, NNa
    RANGE a1_0, a1_1, b1_0, b1_1, a2_0, a2_1
    RANGE b2_0, b2_1, a3_0, a3_1, b3_0, b3_1
    RANGE bh_0, bh_1, bh_2, ah_0, ah_1, ah_2, se
    RANGE NaT,cumsumNa,nextRNa,next_evNa,prev_evNa,sumrtNa,i1
}

UNITS { (mV) = (millivolt) }

: initialize parameters

PARAMETER {
    se = -1
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
    i1           :i1 is treated as assigned because it doesnt obbey a diff eq
    R[2]   (/ms)
    dt      (ms)
    NNa
    area    (micron2)
	NaT[4]
    cumsumNa[4]
    nextRNa
    next_evNa   (ms)
    prev_evNa   (ms)
    sumrtNa     (/ms)
    
}

STATE { c1 c2 c3 i2 i3 i4 o }

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar*o/NNa
    ina = g*(v - ena)
}

INITIAL {
	rates(v)
	NNa = floor((1e-8)*gbar*area/gu_Na + 0.5)
    if (se>=0) {set_seed(se)}
   
	stsum=(1+ah/bh)*(1+(1+(1+a3/b3)*a2/b2)*a1/b1)
	i1=NNa/stsum
	i2=NNa*(a1/b1)/stsum
	i3=NNa*(a1*a2/(b1*b2))/stsum
	i4=NNa*(a1*a2*a3/(b1*b2*b3))/stsum
	c1=NNa*(ah/bh)/stsum
	c2=NNa*(a1*ah/(b1*bh))/stsum
	c3=NNa*(a1*a2*ah/(b1*b2*bh))/stsum
	o=NNa*(a1*a2*a3*ah/(b1*b2*b3*bh))/stsum
	rates(v)
	nextRNa = log(scop_random())
	prev_evNa=0

}

DERIVATIVE states {
    rates(v)
   	i2' = (-a2-b1-ah)*i2 + a1*i1 + b2*i3 + bh*c2
	i3' = (-a3-b2-ah)*i3 + a2*i2 + b3*i4 + bh*c3
	i4' = (-b3)*i4 + a3*i3 : + o*bh + R[0]
	c1' = (-bh-a1)*c1 + b1*c2 + ah*i1
        c2' = (-a2-b1-bh)*c2 + a1*c1 + b2*c3 + ah*i2
        c3' = (-b2-bh)*c3 + a2*c2 + ah*i3 :+b3*o +  R[1]
:         o' = (-b3-bh)*o + a3*c3 + ah*i4 - R[1] - R[0]
	next_evNa = prev_evNa - nextRNa/sumrtNa  
	while (t>= next_evNa){
		transNa()
		prev_evNa = next_evNa
		next_evNa = prev_evNa - nextRNa/sumrtNa  
	}
        i1 = NNa-i2-i3-i4-c1-c2-c3-o

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

	NaT[0]=ah*i4  :i4 --> o
	NaT[1]=bh*o	:o --> i4
	NaT[2]=a3*c3	:c3 --> o
	NaT[3]=b3*o	:o --> c3
	sumrtNa=NaT[0] + NaT[1] + NaT[2] + NaT[3]
    
}

PROCEDURE transNa() {
    LOCAL ev
	rates(v)
	sumrtNa=0
	FROM ii=0 TO 3 {
		sumrtNa = sumrtNa + NaT[ii]
		cumsumNa[ii] = sumrtNa
	}
	FROM ii=0 TO 3 {cumsumNa[ii] = cumsumNa[ii] / sumrtNa}
	ev = scop_random()
	if (ev <= cumsumNa[0]) {
		i4=i4-1
		o=o+1
	}
	if (cumsumNa[0] < ev && ev <= cumsumNa[1]) {
		i4=i4+1
		o=o-1
	}	
	if (cumsumNa[1] < ev && ev <= cumsumNa[2]) {
		c3=c3-1
		o=o+1
	}
	if (cumsumNa[2] < ev && ev <= cumsumNa[3]) {
		c3=c3+1
		o=o-1
	}

	if (i4>NNa){i4=NNa}
	if (c3>NNa){c3=NNa}
	if (o>NNa){o=NNa}
	if (i4<0){i4=0}
	if (c3<0){c3=0}
	if (o<0){o=0}

	nextRNa=log(scop_random())
}

UNITSON


