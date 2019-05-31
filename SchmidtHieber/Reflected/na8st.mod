COMMENT
Eight state kinetic sodium channel gating scheme
Modified from k3st.mod, chapter 9.9 (example 9.7)
of the NEURON book
from Schmidt-Hieber C, Bischofberger J. (2010) J Neurosci 30:10233-42

 **** Converted to DERIVATIVE and added DA stochastics by Patricio Orio, 2014 ****
Stochastic Hodgkin and Huxley equations with diffusion aproximation using
a reflection method to correct normalization and bounding of variables
Equations as in Dangerfield, Kay & Burrage (2012) Phy Rev E 85:051907-1
DOI: 10.1103/PhysRevE.85.051907
 
Implemented for Pezo, Soudry and Orio (2014) Front Comp Neurosci 
ENDCOMMENT

NEURON {
    SUFFIX na8st
    USEION na READ ena WRITE ina
    GLOBAL vShift, vShift_inact, maxrate, gu_Na
    RANGE vShift_inact_local
    RANGE g, gbar, NNa, contm, se
    RANGE a1_0, a1_1, b1_0, b1_1, a2_0, a2_1
    RANGE b2_0, b2_1, a3_0, a3_1, b3_0, b3_1
    RANGE bh_0, bh_1, bh_2, ah_0, ah_1, ah_2
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
    R[10]   (/ms)
    dt      (ms)
    NNa
    area    (micron2)
    contm
}

STATE { i1 i2 i3 i4 c1 c2 c3 o }

BREAKPOINT {
    SOLVE states METHOD cnexp : Projection only works well with cnexp, don't use euler
    g = gbar*o
    ina = g*(v - ena)
}

INITIAL {
	rates(v)
	NNa = floor((1e-8)*gbar*area/gu_Na + 0.5)
    if (se>=0) {set_seed(se)}
	contm = 0
	stsum=(1+ah/bh)*(1+(1+(1+a3/b3)*a2/b2)*a1/b1)
        i1=1/stsum
        i2=(a1/b1)/stsum
        i3=(a1*a2/(b1*b2))/stsum
        i4=(a1*a2*a3/(b1*b2*b3))/stsum
        c1=(ah/bh)/stsum
        c2=(a1*ah/(b1*bh))/stsum
        c3=(a1*a2*ah/(b1*b2*bh))/stsum
         o=(a1*a2*a3*ah/(b1*b2*b3*bh))/stsum
}

DERIVATIVE states {
    rates(v)
	
	i1' = (-ah-a1)*i1 + bh*c1 + b1*i2 + R[0] + R[3] 	
	i2' = (-a2-b1-ah)*i2 + a1*i1 + b2*i3 + bh*c2 -R[0]+R[1]+R[4]	
	i3' = (-a3-b2-ah)*i3 + a2*i2 + b3*i4 + bh*c3 -R[1]+R[2]+R[5]
	i4' = (-b3-ah)*i4 + a3*i3 + o*bh -R[2]+R[6]
	c1' = (-bh-a1)*c1 + b1*c2 + ah*i1 + R[7]-R[3]
        c2' = (-a2-b1-bh)*c2 + a1*c1 + b2*c3 + ah*i2 -R[7]+R[8]-R[4]
        c3' = (-a3-b2-bh)*c3 + a2*c2 + b3*o + ah*i3 -R[8]+R[9]-R[5]
         o' = (-b3-bh)*o + a3*c3 + ah*i4 -R[9]-R[6]

	projection()
}

: FUNCTION_TABLE tau1(v(mV)) (ms)
: FUNCTION_TABLE tau2(v(mV)) (ms)


PROCEDURE projection() { :Computes projection
	LOCAL summh, im, flagm, w, k, tem, ii, tmax, tsum, kk, bget, MH[8], Maux[8]
	UNITSOFF
	MH[0]=i1
	MH[1]=i2
	MH[2]=i3
	MH[3]=i4
	MH[4]=c1
	MH[5]=c2
	MH[6]=c3
	MH[7]=o

	:::::::M:::::::
	summh = 0
	flagm = 0
	FROM im=0 TO 7 {
		summh = summh + MH[im]
		if (MH[im]<0 || MH[im]>1) {flagm = flagm + 1}
		Maux[im]=MH[im]
	}
	:::::::Ordena M:::::::
	w=0
	if (summh != 1 || flagm != 0) {
		while (w != (8-1)) {
			w=0
			FROM k = 0 TO (8-2) {
				if (MH[k]>=MH[k+1]) {
					w=w+1
				} else{
					tem = MH[k]
					MH[k] = MH[k+1]
					MH[k+1] = tem
				}
			}
		}
	:::::::Proyecta M:::::::
		bget = 0
		tsum = 0
		FROM ii = 0 TO 6 {
			tsum = tsum + MH[ii]
			tmax = (tsum - 1)/(ii+1)
			if (tmax > MH[ii+1]){
				bget = 1
				VERBATIM
				break;
				ENDVERBATIM
			}
		}
		if (bget==0) {tmax = (tsum + MH[7] -1)/8} 
		FROM kk=0 TO 7 {
			if (Maux[kk]>tmax) {
				MH[kk]=Maux[kk]-tmax
			} else {
				MH[kk]=0
			}
		}
		contm=contm+1 :cuenta las veces que proyecta m
	}

	
	:::::::Entrega proyeccion:::::::
	i1=MH[0]
	i2=MH[1]
	i3=MH[2]
	i4=MH[3]
	c1=MH[4]
	c2=MH[5]
	c3=MH[6]
	o=MH[7]
	UNITSON
}


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
    
   	FROM ii=0 TO 9 {R[ii]=normrand(0,1/sqrt(NNa*dt))}
	R[0] = R[0]*sqrt(a1*i1+b1*i2)
	R[1] = R[1]*sqrt(a2*i2+b2*i3)
	R[2] = R[2]*sqrt(a3*i3+b3*i4)
	R[3] = R[3]*sqrt(ah*i1+bh*c1)
	R[4] = R[4]*sqrt(ah*i2+bh*c2)
	R[5] = R[5]*sqrt(ah*i3+bh*c3)
	R[6] = R[6]*sqrt(ah*i4+bh*o)
	R[7] = R[7]*sqrt(a1*c1+b1*c2)
	R[8] = R[8]*sqrt(a2*c2+b2*c3)
	R[9] = R[9]*sqrt(a3*c3+b3*o)

}
UNITSON
