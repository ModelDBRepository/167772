TITLE Kdr potassium channels with DA and reflection
 
COMMENT
Hodgkin and Huxley potassium channels

Stochastic equations with diffusion aproximation using
a reflection method to correct normalization and bounding of variables
Equations as in Dangerfield, Kay & Burrage (2012) Phy Rev E 85:051907-1
DOI: 10.1103/PhysRevE.85.051907
 
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
        RANGE gnabar, gkbar, NNa, NK, contn, scale_a, se
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
	n0
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
	R[4]	(/ms)
	contn
}
 
BREAKPOINT {
    SOLVE states METHOD cnexp : Projection only works well with cnexp, don't use euler
	ik = gkbar*n4*(v - ek)   
}
 
 
INITIAL {
	rates(v)
	NK = floor((1e-8)*gkbar*area/gu_K + 0.5)
    if (se>=0) {set_seed(se)}	
	contn = 0
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
	n0' = -4*an*n0 + bn*n1 + R[0]
	n1' = (-3*an-bn)*n1 + 4*an*n0 + 2*bn*n2 - R[0] + R[1]
	n2' = (-2*an-2*bn)*n2 + 3*an*n1 + 3*bn*n3 -R[1] + R[2]
	n3' = (-an-3*bn)*n3 + 2*an*n2 + 4*bn*n4 -R[2] + R[3]
	n4' = -4*bn*n4 + an*n3 -R[3]

	projection()
}
 

PROCEDURE projection() { :Computes projection
	LOCAL sumn, jm, flagn, w, k, tem, jj, tmax, tsum, kk, bget, N[5], Naux[5]
	UNITSOFF
	N[0]=n0
	N[1]=n1
	N[2]=n2
	N[3]=n3
	N[4]=n4

	:::::::N:::::::
	sumn = 0
	flagn = 0
	FROM jm=0 TO 4 {
		sumn = sumn + N[jm]
		if (N[jm]<0 || N[jm]>1) {flagn = flagn + 1}		
		Naux[jm]=N[jm]
	}
	:::::::Ordena N:::::::
	w=0
	if (sumn != 1 || flagn != 0) {
		while (w != (5-1)) {
			w=0
			FROM k = 0 TO (5-2) {
				if (N[k]>=N[k+1]) {
					w=w+1
				} else{
					tem = N[k]
					N[k] = N[k+1]
					N[k+1] = tem
				}
			}
		}
	:::::::Proyecta N:::::::
	bget = 0
	tsum = 0
	FROM jj = 0 TO 3 {
		tsum = tsum + N[jj]
		tmax = (tsum - 1)/(jj+1)
			if (tmax > N[jj+1]){
				bget = 1
				VERBATIM
				break;
				ENDVERBATIM
			}
		}
	if (bget==0) {tmax = (tsum + N[4] -1)/5} 
	FROM kk=0 TO 4 {
		if (Naux[kk]>tmax) {
			N[kk]=Naux[kk]-tmax
		} else {
			N[kk]=0
			}
		}
		contn=contn+1 :cuenta las veces que proyecta n
	}
	
	:::::::Entrega proyeccion:::::::
	n0=N[0]
	n1=N[1]
	n2=N[2]
	n3=N[3]
	n4=N[4]
	UNITSON
}


PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.

UNITSOFF
    
    :"n" activation 
    an = scale_a*.01*vtrap(-(v+55),10) 
    bn = scale_a*.125*exp(-(v+65)/80)

   	FROM ii=0 TO 3 {R[ii]=normrand(0,1/sqrt(NK*dt))}
	R[0] = R[0]*sqrt(4*an*n0+bn*n1)
	R[1] = R[1]*sqrt(3*an*n1+2*bn*n2)
	R[2] = R[2]*sqrt(2*an*n2+3*bn*n3)
	R[3] = R[3]*sqrt(an*n3+4*bn*n4)
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON
