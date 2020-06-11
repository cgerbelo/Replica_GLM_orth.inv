# Toolbox 

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import ortho_group
from scipy import special
import sys
import mpmath
from sklearn.svm import LinearSVC
from sklearn import linear_model
import time
from scipy import optimize


##### This is a basic Python implementation of the (provably correct) replica prediction for learning an arbitrary teacher GLM with an arbitrary student GLM with orthogonally invariant data with arbitrary spectrum #####
##### No noise is implemented in the teacher, apart from the square loss where a delta0 parameter is allowed. Adding noise for other losses is included in the presented formalism, and requires a little calculus. #####

####### Proximal operators and their derivatives #######

# proximal of square loss:

def prox_sqloss(g,y,p):
    return((g/(1+g))*(p)+1/(1+g)*y)
    
# proximal of l1 penalty : soft-thresholding

def soft_thresh(x,thresh):
	n = len(x)
	res = np.zeros(n)
	
	for i in range(n):
		
		if np.abs(x[i])>thresh:
			res[i] = x[i]-np.sign(x[i]-thresh)*thresh
			
		if np.abs(x[i])<=thresh :
			res[i] = 0

	return(res)

# derivative of soft thresholding

def soft_thresh_der(x,thresh):
	n = len(x)
	res = np.zeros(n)
	
	for i in range(n):
		
		if np.abs(x[i])>thresh:
			res[i] = 1
			
		if np.abs(x[i])<=thresh :
			res[i] = 0

	return(res)
 
# proximal of the hinge loss:

def prox_hinge(g,y,p):

    flag = g*(1-y*p)

    if flag>=1:
        return(p+y/g)
    
    if 0<=flag<=1:
        return(y)

    if flag<=0:
        return(p)

# derivative of the proximal of the hinge loss:
def der_prox_hinge(g,y,p):

    flag = g*(1-y*p)
    
    if flag>=1:
        return(1)
        
    if 0<=flag<=1:
        return(0)
    
    if flag<=0:
        return(1)
    
# proximal of logistic loss (no closed form)
def prox_log(g,y,p):
    return(np.float(optimize.fsolve(lambda x: x-p-y/g*1/(1+np.exp(y*np.float(x))),1)))
    
# derivative of the proximal of the logistic loss
def der_prox_log(g,y,p):
    return(np.float(1/(1+1/(2*g)*1/(1+np.cosh(y*prox_log(g,y,p))))))
    
    
#### useful fonctions for synthetic data generation ####

# sampling of Haar matrices, faster than scipy ortho_group and sufficient for considered setup. Based on same method as scipy ortho_group, see scipy userguide.

def Haar_sample(n):
    #We first generate a m x m orthogonal matrix by taking the QR decomposition of a random m x m gaussian matrix
    gaussian_matrix = np.random.normal(0, 1., (n,n))
    O, R = np.linalg.qr(gaussian_matrix)
    #Then we multiply on the right by the signs (or phases in the complex case) of the diagonal of R to really get the Haar measure
    D = np.diagonal(R)
    Lambda = D / np.abs(D)   # normalization
    O = np.multiply(O, Lambda)
    return(O)
    
# gauss-bernoulli distribution on the gorund-truth weights x0, allows the modelling of sparsity.

def gauss_bernoulli(rho,n):
	
	x = np.zeros(n)
	
	for i in range(n):
		
		indicator = np.random.random()
		
		if indicator > rho :
			x[i] = 0
		
		if indicator <= rho : 
			x[i] = np.random.randn()
			
	return(x)


## useful eigenvalue distributions ##
# Marcenko-Pastur density

# careful, scaling (aspect ratio) is done when integrating, the following density is expected to be normalized whatever the aspect ratio.

def MP(x,alpha):
	a = (1-np.sqrt(1/alpha))**2
	b = (1+np.sqrt(1/alpha))**2
	return(max(1,alpha)*np.sqrt(max(0,(x-a))*max(0,b-x))/(2*np.pi*x))

# rescaled uniform for prescribing the spectrum for rotationally invariant data matrix.
    
def unif(x,asp):

    a = (1-asp)**2
    b = (1+asp)**2
    
    if a<=np.sqrt(x)<=b:
        return(1/(2*(b-a))*1/np.sqrt(x))
        
    else:
        return(0)

# Building orthogonally invariant random matrix with squared uniform eigenvalue distribution

def build_matrix(a,b,n,d,asp,select):
	
	V = Haar_sample(n)
	O = Haar_sample(d)
	if select == 0:
		eigen = np.random.uniform(a,b,min(n,d))
	if select == 1:
		eigen = np.ones(min(n,d))
	D_prev = np.diag(eigen)
	comp = np.zeros((max(n,d)-min(n,d),min(n,d)))
	
	if n>=d :
		
		D = np.concatenate((D_prev,comp),0)
	
	if n<d:
		
		D = np.concatenate((D_prev,np.transpose(comp)),1)
		
	F = V@D@np.transpose(O)
	return(F,D)


###### updates stemming from the regularization, an elastic net, with a Gauss-Bernoulli prior ########

def aux_enet(Q1x,l1,t1):
    return((l1**2+Q1x**2*t1)/(Q1x**2)*special.erfc(l1/(Q1x*np.sqrt(2*t1)))-l1*np.sqrt(2*t1)*np.exp(-l1**2/(2*Q1x**2*t1))/(Q1x*np.sqrt(np.pi)))

### functions to update m1x,chi1x and q1x as presented in the equation block (90) ###

def m1x_func(rho,Q1x,mh1x,chih1x,l1,l2):
    t0 = (mh1x/Q1x)**2
    t1 = chih1x/Q1x**2
    s = 1/(1+l2/Q1x)
    return(rho*s*mh1x/Q1x*special.erfc(l1/(Q1x*np.sqrt(2*(t1+t0)))))

def chi1x_func(rho,Q1x,mh1x,chih1x,l1,l2):
    t0 = (mh1x/Q1x)**2
    t1 = chih1x/Q1x**2
    s = 1/(1+l2/Q1x)
    return(1/Q1x*s*((1-rho)*special.erfc(l1/(Q1x*np.sqrt(2*t1)))+rho*special.erfc(l1/(Q1x*np.sqrt(2*(t1+t0))))))

def q1x_func(rho,Q1x,mh1x,chih1x,l1,l2):
    t0 = (mh1x/Q1x)**2
    t1 = chih1x/Q1x**2
    s = 1/(1+l2/Q1x)
    return(s**2*((1-rho)*aux_enet(Q1x,l1,t1)+rho*aux_enet(Q1x,l1,t1+t0)))
    
##### updates stemming from the loss functions and teacher : square loss with noisy linear teacher, square loss with signed teacher, hinge loss with signed teacher, logistic loss with signed teacher  #####

##### specifc functions for each loss to update m1z, chi1z, q1z from equation block (90) #####

## square loss with noisy linear teacher lo==0 (simple closed form solution) ##
def m1z_sqloss(mh1z,Tz,Q1z):
    return(Tz/(1+Q1z)*(mh1z+1))

def chi1z_sqloss(Q1z):
    return(1/(1+Q1z))

def q1z_sqloss(Q1z,Tz,mh1z,delta0,chih1z):
    return(1/(1+Q1z)**2*(Tz*(1+mh1z)**2+delta0+chih1z))


## square loss with signed teacher, no noise lo==1 (simple closed form solution) ##

def m1z_sgn_sqloss(Q1z,mh1z,Tz):
    return(1/(1+Q1z)*(mh1z*Tz+np.sqrt(2*Tz/np.pi)))
    
def chi1z_sgn_sqloss(Q1z,mh1z,Tz):
    return(1/(1+Q1z))

def q1z_sgn_sqloss(Q1z,mh1z,Tz,chih1z):
    return((1/(1+Q1z))**2*(mh1z**2*Tz+chih1z+1+2*mh1z*np.sqrt(2*Tz/np.pi)))

## auxiliary functions for loss integration when there is no simple closed form, corresponding to the marginalization of a correlated gaussian ##

def aux_chi1z_minus(r,w):
    w2 = r*w/(np.sqrt(2*(1-r**2)))
    return(special.erfc(w2)/2)

def aux_chi1z_plus(r,w):
    w2 = r*w/(np.sqrt(2*(1-r**2)))
    return((1+special.erf(w2))/2)
    
def aux_m1z_minus(r,Tz,tau,w):
    w2 = r*w/(np.sqrt(2*(1-r**2)))
    return(1/(2*np.sqrt(np.pi))*(-np.exp(-(w2)**2)*np.sqrt(2*Tz*(1-r**2))+np.sqrt(np.pi*Tz)*w*r*(special.erfc(w2))))

def aux_m1z_plus(r,Tz,tau,w):
    w2 = r*w/(np.sqrt(2*(1-r**2)))
    return(1/(2*np.sqrt(np.pi))*(np.exp(-(w2)**2)*np.sqrt(2*Tz*(1-r**2))+np.sqrt(np.pi*Tz)*w*r*(1+special.erf(w2))))
    
    
## hinge loss with noiseless sign teacher lo==2 ##

def m1z_hinge(Q1z,mh1z,Tz,chih1z):
    tau = (mh1z/Q1z)**2*Tz+chih1z/Q1z**2
    c = mh1z/Q1z*Tz
    r = c/(np.sqrt(Tz*tau))
    return(integrate.quad(lambda w : 1/np.sqrt(2*np.pi)*np.exp(-w**2/2)*(prox_hinge(Q1z,-1,w*np.sqrt(tau))*aux_m1z_minus(r,Tz,tau,w)+prox_hinge(Q1z,1,w*np.sqrt(tau))*aux_m1z_plus(r,Tz,tau,w)),-100,100)[0])
    
def chi1z_hinge(Q1z,mh1z,Tz,chih1z):
    tau = (mh1z/Q1z)**2*Tz+chih1z/Q1z**2
    c = mh1z/Q1z*Tz
    r = c/(np.sqrt(Tz*tau))
    return(1/Q1z*integrate.quad(lambda w : 1/np.sqrt(2*np.pi)*np.exp(-w**2/2)*(der_prox_hinge(Q1z,-1,w*np.sqrt(tau))*aux_chi1z_minus(r,w)+der_prox_hinge(Q1z,1,w*np.sqrt(tau))*aux_chi1z_plus(r,w)),-100,100)[0])
    
def q1z_hinge(Q1z,mh1z,Tz,chih1z):
    tau = (mh1z/Q1z)**2*Tz+chih1z/Q1z**2
    c = mh1z/Q1z*Tz
    r = c/(np.sqrt(Tz*tau))
    return(integrate.quad(lambda w :  1/np.sqrt(2*np.pi)*np.exp(-w**2/2)*(prox_hinge(Q1z,-1,w*np.sqrt(tau))**2*aux_chi1z_minus(r,w)+prox_hinge(Q1z,1,w*np.sqrt(tau))**2*aux_chi1z_plus(r,w)),-100,100)[0])
    

## logistic loss with sign teacher lo==3 ##

def m1z_log(Q1z,mh1z,Tz,chih1z):
    tau = (mh1z/Q1z)**2*Tz+chih1z/Q1z**2
    c = mh1z/Q1z*Tz
    r = c/(np.sqrt(Tz*tau))
    return(integrate.quad(lambda w : 1/np.sqrt(2*np.pi)*np.exp(-w**2/2)*(prox_log(Q1z,-1,w*np.sqrt(tau))*aux_m1z_minus(r,Tz,tau,w)+prox_log(Q1z,1,w*np.sqrt(tau))*aux_m1z_plus(r,Tz,tau,w)),-100,100)[0])
    
def chi1z_log(Q1z,mh1z,Tz,chih1z):
    tau = (mh1z/Q1z)**2*Tz+chih1z/Q1z**2
    c = mh1z/Q1z*Tz
    r = c/(np.sqrt(Tz*tau))
    return(1/Q1z*integrate.quad(lambda w : 1/np.sqrt(2*np.pi)*np.exp(-w**2/2)*(der_prox_log(Q1z,-1,w*np.sqrt(tau))*aux_chi1z_minus(r,w)+der_prox_log(Q1z,1,w*np.sqrt(tau))*aux_chi1z_plus(r,w)),-100,100)[0])
    
def q1z_log(Q1z,mh1z,Tz,chih1z):
    tau = (mh1z/Q1z)**2*Tz+chih1z/Q1z**2
    c = mh1z/Q1z*Tz
    r = c/(np.sqrt(Tz*tau))
    return(integrate.quad(lambda w :  1/np.sqrt(2*np.pi)*np.exp(-w**2/2)*(prox_log(Q1z,-1,w*np.sqrt(tau))**2*aux_chi1z_minus(r,w)+prox_log(Q1z,1,w*np.sqrt(tau))**2*aux_chi1z_plus(r,w)),-100,100)[0])
    

##### global functions for m1z,chi1z,q1z #####

def m1z_func(lo,Q1z,Tz,mh1z,delta0,chih1z):
    if lo==0:
        return(m1z_sqloss(mh1z,Tz,Q1z))
    if lo==1:
        return(m1z_sgn_sqloss(Q1z,mh1z,Tz))
    if lo==2:
        return(m1z_hinge(Q1z,mh1z,Tz,chih1z))
    if lo==3:
        return(m1z_log(Q1z,mh1z,Tz,chih1z))
        
def chi1z_func(lo,Q1z,Tz,mh1z,delta0,chih1z):
    if lo==0:
        return(chi1z_sqloss(Q1z))
    if lo==1:
        return(chi1z_sgn_sqloss(Q1z,mh1z,Tz))
    if lo==2:
        return(chi1z_hinge(Q1z,mh1z,Tz,chih1z))
    if lo==3:
        return(chi1z_log(Q1z,mh1z,Tz,chih1z))

def q1z_func(lo,Q1z,Tz,mh1z,delta0,chih1z):
    if lo==0:
        return(q1z_sqloss(Q1z,Tz,mh1z,delta0,chih1z))
    if lo==1:
        return(q1z_sgn_sqloss(Q1z,mh1z,Tz,chih1z))
    if lo==2:
        return(q1z_hinge(Q1z,mh1z,Tz,chih1z))
    if lo==3:
        return(q1z_log(Q1z,mh1z,Tz,chih1z))
        
#### Stieltje-like integrals. Remark : this step could likely be simplified by spotting the common terms in each integral and reuse those already calculated. d==0 is an arbitrary continuous spectrum like MP or squared uniform. d==1 is for row orthogonal matrices and is unstable, heavy damping is needed. ####

def esp_vap(f,asp,d):
    if d==0:
        return(integrate.quad(lambda p : min(1,asp)*p*f(p),0,np.inf)[0])
    if d==1:
        return(min(1,asp))

def e1(f,asp,Tx,mh2x,mh2z,Q2x,Q2z,d):
    if d==0:
        return(Tx*(max(0,1-asp)*mh2x/Q2x+integrate.quad(lambda p: min(1,asp)*f(p)*(mh2x+p*mh2z)/(Q2x+p*Q2z),0,np.inf)[0]))
    if d==1:
        return(Tx*(max(0,1-asp)*mh2x/Q2x+min(1,asp)*(mh2x+mh2z)/(Q2x+Q2z)))

def e2(f,asp,Q2x,Q2z,d):
    if d==0:
        return(max(0,1-asp)*1/Q2x+min(1,asp)*integrate.quad(lambda p : f(p)*1/(Q2x+p*Q2z),0,np.inf)[0])
    if d==1:
        return(max(0,1-asp)*1/Q2x+min(1,asp)*1/(Q2x+Q2z))
    
def e3(f,asp,chih2x,chih2z,Q2x,Q2z,d):
    if d==0:
        return(max(0,1-asp)*chih2x/Q2x**2+min(1,asp)*integrate.quad(lambda p :f(p)*(chih2x+p*chih2z)/(Q2x+p*Q2z)**2,0,np.inf)[0])
    if d==1:
        return(max(0,1-asp)*chih2x/Q2x**2+min(1,asp)*(chih2x+chih2z)/(Q2x+Q2z)**2)
    
def e4(f,asp,mh2x,mh2z,Q2x,Q2z,Tx,d):
    if d==0:
        return(Tx*(max(0,1-asp)*(mh2x/Q2x)**2+min(1,asp)*integrate.quad(lambda p :f(p)*(mh2x+p*mh2z)**2/(Q2x+p*Q2z)**2,0,np.inf)[0]))
    if d==1:
        return(Tx*(max(0,1-asp)*(mh2x/Q2x)**2+min(1,asp)*(mh2x+mh2z)**2/(Q2x+Q2z)**2))

def e5(f,asp,mh2x,mh2z,Q2x,Q2z,Tx,d):
    if d==0:
        return(Tx/asp*min(1,asp)*integrate.quad(lambda p :f(p)*(p*(mh2x+p*mh2z)/(Q2x+p*Q2z)),0,np.inf)[0])
    if d==1:
        return(Tx/asp*min(1,asp)*(mh2x+mh2z)/(Q2x+Q2z))

def e6(f,asp,Q2x,Q2z,d):
    if d==0:
        return(1/asp*min(1,asp)*integrate.quad(lambda p : f(p)*p/(Q2x+p*Q2z),0,np.inf)[0])
    if d==1:
        return(1/asp*min(1,asp)*1/(Q2x+Q2z))
    
def e7(f,asp,chih2x,chih2z,Q2x,Q2z,d):
    if d==0:
        return(1/asp*min(1,asp)*integrate.quad(lambda p :f(p)*(p*(chih2x+p*chih2z)/(Q2x+p*Q2z)**2),0,np.inf)[0])
    if d==1:
        return(1/asp*min(1,asp)*(chih2x+chih2z)/(Q2x+Q2z)**2)

def e8(f,asp,mh2x,mh2z,Q2x,Q2z,Tx,d):
    if d==0:
        return(Tx/asp*min(1,asp)*integrate.quad(lambda p : f(p)*p*(mh2x+p*mh2z)**2/(Q2x+p*Q2z)**2,0,np.inf)[0])
    if d==1:
        return(Tx/asp*min(1,asp)*(mh2x+mh2z)**2/(Q2x+Q2z)**2)
        
##### one iteration of SE Kabashima style (i.e. equation block (90) in a different order see the original paper by Takahashi and Kabashima #####

def SE_it(lo,l1,l2,rho,delta0,Q1x,Q1z,mh1x,mh1z,chih1x,chih1z,Q2x,Q2z,mh2x,mh2z,chih2x,chih2z,Tx,Tz,f,asp,damp,d):
    
        m1x = m1x_func(rho,Q1x,mh1x,chih1x,l1,l2)
        chi1x = chi1x_func(rho,Q1x,mh1x,chih1x,l1,l2)
        q1x = q1x_func(rho,Q1x,mh1x,chih1x,l1,l2)
        m1z = m1z_func(lo,Q1z,Tz,mh1z,delta0,chih1z)
        chi1z = chi1z_func(lo,Q1z,Tz,mh1z,delta0,chih1z)
        q1z = q1z_func(lo,Q1z,Tz,mh1z,delta0,chih1z)
        
        Q2x_new = damp*Q2x+(1-damp)*(1/chi1x-Q1x)
        Q2z_new = damp*Q2z+(1-damp)*(1/chi1z-Q1z)
        mh2x_new = damp*mh2x+(1-damp)*(m1x/(Tx*chi1x)-mh1x)
        mh2z_new = damp*mh2z+(1-damp)*(m1z/(Tz*chi1z)-mh1z)
        chih2x_new = damp*chih2x+(1-damp)*(q1x/(chi1x**2)-(m1x/chi1x)**2/Tx-chih1x)
        chih2z_new = damp*chih2z+(1-damp)*(q1z/(chi1z**2)-(m1z/chi1z)**2/Tz-chih1z)
        
        m2x = e1(f,asp,Tx,mh2x,mh2z,Q2x,Q2z,d)
        chi2x = e2(f,asp,Q2x,Q2z,d)
        q2x = e3(f,asp,chih2x,chih2z,Q2x,Q2z,d)+e4(f,asp,mh2x,mh2z,Q2x,Q2z,Tx,d)
        m2z = e5(f,asp,mh2x,mh2z,Q2x,Q2z,Tx,d)
        chi2z = e6(f,asp,Q2x,Q2z,d)
        q2z = e7(f,asp,chih2x,chih2z,Q2x,Q2z,d)+e8(f,asp,mh2x,mh2z,Q2x,Q2z,Tx,d)
        
        Q1x_new = damp*Q1x+(1-damp)*(1/chi2x-Q2x)
        Q1z_new = damp*Q1z+(1-damp)*(1/chi2z-Q2z)
        mh1x_new = damp*mh1x+(1-damp)*(m2x/(Tx*chi2x)-mh2x)
        mh1z_new = damp*mh1z+(1-damp)*(m2z/(Tz*chi2z)-mh2z)
        chih1x_new = damp*chih1x+(1-damp)*(q2x/chi2x**2-(m2x/chi2x)**2/Tx-chih2x)
        chih1z_new = damp*chih1z+(1-damp)*(q2z/chi2z**2-(m2z/chi2z)**2/Tz-chih2z)
        
        E = Tx-2*m1x+q1x
        ang = np.arccos(m1x/np.sqrt(Tx*q1x))
        
        #if max(Q2x_new,Q2z_new,chih2x_new,chih2z_new,Q1x_new,Q1z_new,chih1x_new,chih1z_new)<=0:
            #damp = 1-0.9*(1-damp)
            #print("changed damping",damp)
            #Q1x_new,Q1z_new,mh1x_new,mh1z_new,chih1x_new,chih1z_new,Q2x_new,Q2z_new,mh2x_new,mh2z_new,chih2x_new,chih2z_new = Q1x,Q1z,mh1x,mh1z,chih1x,chih1z,Q2x,Q2z,mh2x,mh2z,chih2x,chih2z
        #print(Q1x_new,Q1z_new,mh1x_new,mh1z_new,chih1x_new,chih1z_new,Q2x_new,Q2z_new,mh2x_new,mh2z_new,chih2x_new,chih2z_new)
        return(Q1x_new,Q1z_new,mh1x_new,mh1z_new,chih1x_new,chih1z_new,Q2x_new,Q2z_new,mh2x_new,mh2z_new,chih2x_new,chih2z_new,E,ang)
    
##### iterating the SE equations until target precision is reached #####

def SE(lo,l1,l2,rho,delta0,f,asp,damp,niter,tol,d,cooling,l2_c):

    Tx = rho
    Tz = 1/asp*esp_vap(f,asp,d)*Tx
    E = 1
    ang = 1
    conv = 1
    c = 0
    Q1x,Q1z,mh1x,mh1z,chih1x,chih1z,Q2x,Q2z,mh2x,mh2z,chih2x,chih2z = np.ones(12)
    l2_init = l2
    
    while c<niter and conv>tol:
    
        Q1x_new,Q1z_new,mh1x_new,mh1z_new,chih1x_new,chih1z_new,Q2x_new,Q2z_new,mh2x_new,mh2z_new,chih2x_new,chih2z_new,E_new,ang_new = SE_it(lo,l1,l2,rho,delta0,Q1x,Q1z,mh1x,mh1z,chih1x,chih1z,Q2x,Q2z,mh2x,mh2z,chih2x,chih2z,Tx,Tz,f,asp,damp,d)
        conv = max(np.abs((E-E_new)/E),np.abs((ang-ang_new)/ang))
        Q1x,Q1z,mh1x,mh1z,chih1x,chih1z,Q2x,Q2z,mh2x,mh2z,chih2x,chih2z,E,ang = Q1x_new,Q1z_new,mh1x_new,mh1z_new,chih1x_new,chih1z_new,Q2x_new,Q2z_new,mh2x_new,mh2z_new,chih2x_new,chih2z_new,E_new,ang_new
        print("iteration",c,"prec",conv)
        c = c+1
        if cooling == 1:
            l2 = l2_init+(0.95)**c*l2_c
            print("cooled_l2",l2)
    
    return(E,ang,conv)
        
        
    
        
        
        



    
    
