import numpy as np
import scipy as scp
import scipy.integrate as integrate
from scipy.interpolate import interp1d

# Script to inverse a stieltjes transform based on the Bai-Silverstein formula, here written with a uniform distribution on an interval [a,b] for the spectrum of the covariance matrix

def I1(A,B,a,b):
    res = integrate.quad(lambda t : t*(1+t*A)/((1+t*A)**2+(t*B)**2)*1/(b-a),a,b)[0]
    return(res)

def I2(A,B,a,b):
    res = integrate.quad(lambda t : t*(t*B)/((1+t*A)**2+(t*B)**2)*1/(b-a),a,b)[0]
    return(res)

def inverse_stieltjes(l,a,b,asp,w,niter_max,damp):
    A = 1
    B = 1
    conv = 1
    niter=0
    rho = damp
    
    while conv>1e-6 and niter<niter_max:

        A_new = rho*(-l+asp*I1(A,B,a,b))/((l-asp*I1(A,B,a,b))**2+(w+asp*I2(A,B,a,b))**2)+(1-rho)*A
        B_new = rho*(w+asp*I2(A,B,a,b))/((l-asp*I1(A,B,a,b))**2+(w+asp*I2(A,B,a,b))**2)+(1-rho)*B

        conv = max(abs(A_new-A),abs(B_new-B))

        A = A_new
        B = B_new
        niter+=1

    return(niter,A,B)


def mu(a,b,asp,w,niter_max,damp,x_start,x_stop,prec):
    
    mu_vec = np.zeros(prec)
    x_vec = np.linspace(x_start,x_stop,prec)
    
    for i in range(prec):
        n,A,B = inverse_stieltjes(x_vec[i],a,b,asp,w,niter_max,rho)
        mu_vec[i] = B/np.pi
     
    f = interp1d(x_vec,mu_vec)
    
    return(mu_vec,f)
