# Toolbox 

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import ortho_group
from scipy import special
from sklearn.svm import LinearSVC
from sklearn import linear_model
from scipy import optimize

    
    
#Useful fonctions for synthetic data generation

#Sampling of Haar matrices of size N, faster than scipy ortho_group and sufficient for considered setup. Based on same method as scipy ortho_group, see scipy userguide.
def Haar_sample(n):
    #We first generate a N x N orthogonal matrix by taking the QR decomposition of a random N x N gaussian matrix
    gaussian_matrix = np.random.normal(0, 1., (N,N))
    O, R = np.linalg.qr(gaussian_matrix)
    #Then we multiply on the right by the signs (or phases in the complex case) of the diagonal of R to really get the Haar measure
    D = np.diagonal(R)
    #Normalization
    Lambda = D / np.abs(D)   
    O = np.multiply(O, Lambda)
    return(O)
    
#Gauss-Bernoulli distribution on the ground-truth vector x0 with sparsity rho
def gauss_bernoulli(rho,N):
	x = np.zeros(N)
	for i in range(N):
		indicator = np.random.random()
		if indicator > rho :
			x[i] = 0
		if indicator <= rho : 
			x[i] = np.random.randn()	
	return(x)


#Useful eigenvalue distributions

#Marcenko-Pastur density
#Careful, scaling (aspect ratio) is done when integrating, the following density is expected to be normalized whatever the aspect ratio.
def MP(x,alpha):
	a = (1-np.sqrt(1/alpha))**2
	b = (1+np.sqrt(1/alpha))**2
	return(max(1,alpha)*np.sqrt(max(0,(x-a))*max(0,b-x))/(2*np.pi*x))

#Rescaled uniform density
def unif(x,alpha):

    a = (1-alpha)**2
    b = (1+alpha)**2
    
    if a<=np.sqrt(x)<=b:
        return(1/(2*(b-a))*1/np.sqrt(x))
        
    else:
        return(0)

#Building orthogonally invariant random matrix with squared uniform eigenvalue distribution

def build_matrix(a,b,M,N,select):	
	V = Haar_sample(M)
	O = Haar_sample(N)
	if select == 0:
		eigen = np.random.uniform(a,b,min(M,N))
	if select == 1:
		eigen = np.ones(min(M,N))
	D_prev = np.diag(eigen)
	comp = np.zeros((max(M,N)-min(M,N),min(M,N)))	
	if M>=N :
		D = np.concatenate((D_prev,comp),0)
	if M<N:
		D = np.concatenate((D_prev,np.transpose(comp)),1)
	F = V@D@np.transpose(O)
	return(F,D)
 

#Proximals

#Proximal of squared loss:
def prox_squared(h, Q, y):
    return (Q/(1+Q))*h+1/(1+Q)*y
    

def der_prox_squared(h, Q, y):
    return Q/(1+Q)


#Proximal of the hinge loss:
def prox_hinge(h, Q, y):

    res=np.zeros(len(h))
    for i in range(len(h)):
        flag = Q*(1-y[i]*h[i])

        if flag>=1:
            res[i]=h[i]+y[i]/Q
    
        if 0<=flag<=1:
            res[i]=y[i]

        if flag<=0:
            res[i]=h[i]

    return res

def der_prox_hinge(h, Q, y):
    res=np.zeros(len(h))
    for i in range(len(h)):
        flag = Q*(1-y[i]*h[i])

        if flag>=1:
            res[i]=1
    
        if 0<=flag<=1:
            res[i]=0

        if flag<=0:
            res[i]=1

    return res
    
#Proximal of logistic loss (no closed form)
def prox_log(h, Q, y):
    res=np.zeros(len(h))
    for i in range(len(h)):
        res[i] = np.float(optimize.fsolve(lambda x: x-h[i]-y[i]/Q*1/(1+np.exp(y[i]*np.float(x))),1))
    return res
    
def der_prox_log(h, Q, y):
    res=1/(1+1/(2*Q)*1/(1+np.cosh(y*prox_log(h, Q, y))))
    for i in range(len(res)):
        res[i] = np.float(res[i])
    return res

#Unified functions
def prox(h, Q, y, loss):
    if loss=='squared':
        return prox_squared(h,Q,y)
    elif loss=='hinge':
        return prox_hinge(h,Q,y)
    elif loss=='logistic':
        return prox_log(h,Q,y)

def der_prox(h, Q, y, loss):
    if loss=='squared':
        return der_prox_squared(h,Q,y)
    elif loss=='hinge':
        return der_prox_hinge(h,Q,y)
    elif loss=='logistic':
        return der_prox_log(h,Q,y)


#Soft-thresholding

def soft_thresh(x,thresh):
    n = len(x)
    res = np.zeros(n)
    for i in range(n):
        if np.abs(x[i])>thresh:
            res[i] = x[i]-np.sign(x[i]-thresh)*thresh
        if np.abs(x[i])<=thresh :
            res[i] = 0
    return(res)

def soft_thresh_der(x,thresh):
    n = len(x)
    res = np.zeros(n)
    for i in range(n):
        if np.abs(x[i])>thresh:
            res[i] = 1
        if np.abs(x[i])<=thresh :
            res[i] = 0
    return(res)

#GVAMP    
def GVAMP(F, y, l1, l2, loss='squared', damp=0, nb_iterations=100, tol=1e-15):

    #Initialization
    M,N = np.shape(F)
    Q1x_0=1
    Q2z_0=1
    h1x_0=np.random.rand(N)
    h2z_0=np.random.rand(M)

    conv_vec = np.zeros(nb_iterations)
    c = 0

    #Algorithm:
    while c<nb_iterations:

        #First x block
        x1 = 1/(1+l2/Q1x_0)*soft_thresh(h1x_0,l1/Q1x_0)
        chi1x= 1/(1+l2/Q1x_0)*np.mean(soft_thresh_der(h1x_0,l1/Q1x_0))/Q1x_0
        Q2x=1/chi1x - Q1x_0
        h2x=1/Q2x*(x1/chi1x - Q1x_0*h1x_0)

        #First z block
        A=np.linalg.inv((Q2z_0*np.transpose(F)@F + Q2x*np.eye(N)))
        z2 = F@A@(Q2x*h2x+Q2z_0*np.transpose(F)@h2z_0)
        chi2z=1/M*np.trace(F@A@np.transpose(F))
        Q1z=1/chi2z-Q2z_0
        h1z=1/Q1z*(z2/chi2z-Q2z_0*h2z_0) 

        #Second z block
        z1=prox(h1z, Q1z, y, loss)
        chi1z=1/Q1z*np.mean(der_prox(h1z, Q1z, y, loss))
        Q2z=Q2z_0*damp+(1/chi1z-Q1z)*(1-damp)
        h2z=1/Q2z*(z1/chi1z-Q1z*h1z)

        #Second x block
        A2=np.linalg.inv((Q2z*np.transpose(F)@F+Q2x*np.eye(N)))
        x2=A2@(Q2x*h2x+Q2z*np.dot(np.transpose(F),h2z))
        chi2x=1/N*np.trace(A2)
        Q1x=Q1x_0*damp+(1/chi2x-Q2x)*(1-damp)
        h1x=1/Q1x*(x2/chi2x-Q2x*h2x)

        #Convergence control:
        conv_vec[c] = 1/N*np.linalg.norm(h1x-h1x_0)
        Q1x_0 = Q1x
        Q2z_0 = Q2z
        h1x_0 = h1x
        h2z_0 = h2z
        if conv_vec[c]<tol:
            break
        c = c+1

    return x1, conv_vec 



