import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
''' Functions for importing'''


# this is the ii transfer function 
def phi(x,nu,theta,uc):
	myphi=nu*(x-theta)
	myphi[x>uc]=nu*(uc-theta)
	myphi[theta>x]=0.
	return myphi

def phi_brunel(x,nu_brunel,theta,uc):
	myphi=nu_brunel*((x-theta)/uc)**2
	myphi[x>(uc+theta)]=2*nu_brunel*np.sqrt((x[x>(uc+theta)]-theta)/uc-3./4.)
	myphi[theta>x]=0.
	return myphi

def phi_tanh(x,a1,b1):
	return 0.5*(1+np.tanh(a1*(x+b1)))

