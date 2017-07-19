import numpy as np
import math as mt
import matplotlib.pyplot as plt
#This are a the parameters of the simulation

#parameters single neuron
n=10 #n pop
tau=10.  #timescale of populations
nu=1. #slope tf
theta=0. #lower thres tf
uc=1. # upper thres tf
# transfer functions
a1=2.
b1=0.5
a1_inh=3.2
b1_inh=0.15



#parameters inhibition
wIE=1.

#parameters homeostatic plasticity
r0=.065*np.ones(n)
tau_WEI=100. #fastest?

#parameters stimulation
dt=0.5
lagStim=400. # before presentation
times=230 #times stimulation
amp=3.5 # amplitde current 
delta=31.3 # time between stim
period=20. # peridod stim

# parameters GHLR f=g
delay=15.3 # delay pre-post
wmax=1.6 # max learned
thres=0.6
bf=10. #slope
xf=0.7 # threshold
a_post=bf
b_post=xf
a_pre=bf
b_pre=xf
tau_learning=400.#30000.






#------------------------------------------------------------------
#-------------------Functions dynamics-----------------------------
#------------------------------------------------------------------

# this is the transfer function 
def phi(x,theta,uc):
	myresult=nu*(x-theta)
	myresult[x<theta]=0.
	myresult[x>uc]=nu*(uc-theta)
	return myresult

def phi_tanh(x):
	return 0.5*(1+np.tanh(a1*(x-b1)))

def phi_tanh_inh(x):
	return 0.5*(1+np.tanh(a1_inh*(x-b1_inh)))

def mytauInv(x): #time scale function synapses
	myresult=np.zeros(len(x))
	myresult[x>thres]=1/tau_learning
	return myresult

def winf(x_hist):
	pre_u=phi_tanh(x_hist[0])
	post_u=phi_tanh(x_hist[-1])
	#parameters
	n=len(pre_u)
	vec_pre=0.5*(np.ones(n)+np.tanh(a_pre*(pre_u-b_pre)))
	return (wmax/2.)*np.outer((np.ones(n)+np.tanh(a_post*(post_u-b_post))),vec_pre)

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest
def tauWinv(x_hist):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	#return  np.add.outer(1/mytau(post_u),1/mytau(pre_u))
	return  tau_learning*np.outer(mytauInv(post_u),mytauInv(pre_u))

def field(t,x_hist,uI,W,WEI,stim):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	field_u=(1./tau)*(stim(t)+W.dot(phi_tanh(x_hist[-1]))-x_hist[-1]-WEI*phi_tanh_inh(uI))# dynamics recurrent neurons
	field_uI=(1./(tau/2.))*(-uI+wIE*np.mean(phi_tanh(x_hist[-1]))) # uniform weight IE
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W) # rec connections
	field_WEI=(1./tau_WEI)*phi_tanh_inh(uI)*(phi_tanh(x_hist[-1])-r0) # inhibitiory plastisicity
	return field_u,field_uI,field_w,field_WEI

#x=np.linspace(-1,2,100)
#plt.plot(x,phi(x,0,1))
#plt.plot(x,phi_tanh(x))
#plt.show()
