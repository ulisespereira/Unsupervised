import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
from stimulus import *
from myintegrator import *
import cProfile
import json
import scipy.integrate as integrate

# this is the transfer function 
def phi(x,theta,uc):
	myresult=nu*(x-theta)
	myresult[x<theta]=0.
	myresult[x>uc]=nu*(uc-theta)
	return myresult

def phi_tanh(x):
	return 0.5*(1+np.tanh(a1*(x+b1)))

def mytauInv(x): #time scale function synapses
	myresult=np.zeros(len(x))
	myresult[x>thres]=1/tau_learning
	return myresult

def winf(x_hist):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	mynu=5.5
	mytheta=-0.8
	#parameters
	n=len(pre_u)
	vec_pre=0.5*(np.ones(n)+np.tanh(a_pre*pre_u+b_pre))
	vec_post=0.5*(np.ones(n)+np.tanh(a_post*post_u+b_post))
	#return (wmax/2.)*np.outer((np.ones(n)+np.tanh(a_post*post_u+b_post)),vec_pre)
	return wmax*np.outer(vec_post,vec_pre)

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest

def tauWinv(x_hist):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	#return  np.add.outer(1/mytau(post_u),1/mytau(pre_u))
	return  tau_learning*np.outer(mytauInv(post_u),mytauInv(pre_u))
def F(u):
	return np.sqrt(wmax)*.5*(1.+np.tanh(a_post*u+b_post))

def field(t,a,x_hist,W,H):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	conn_matrix=(W.T*H).T
	field_u=(1/tau)*(mystim.stim(t)+conn_matrix.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi(x_hist[-1],theta,uc)))#-a
	field_a=0.#in the paper we are not using adaptation during learning
	field_H=(H*(1.-(post_u/y0))-H**2)/tau_H
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W)
	return field_a,field_u,field_w,field_H

#This are a the parameters of the simulation

#open parameters of the model
n=20 #n pop
delay=15.3
tau=10.   #timescale of populations
tau_H=10000.
y0=.12*np.ones(n)
w_i=1.
w_inh=w_i/n
nu=1.
theta=0.
uc=1.
wmax=2.40
thres=0.9
beta=1.6
tau_a=10.
#parameters stimulation
dt=0.5
lagStim=100.
times=1
amp=5.




a_post=1.
b_post=-2.25
a_pre=1.0
b_pre=-2.25
tau_learning=400.

a1=6.
b1=-0.25



#---------------------------------------------------------------
#----------------Learning vector Field--------------------------
#---------------------------------------------------------------


def tau_u0_theta(T):
	return -tau*np.log(1.-(thres/amp))

def tau_umax_theta(T):
	return -tau*np.log((thres/amp)*(1./(1.-np.exp(-T/tau))))
def tau_theta(T):
	return T-tau_u0_theta(T)+tau_umax_theta(T)

#approximation pupulation dynamics
def myu(t,T,tstart):
	ttilda=t-tstart
	if tau_u0_theta(T)<=ttilda and ttilda<=T:
		return amp*(1.-np.exp(-ttilda/tau))
	elif ttilda>T:
		return amp*(1.-np.exp(-T/tau))*np.exp(-(ttilda-T)/tau)
	elif tau_u0_theta(T)>ttilda:
		return amp*(1.-np.exp(-ttilda/tau))
	elif ttilda<0:
		print 'holi',ttilda


def recurrentTheo(t,T,tstart):
	if tstart+delay+tau_u0_theta(T)<=t and t<=tstart+T+tau_umax_theta(T):
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart))*np.exp((x-delay-tstart-tau_u0_theta(T))/tau_learning) 
		myintegral=lambda y:integrate.quad(df,tstart+delay+tau_u0_theta(T),y)
		val,err=myintegral(t)
		return np.exp(-(t-(delay+tau_u0_theta(T)+tstart))/tau_learning)*(w0+val*(1./tau_learning))
	
	elif t>tstart+T+tau_umax_theta(T):
		myt=tstart+T+tau_umax_theta(T)
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart))*np.exp((x-delay-tstart-tau_u0_theta(T))/tau_learning) 
		myintegral=lambda y:integrate.quad(df,tstart+delay+tau_u0_theta(T),y)
		val,err=myintegral(myt)
		return np.exp(-(myt-(delay+tau_u0_theta(T)+tstart))/tau_learning)*(w0+val*(1./tau_learning))
	else:
		return w0

def feedforwardTheo(t,T,delta,tstart):
	if tstart+tau_u0_theta(T)<=t and t<=tstart+delay-delta+tau_umax_theta(T):
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart-delta-T))*np.exp((x-tau_u0_theta(T)-tstart)/tau_learning) 
		myintegral=lambda y:integrate.quad(df,tau_u0_theta(T)+tstart,y)
		val,err=myintegral(t)
		return np.exp(-(t-(tstart+tau_u0_theta(T)))/tau_learning)*(w0+val*(1./tau_learning))
	
	elif t>tstart+delay-delta+tau_umax_theta(T):
		myt=tstart+delay-delta+tau_umax_theta(T)
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart-delta-T))*np.exp((x-tau_u0_theta(T)-tstart)/tau_learning) 
		myintegral=lambda y:integrate.quad(df,tau_u0_theta(T)+tstart,y)
		val,err=myintegral(myt)
		return np.exp(-(myt-(tstart+tau_u0_theta(T)))/tau_learning)*(w0+val*(1./tau_learning))
	else:
		return w0


#-------------------------------------------------------------------
#-----------------Stimulation of Populations------------------------
#-------------------------------------------------------------------

# setting up the simulation 

period=40.
delta=8

r1_matrix=np.ones((n,n))
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
npts=int(np.floor(delay/dt)+1)         # points delay

#initial conditions
w0=0.1
a0=np.zeros((npts,n))
x0=0.01*np.ones((npts,n))
W0=[w0*np.ones((n,n)) for i in range(npts)]
H0=[0.01*np.ones(n) for i in range(npts)]

mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp

#integrator

#tmax=times*(lagStim+n*(period+delta))+mystim.delay_begin
tmax=300.
tstart1=mystim.delay_begin+period+delta
tstart2=mystim.delay_begin+2*period+2*delta
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False
adapt,u,connectivity,W01,myH,t=theintegrator.DDE_Norm_Miller(field,a0,x0,W0,H0)
theo_u1=np.array([myu(myt,period,tstart1) for myt in t])
theo_u2=np.array([myu(myt,period,tstart2) for myt in t])
theo_w=np.array([recurrentTheo(myt,period,tstart1) for myt in t])
theo_s=np.array([feedforwardTheo(myt,period,delta,tstart2) for myt in t])
# figrue 1

rc={'axes.labelsize': 30, 'font.size': 30, 'legend.fontsize': 28.0, 'axes.titlesize': 30}
plt.rcParams.update(**rc)



figure=plt.figure(figsize=(25,10))
recurrentFig1=figure.add_subplot(121)
recurrentFig1.plot(t,u[:,1],'g',lw=6)
recurrentFig1.plot(t,u[:,2],'b',lw=6)
#recurrentFig1.plot(t,thres*np.ones(len(u[:,1])),'r',lw=3)
recurrentFig1.plot(t,theo_u1,'r--',lw=6,alpha=0.8)
recurrentFig1.plot(t,theo_u2,'r--',lw=6,alpha=0.8)
recurrentFig1.set_xlabel('Time (ms)',size=28)
recurrentFig1.set_ylabel(r'$u$',size=28)
recurrentFig1.set_xticks([100,150,200])
recurrentFig1.set_yticks([1.,2,3.,4,5.])
recurrentFig1.set_ylim([thres,5])
recurrentFig1.set_xlim([90,210])
recurrentFig1.set_title('(A)')

recurrentFig2=figure.add_subplot(122)
recurrentFig2.plot(t,connectivity[:,1,1],'c',lw=5)
recurrentFig2.plot(t,theo_w,'r--',lw=6,alpha=0.8)
recurrentFig2.set_ylabel('Synaptic Weight',size=28)
recurrentFig2.set_xlabel('Time (ms)',size=28)
recurrentFig2.set_xticks([100,150,200])
recurrentFig2.set_yticks([0.1,0.2,0.3])
recurrentFig2.set_ylim([0.1,0.3])
recurrentFig2.set_xlim([100,200])
recurrentFig2.set_title('(B)')

#recurrentFig3=figure.add_subplot(133)
recurrentFig2.plot(t,connectivity[:,2,1],'y',lw=7)
recurrentFig2.plot(t,theo_s,'r--',lw=6,alpha=0.8)
#recurrentFig3.set_ylabel('Synaptic Weight',size=28)
#recurrentFig3.set_xlabel('Time (ms)',size=28)
#recurrentFig3.set_xticks([140,160,180,200])
#recurrentFig3.set_yticks([0.1,0.12,0.14,0.16])
#recurrentFig3.set_ylim([.1,0.16])
#recurrentFig3.set_xlim([140,200])

plt.savefig('recurrent_test.pdf', bbox_inches='tight')
plt.show()





