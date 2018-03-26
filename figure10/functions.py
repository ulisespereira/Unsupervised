import numpy as np
import math as mt
from stimulus import *
from myintegrator import *
import scipy.integrate as integrate

class Network:
	''' Network class for homeostatic plasticity'''
	def __init__(self):#Global parameters needed
		# single neuron
		self.n=10 #n pop
		self.tau=10.   #timescale of populations
		self.nu=1. # slope PWL TF
		self.theta=0. # lower threshold
		self.uc=1. #upper threshold

		# Learning Rule
		self.wmax=1.8 # maximum learned
		self.thres=0.6 # threshold
		self.delay=15.3 # delay
		self.w0=0.01 # initial condition dynamics
		bf=10. # slope f and g
		xf=0.7 # threshold f and g
		self.a_post=bf
		self.b_post=xf
		self.a_pre=bf
		self.b_pre=xf
		self.tau_learning=400. # time scale LR

		# stimulus
		self.amp=5.5 #amplitude
		self.patterns=np.identity(self.n) # stimulation matrix
		self.patterns=[self.patterns[:,i] for i in range(self.n)]  # puting in a list
		self.dt=0.5 # dt dinamics
		self.npts=int(np.floor(self.delay/self.dt)+1)         # points delay
		self.lagStim=100. # lag before stimulation
		self.delta=10. # period bt stim
		self.T=10. # period stim
		self.times=150
		self.mystim=stimulus(self.patterns,self.lagStim,self.delta,self.T,self.times)
		self.mystim.inten=self.amp

		#Homeostatic LR
		self.tau_H=200000. # tau
		self.y0=.12*np.ones(self.n) # target rate

		#inhibition
		self.w_i=2.
		
		#if there is noise in the simulations
		self.amp_noise = 0
	#----------------------------------------------------------------------------
	#--------------------Basic functions ----------------------------------------
	#----------------------------------------------------------------------------
	def setStim(self,amp,delta,T,times):
		self.amp=amp #amplitude
		self.delta=delta # period bt stim
		self.T=T # period stim
		self.times=times
		self.mystim=stimulus(self.patterns,self.lagStim,self.delta,self.T,self.times)
		self.mystim.inten=self.amp


	# this is the transfer function 
	def phi(self,x):
		myresult=self.nu*(x-self.theta)
		myresult[x<self.theta]=0.
		myresult[x>self.uc]=self.nu*(self.uc-self.theta)
		return myresult

	def mytauInv(self,x): #time scale function synapses
		myresult=np.zeros(len(x))
		myresult[x>self.thres]=(1/self.tau_learning)#*0.5*(1+np.tanh(a_tau*(x[x>thres]+b_tau)))
		return myresult

	def winf(self,x_hist):
		pre_u=self.phi(x_hist[0])
		post_u=self.phi(x_hist[-1])
		#parameters
		n=len(pre_u)
		vec_pre=0.5*(np.ones(n)+np.tanh(self.a_pre*(pre_u-self.b_pre)))
		return (self.wmax/2.)*np.outer((np.ones(n)+np.tanh(self.a_post*(post_u-self.b_post))),vec_pre)

	#function for the field
	#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest
	def tauWinv(self,x_hist):
		pre_u=x_hist[0]
		post_u=x_hist[-1]
		n=len(pre_u)
		return self. tau_learning*np.outer(self.mytauInv(post_u),self.mytauInv(pre_u))
	# function for  convenience
	def F(self,u):
		if self.theta<=u and u<=self.uc:
			r=self.nu*(u-self.theta)
		elif u<=self.theta:
			r=0
		elif self.uc<=u:
			r=self.nu*(self.uc-self.theta)
		return np.sqrt(self.wmax)*.5*(1.+np.tanh(self.a_post*(r-self.b_post)))

	def field(self,t,x_hist,W,H,stim):
		pre_u=x_hist[0]
		post_u=x_hist[-1]
		n=len(pre_u)
		conn_matrix=(W.T*H).T
		noise = self.amp_noise * np.random.normal(0,1.,n) * np.sqrt(self.tau/self.dt)
		field_u=(1./self.tau)*(stim(t)+conn_matrix.dot(self.phi(x_hist[-1]))-x_hist[-1]-self.w_i*np.mean(self.phi(x_hist[-1])))
		field_H=(H*(1.-(self.phi(post_u)/self.y0))-H**2)/self.tau_H
		field_w=np.multiply(self.tauWinv(x_hist),self.winf(x_hist)-W)
		return field_u,field_w,field_H

	#---------------------------------------------------------------
	#----------------Learning vector Field--------------------------
	#---------------------------------------------------------------

	def tau_u0_theta(self,T):
		return -self.tau*np.log(1.-(self.thres/self.amp))

	def tau_umax_theta(self,T):
		return -self.tau*np.log((self.thres/self.amp)*(1./(1.-np.exp(-T/self.tau))))
	
	def tau_theta(T):
		return T-self.tau_u0_theta(T)+self.tau_umax_theta(T)

	#approximation pupulation dynamics
	def myu(self,t,T,tstart):
		ttilda=t-tstart
		if self.tau_u0_theta(T)<=ttilda and ttilda<=T:
			return self.amp*(1.-np.exp(-ttilda/self.tau))
		elif ttilda>T:
			return self.amp*(1.-np.exp(-T/self.tau))*np.exp(-(ttilda-T)/self.tau)
		elif self.tau_u0_theta(T)>ttilda:
			return self.amp*(1.-np.exp(-ttilda/self.tau))
		elif ttilda<0:
			print 'holi',ttilda

	def recurrentTheo(self,T,k):
		tstart=0.
		myt=tstart+T+self.tau_umax_theta(T)
		wdyn=[self.w0]
		wk=self.w0
		for i in range(k):
			df=lambda x:self.F(self.myu(x,T,tstart))*self.F(self.myu(x-self.delay,T,tstart))*np.exp((x-self.delay-tstart-self.tau_u0_theta(T))/self.tau_learning) 
			myintegral=lambda y:integrate.quad(df,tstart+self.delay+self.tau_u0_theta(T),y,epsabs=1e-5)
			val,err=myintegral(myt)
			wk=np.exp(-(myt-(self.delay+self.tau_u0_theta(T)+tstart))/self.tau_learning)*(wk+val*(1./self.tau_learning))
			wdyn.append(wk)
		return wdyn

	def feedforwardTheo(self,T,delta,k):
		tstart=0.
		myt=tstart+self.delay-delta+self.tau_umax_theta(T)
		wdyn=[self.w0]
		wk=self.w0
		for i in range(k):
			df=lambda x:self.F(self.myu(x,T,tstart))*self.F(self.myu(x-self.delay,T,tstart-self.delta-T))*np.exp((x-self.tau_u0_theta(T)-tstart)/self.tau_learning) 
			myintegral=lambda y:integrate.quad(df,self.tau_u0_theta(T)+tstart,y,epsabs=1e-5)
			val,err=myintegral(myt)
			wk=np.exp(-(myt-(tstart+self.tau_u0_theta(T)))/self.tau_learning)*(wk+val*(1./self.tau_learning))
			wdyn.append(wk)
		return wdyn

