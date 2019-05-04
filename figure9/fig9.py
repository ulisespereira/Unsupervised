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
from functions import *

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
	pre_u=phi(x_hist[0],theta,uc)
	post_u=phi(x_hist[-1],theta,uc)
	mynu=5.5
	mytheta=-0.8
	#parameters
	n=len(pre_u)
	vec_pre=0.5*(np.ones(n)+np.tanh(a_pre*(pre_u-b_pre)))
	vec_post=0.5*(np.ones(n)+np.tanh(a_post*(post_u-b_post)))
	#return (wmax/2.)*np.outer((np.ones(n)+np.tanh(a_post*post_u+b_post)),vec_pre)
	return wmax*np.outer(vec_post,vec_pre)

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest

def tauWinv(x_hist):
	pre_u=phi(x_hist[0],theta,uc)
	post_u=phi(x_hist[-1],theta,uc)

	tau_inv =   np.add.outer(mytauInv(post_u),mytauInv(pre_u))
	tau_inv[tau_inv == 2. / tau_learning] = 1./tau_learning
	return tau_inv
	#return tau_learning*np.outer(1./mytau(post_u),1./mytau(pre_u))

def F(u):
	if theta<=u and u<=uc:
		r=nu*(u-theta)
	elif u<=theta:
		r=0
	elif uc<=u:
		r=nu*(uc-theta)
	return np.sqrt(wmax)*.5*(1.+np.tanh(a_post*(r-b_post)))

def field(t,a,x_hist,W,H):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	conn_matrix=(W.T*H).T
	field_u=(1/tau)*(mystim.stim(t)+conn_matrix.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi(x_hist[-1],theta,uc)))#-a
	field_a=0.#in the paper we are not using adaptation during learning
	field_H=(H*(1.-(phi(post_u,theta,uc)/y0))-H**2)/tau_H
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W)
	return field_a,field_u,field_w,field_H

#This are a the parameters of the simulation

#open parameters of the model
n=20 #n pop
delay=15.3
tau=10.   #timescale of populations
tau_H=20000.
y0=.01*np.ones(n)
w_i=1.
w_inh=w_i/n
nu=1.
theta=0.
uc=1.
wmax=2.40
thres=0.6
beta=1.6
tau_a=10.
#parameters stimulation
dt=0.5
lagStim=100.
times=20
amp=1.5



bf=10.
xf=0.7
a_post=bf
b_post=xf
a_pre=bf
b_pre=xf
tau_learning=400.

a1=6.
b1=-0.25



#---------------------------------------------------------------
#----------------Learning vector Field--------------------------
#---------------------------------------------------------------


def tau_u0_theta(T):
	return -tau*np.log(1.-((thres-amp_dc)/(amp-amp_dc))) #including amp_dc

def tau_umax_theta(T):
	return -tau*np.log(((thres-amp_dc)/((amp-amp_dc)*(1.-np.exp(-T/tau)))))
def tau_theta(T):
	return T-tau_u0_theta(T)+tau_umax_theta(T)

#approximation pupulation dynamics
def myu(t,T,tstart):
	ttilda=t-tstart
	if tau_u0_theta(T)<=ttilda and ttilda<=T:
		return (amp-amp_dc)*(1.-np.exp(-ttilda/tau))+amp_dc
	elif ttilda>T:
		return ((amp-amp_dc)*(1.-np.exp(-T/tau))+amp_dc)*np.exp(-(ttilda-T)/tau)+amp_dc*(1.-np.exp(-(ttilda-T)/tau))
	elif tau_u0_theta(T)>ttilda:
		return (amp-amp_dc)*(1.-np.exp(-ttilda/tau))+amp_dc
	elif ttilda<0:
		print 'holi',ttilda


def recurrentTheo(t,T,tstart,w0):
	t0 = tstart + tau_u0_theta(T)
	t1 = tstart + T +tau_umax_theta(T) + delay
	if t0<=t and t<=t1:
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart))*np.exp((x-t0)/tau_learning) 
		myintegral=lambda y:integrate.quad(df, t0, y)
		val,err=myintegral(t)
		return np.exp(-(t-t0)/tau_learning)*(w0+val*(1./tau_learning))
	
	elif t>t1:
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart))*np.exp((x-t0)/tau_learning) 
		myintegral=lambda y:integrate.quad(df, t0, y)
		val,err=myintegral(t1)
		return np.exp(-(t1-t0)/tau_learning)*(w0+val*(1./tau_learning))
	
	else:
		return w0

def feedforwardTheo(t,T,delta,tstart,w0):
	t0 = tstart + tau_u0_theta(T) + delay
	t1 = tstart + delta + 2 * T + tau_umax_theta(T)# old
	if t0<=t and t<=t1:
		df=lambda x:F(myu(x,T,tstart+T+delta))  * F(myu(x-delay,T,tstart))*np.exp((x-t0)/tau_learning) 
		myintegral=lambda y:integrate.quad(df,t0,y)
		val,err=myintegral(t)
		return np.exp(-(t-t0)/tau_learning) * (w0 + val * (1./tau_learning))
	
	elif t>t1:
		df=lambda x:F(myu(x,T,tstart+T+delta))*F(myu(x-delay,T,tstart))*np.exp((x-t0)/tau_learning) 
		myintegral = lambda y:integrate.quad(df,t0,y)
		val,err = myintegral(t1)
		return np.exp(-(t1-t0)/tau_learning) * (w0 + val * (1./tau_learning))
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
amp_dc=0.
amp=5.
w0=0.1
a0=np.zeros((npts,n))
x0=0.01*np.ones((npts,n))
W0=[w0*np.ones((n,n)) for i in range(npts)]
H0=[0.1*np.ones(n) for i in range(npts)]

mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp-amp_dc
mystim.amp_dc=amp_dc
#integrator


#tmax
tmax = times * (lagStim+n*(period+delta)) +mystim.delay_begin
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False
adapt,u,connectivity,W01,myH,t=theintegrator.DDE_Norm_Miller(field,a0,x0,W0,H0)

#tmax=times*(lagStim+n*(period+delta))+mystim.delay_begin
# starting stimulation
tstart1=mystim.delay_begin+period+delta
tstart2=mystim.delay_begin+2*period+2*delta

num_stim  = 5
#ending stimulation
tstart1_a= 1 * (period+delta) + (num_stim - 1) * (lagStim+n*(period+delta)) +mystim.delay_begin
tstart2_a= 2 * (period+delta) + (num_stim - 1) * (lagStim+n*(period+delta)) +mystim.delay_begin

# current and weights from theory
t_first = np.arange(0,300,dt) 
theo_u1=np.array([myu(myt,period,tstart1) for myt in t_first])
theo_u2=np.array([myu(myt,period,tstart2) for myt in t_first])

theo_w=np.array([recurrentTheo(myt,period,tstart1,w0) for myt in t_first])
theo_s=np.array([feedforwardTheo(myt,period,delta,tstart1,w0) for myt in t_first])

t_last = np.arange(tstart1_a - 100,tstart1_a+ 300,dt)
theo_u1_a=np.array([myu(myt,period,tstart1_a) for myt in t_last])
theo_u2_a=np.array([myu(myt,period,tstart2_a) for myt in t_last])


w0_a = w0
s0_a = w0
for k in range(num_stim):
	tstart_loop= 1 * (period+delta) + (k - 1) * (lagStim+n*(period+delta)) +mystim.delay_begin
	t_loop = np.arange(tstart_loop - 100,tstart_loop + 300,dt)
	w0_a=np.array([recurrentTheo(myt,period,tstart1,w0_a) for myt in t_loop])[-1]
	s0_a=np.array([feedforwardTheo(myt,period,delta,tstart1,s0_a) for myt in t_loop])[-1]
	print k
	print 'w0_a =',w0_a,'s0_a',s0_a

theo_w_a=np.array([recurrentTheo(myt,period,tstart1_a,w0_a) for myt in t_last])
theo_s_a=np.array([feedforwardTheo(myt,period,delta,tstart1_a,s0_a) for myt in t_last])
# figrue 1


rc={'axes.labelsize': 50, 'font.size': 40, 'legend.fontsize': 25, 'axes.titlesize': 30}
plt.rcParams.update(**rc)

mylw=6
figure=plt.figure(figsize=(25,20))
plt.subplots_adjust(wspace=0.15)
plt.subplots_adjust(hspace=0.1)


fig1=figure.add_subplot(221)
fig1.plot(t,u[:,1],'k',lw=mylw,alpha = 0.5)
fig1.plot(t,u[:,2],'k',lw=mylw,alpha = 0.5)
#fig1.plot(t,thres*np.ones(len(u[:,1])),'r',lw=3)
fig1.plot(t_first,theo_u1,'r:',lw=6,alpha=0.6)
fig1.plot(t_first,theo_u2,'r:',lw=6,alpha=0.6)
#fig1.set_xlabel('Time (ms)',size=40)
fig1.set_ylabel(r'$u$',size=70)
fig1.set_xticks([])
fig1.set_yticks([1.,2,3.,4,5.])
fig1.set_ylim([0,5])
fig1.set_xlim([80,250])
fig1.set_title('(A)',size=60,y=1.04)
#fig1.axhline(y=thres,xmin=0,xmax=500,linewidth=8,color='gray',linestyle=':',alpha=0.8)

fig2=figure.add_subplot(222)
fig2.plot(t,u[:,1],'k',lw=mylw,alpha = 0.5)
fig2.plot(t,u[:,2],'k',lw=mylw,alpha = 0.5)
#fig2.plot(t,thres*np.ones(len(u[:,1])),'r',lw=3)
fig2.plot(t_last,theo_u1_a,'r:',lw=6,alpha=0.6)
fig2.plot(t_last,theo_u2_a,'r:',lw=6,alpha=0.6)
#fig2.set_xlabel('Time (ms)',size=40)
#fig2.set_ylabel(r'$u$',size=70)
fig2.set_xticks([])
fig2.set_yticks([1.,2,3.,4,5.])
fig2.set_ylim([0,5])
#fig2.set_xlim([tstart1_a-10,tstart1_a+100])
fig2.set_xlim([4310,4480])
fig2.set_title('(B)',size=60,y=1.04)
#fig2.set_title('(A)',size=60,y=1.04)
#fig2.axhline(y=thres,xmin=0,xmax=500,linewidth=8,color='gray',linestyle=':',alpha=0.8)



fig3=figure.add_subplot(223)
fig3.plot(t,connectivity[:,1,1],'c',lw=mylw)
fig3.plot(t_first,theo_w,'r:',lw=6,alpha=0.8)
fig3.plot(t,connectivity[:,2,1],'y',lw=mylw)
fig3.plot(t_first,theo_s,'r:',lw=6,alpha=0.8)
fig3.set_ylabel('Synaptic Weight',size=40)
fig3.set_xlabel('Time (ms)',size=40)
fig3.set_xticks([100,150,200])
fig3.set_yticks([0,0.2,0.4,0.6])
fig3.set_ylim([0.0,0.65])
fig3.set_xlim([80,250])

fig4=figure.add_subplot(224)
fig4.plot(t,connectivity[:,1,1],'c',lw=mylw)
fig4.plot(t_last,theo_w_a,'r:',lw=6,alpha=0.8)
fig4.plot(t,connectivity[:,2,1],'y',lw=mylw)
fig4.plot(t_last,theo_s_a,'r:',lw=6,alpha=0.8)
#fig4.set_ylabel('Synaptic Weight',size=40)
fig4.set_xlabel('Time (ms)',size=40)
#fig4.set_xticks([100,150,200])
#fig4.set_xlim([tstart1_a-10,tstart1_a+100])
fig4.set_xlim([4310,4480])
fig4.set_xticks([4350,4400,4450])
fig4.set_yticks([0.4,0.6,0.8,1.])
fig4.set_ylim([0.3,1.05])


#plt.suptitle(r'$I_{DC}=$'+str(amp_dc))
plt.savefig('fig9.pdf', bbox_inches='tight')
#plt.show()





