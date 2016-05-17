import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
from stimulus import *
from myintegrator import *
import json
import scipy.integrate as integrate
import pickle
from scipy import optimize

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
	return (wmax/2.)*np.outer((np.ones(n)+np.tanh(a_post*post_u+b_post)),vec_pre)
	#return (wmax/2.)*np.outer(vec_post,vec_pre)

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest

def tauWinv(x_hist):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	return  tau_learning*np.outer(mytauInv(post_u),mytauInv(pre_u))

def F(u):
	return np.sqrt(wmax)*.5*(1.+np.tanh(a_post*u+b_post))

def field(t,a,x_hist,W,H):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	conn_matrix=(W.T*H).T
	field_u=(1/tau)*(mystim.stim(t)+conn_matrix.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-(w_inh/n)*np.dot(r1_matrix,phi(x_hist[-1],theta,uc)))#-a
	field_a=0.#in the paper we are not using adaptation during learning
	field_H=(H*(1.-(post_u/y0))-H**2)/tau_H
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W)
	return field_a,field_u,field_w,field_H

#This are a the parameters of the simulation

#open parameters of the model
n=10 #n pop
delay=15.3
tau=10.   #timescale of populations
tau_H=10000.
af=0.1
bf=0.
y0=.12*np.ones(n)
w_i=1.
w_inh=w_i/n
nu=1.
theta=0.
uc=1.
wmax=2.22
thres=0.9
beta=1.6
tau_a=10.
#parameters stimulation
dt=0.5
lagStim=100.
times=100
amp=5.


delta=8
period=20.


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

w0=0.1
def recurrentTheo(T,k):
	tstart=0.
	myt=tstart+T+tau_umax_theta(T)
	wdyn=[w0]
	wk=w0
	for i in range(k):
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart))*np.exp((x-delay-tstart-tau_u0_theta(T))/tau_learning) 
		myintegral=lambda y:integrate.quad(df,tstart+delay+tau_u0_theta(T),y)
		val,err=myintegral(myt)
		wk=np.exp(-(myt-(delay+tau_u0_theta(T)+tstart))/tau_learning)*(wk+val*(1./tau_learning))
		wdyn.append(wk)
	return wdyn

def feedforwardTheo(T,delta,k):
	tstart=0.
	myt=tstart+delay-delta+tau_umax_theta(T)
	wdyn=[w0]
	wk=w0
	for i in range(k):
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart-delta-T))*np.exp((x-tau_u0_theta(T)-tstart)/tau_learning) 
		myintegral=lambda y:integrate.quad(df,tau_u0_theta(T)+tstart,y)
		val,err=myintegral(myt)
		wk=np.exp(-(myt-(tstart+tau_u0_theta(T)))/tau_learning)*(wk+val*(1./tau_learning))
		wdyn.append(wk)
	return wdyn

def recurrentStationary(T):
	tstart=0.
	myt=tstart+T+tau_umax_theta(T)
	mytlower=tstart+delay+tau_u0_theta(T)
	df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart))*np.exp((x-delay-tstart-tau_u0_theta(T))/tau_learning) 
	myintegral=lambda y:integrate.quad(df,tstart+delay+tau_u0_theta(T),y)
	val,err=myintegral(myt)
	exps=np.exp(-(myt-mytlower)/tau_learning)/(1.-np.exp(-(myt-mytlower)/tau_learning))
	return val*exps*(1/tau_learning)



def feedforwardStationary(T,delta):
	tstart=0.
	myt=tstart+delay-delta+tau_umax_theta(T)
	mytlower=tau_u0_theta(T)+tstart
	df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart-delta-T))*np.exp((x-tau_u0_theta(T)-tstart)/tau_learning) 
	myintegral=lambda y:integrate.quad(df,tau_u0_theta(T)+tstart,y)
	val,err=myintegral(myt)
	exps=np.exp(-(myt-mytlower)/tau_learning)/(1.-np.exp(-(myt-mytlower)/tau_learning))
	return val*exps*(1/tau_learning)

def fieldStationary(param,w0,delta0):
	T=param[0]
	delt=param[1]
	return np.array([recurrentStationary(T)-w0,feedforwardStationary(T,delt)-delta0])

#----------------------------------------------------------------------------------------------
#-------------------------Plotting Stationary-----------------------------------------------
#---------------------------------------------------------------------------------------------

#rc={'axes.labelsize': 30, 'font.size': 22, 'legend.fontsize': 28.0, 'axes.titlesize': 30}
#myperiod=np.linspace(16.,40,100)
#mystatrec=[recurrentStationary(t) for t in myperiod]
#plt.plot(myperiod,mystatrec,lw=3)
#plt.xlabel(r'$T$')
#plt.ylabel(r'$w$')
#plt.show()
#
#myT=np.linspace(delay,40,10)
#myDelta=np.linspace(4,delay,10)
#data=[]
#for T in myT:
#	data_delt=[]
#	for delt in myDelta:
#		data_delt.append(feedforwardStationary(T,delt))
#	print T
#	data.append(data_delt)
#data=np.array(data)
#
#myplot=plt.contourf(myDelta,myT,data.transpose(),10,alpha=0.5,cmap=plt.cm.autumn,origin='lower')
#plt.xlabel(r'$\Delta$')
#plt.ylabel(r'$T$')
#plt.colorbar(myplot,ticks=[0,0.2,0.4,0.6,0.8,1.])
#plt.show()

#------------------------------------------------------------------
#---------------Bifurcation Diagram--------------------------------
#------------------------------------------------------------------
# This par of the code is to build a bifurcation diagram 
# that depends on the stimulation parameters T and delta -> period,delta



rc={'axes.labelsize': 30, 'font.size': 22, 'legend.fontsize': 28.0, 'axes.titlesize': 30}


#bifcurve=np.load('mybifcurve.npy')
#bifcurve_T_Delta=[]
#for bif in bifcurve:
#	val=optimize.root(fieldStationary,np.array([delay+5.,delay-5.]),args=(bif[0],bif[1])).x
#	print optimize.root(fieldStationary,np.array([delay+5.,delay-5.]),args=(bif[0],bif[1]))
#	bifcurve_T_Delta.append(val)
#	print bif
#
#bifcurve_T_Delta=np.array(bifcurve_T_Delta)

#line_PA_SAPA=np.linspace(0.1,1.1,100)+1.
#myline=np.linspace(0.1,1.1,100)
#bifcurve_PA_SAPA=[]
#for i in range(0,100):
#	val=optimize.root(fieldStationary,np.array([delay+5.,delay-5.]),args=(line_PA_SAPA[i],myline[i])).x
#	print optimize.root(fieldStationary,np.array([delay+5.,delay-5.]),args=(line_PA_SAPA[i],myline[i]))
#	bifcurve_PA_SAPA.append(val)
#	print i
#
#
#bifcurve_PA_SAPA=np.array(bifcurve_PA_SAPA)
#
the_filename='mycurve_T_Delta.npy'
the_filename2='mycurve_PA_SAPA.npy'

#with open(the_filename, 'wb') as f:
#	    pickle.dump(bifcurve_T_Delta, f)
#with open(the_filename2, 'wb') as f:
#	    pickle.dump(bifcurve_PA_SAPA, f)

with open(the_filename, 'rb') as f:
	    bifcurve_T_Delta = pickle.load(f)

with open(the_filename2, 'rb') as f:
	    bifcurve_PA_SAPA = pickle.load(f)



ub=optimize.root(fieldStationary,np.array([delay+5.,delay-5.]),args=(1.1,0.7)).x[0]
SA_to_PA=np.array([ub for i in range(100)])
SA_to_PA0=np.array([ub for i in range(len(bifcurve_T_Delta[:,0]))])

myT=np.linspace(delay,35.,100)
myDelta=np.linspace(4,24,100.)


colormap = plt.cm.afmhot
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])

plt.plot(bifcurve_T_Delta[:,1],bifcurve_T_Delta[:,0],'k')
plt.plot(myDelta,SA_to_PA,c='k',lw=1.)
plt.fill_between(bifcurve_T_Delta[:,1],bifcurve_T_Delta[:,0],SA_to_PA0,alpha=0.5,edgecolor='k', facecolor='red')
plt.fill_between(bifcurve_T_Delta[:,1],np.zeros(len(bifcurve_T_Delta[:,1])),bifcurve_T_Delta[:,0],alpha=0.5, facecolor='darkgrey',linewidth=0)
plt.fill_between(np.linspace(bifcurve_T_Delta[0,1],22.5,100),np.zeros(100),SA_to_PA,alpha=0.5,edgecolor='k', facecolor='darkgrey',linewidth=0)


plt.plot(bifcurve_PA_SAPA[:,1],bifcurve_PA_SAPA[:,0],'k')
plt.fill_between(bifcurve_PA_SAPA[:,1],bifcurve_PA_SAPA[:,0],80*np.ones(100),alpha=0.5,edgecolor='k', facecolor='blue')
plt.fill_between(bifcurve_PA_SAPA[:,1],SA_to_PA,bifcurve_PA_SAPA[:,0],alpha=0.5,edgecolor='k', facecolor='green',linewidth=0.)
plt.fill_between(np.linspace(1,min(bifcurve_PA_SAPA[:,1]),100),SA_to_PA,80*np.ones(100),alpha=0.5, facecolor='green',linewidth=0.)


plt.xlim([4.,22.5])
plt.ylim([5.,80.])
plt.xticks([5,10,15,20],fontsize=22)
plt.yticks([0,15,30,45,60.,75],fontsize=22)
plt.xlabel(r'$\Delta$ (ms)',fontsize='28')
plt.ylabel(r'$T$ (ms)',fontsize='28')
plt.savefig('bifdiagram_Delta_T.pdf', bbox_inches='tight')
plt.show()



