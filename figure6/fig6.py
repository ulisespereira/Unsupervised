import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
from stimulus import *
from myintegrator import *
import cProfile
import json
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
	#parameters
	n=len(pre_u)
	vec_pre=0.5*(np.ones(n)+np.tanh(a_pre*pre_u+b_pre))
	return (wmax/2.)*np.outer((np.ones(n)+np.tanh(a_post*post_u+b_post)),vec_pre)

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest

def tauWinv(x_hist):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	#return  np.add.outer(1/mytau(post_u),1/mytau(pre_u))
	return  tau_learning*np.outer(mytauInv(post_u),mytauInv(pre_u))
def F(u):
	return .5*(1.+np.tanh(af*(u-bf)))

def field(t,a,x_hist,W,H):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	conn_matrix=(W.T*H).T
	field_u=(1/tau)*(mystim.stim(t)+conn_matrix.dot(phi_tanh(x_hist[-1]))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi_tanh(x_hist[-1])))#-a
	field_a=0.#in the paper we are not using adaptation during learning
	field_H=(H*(1.-(post_u/y0)))/tau_H
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W)
	return field_a,field_u,field_w,field_H

#This are a the parameters of the simulation

#open parameters of the model
n=10 #n pop
delay=15.3
tau=10.   #timescale of populations
tau_H=10000.#10000
af=0.1
bf=0.
y0=.12*np.ones(n)# 0.12
w_i=1.
w_inh=w_i/n
nu=1.
theta=0.
uc=1.
wmax=3.500
thres=0.9
beta=1.6
tau_a=10.
#parameters stimulation
dt=0.5
lagStim=100.
times=135
amp=5.8


delta=13.5
period=15.


a_post=1.
b_post=-2.3
a_pre=1.0
b_pre=-2.3
tau_learning=400.

a1=2.
b1=-1.5


#-------------------------------------------------------------------
#-----------------Stimulation of Populations------------------------
#-------------------------------------------------------------------

# setting up the simulation 

r1_matrix=np.ones((n,n))
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp

#integrator
npts=int(np.floor(delay/dt)+1)         # points delay
tmax=times*(lagStim+n*(period+delta))+100.+mystim.delay_begin
thetmax=tmax+15.5*tau_H

#initial conditions
a0=np.zeros((npts,n))
x0=0.01*np.ones((npts,n))
W0=[0.1*np.ones((n,n)) for i in range(npts)]
H0=[0.5*np.ones(n) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,thetmax)
theintegrator.fast=False
adapt,u,connectivity,W01,myH,t=theintegrator.DDE_Norm_Miller(field,a0,x0,W0,H0)
#y0=.02*np.ones(n)# 0.12
#b1=2.25
#adapt2,u2,connectivity2,W012,myH2,t2=theintegrator.DDE_Norm_Miller(field,a0,x0,W0,H0)


#----------------------------------------------------------------------
#------------Synaptic Weights------------------------------------------
#----------------------------------------------------------------------

rc={'axes.labelsize': 50, 'font.size': 40, 'legend.fontsize': 25, 'axes.titlesize': 30}
plt.rcParams.update(**rc)

for i in range(10):
		plt.plot(t,connectivity[:,i,i],'c',lw=2)
for i in range(0,9):
		plt.plot(t,connectivity[:,i+1,i],'y',lw=2)
for i in range(8):
		plt.plot(t,connectivity[:,i+2,i],'g',lw=2)
for i in range(9):
		plt.plot(t,connectivity[:,i,i+1],'r',lw=2)
for i in range(8):
		plt.plot(t,connectivity[:,i,i+2],'b',lw=2)

plt.xlim([0,thetmax])
plt.xticks([0,50000,100000,150000,200000],[0,50,100,150,200])
plt.ylim([0,1.2])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
#plt.show()
plt.savefig('connectivitystimulationH.pdf', bbox_inches='tight')
plt.xlim([0,tmax])
plt.xticks([0,10000,20000,30000,40000,50000],[0,10,20,30,40,50])
plt.ylim([0,1.2])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitystimulationHzoom.pdf', bbox_inches='tight')
plt.close()

print 'connectivitystimulationHzoom.pdf',' is saved'

#for i in range(10):
#		plt.plot(t,connectivity2[:,i,i],'c',lw=2)
#for i in range(0,9):
#		plt.plot(t,connectivity2[:,i+1,i],'y',lw=2)
#for i in range(8):
#		plt.plot(t,connectivity2[:,i+2,i],'g',lw=2)
#for i in range(9):
#		plt.plot(t,connectivity2[:,i,i+1],'r',lw=2)
#for i in range(8):
#		plt.plot(t,connectivity2[:,i,i+2],'b',lw=2)
#
#plt.xlim([0,thetmax])
#plt.xticks([0,50000,100000,150000,200000],[0,50,100,150,200])
#plt.ylim([0,1.2])
#plt.yticks([0,0.4,0.8,1.2])
#plt.xlabel('Time (s)')
#plt.ylabel('Synaptic Weights')
##plt.show()
#plt.savefig('connectivitystimulationH2.pdf', bbox_inches='tight')
#plt.xlim([0,tmax])
#plt.xticks([0,10000,20000,30000,40000,50000],[0,10,20,30,40,50])
#plt.ylim([0,1.2])
#plt.yticks([0,0.4,0.8,1.2])
#plt.xlabel('Time (s)')
#plt.ylabel('Synaptic Weights')
#plt.savefig('connectivitystimulationHzoom2.pdf', bbox_inches='tight')
#plt.close()
#
#print 'connectivitystimulationHzoom2.pdf',' is saved'
#------------------------------------------------------------------------
#-------------Homeostatic Variable --------------------------------------
#------------------------------------------------------------------------


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,myH[:],lw=2)
plt.ylim([0,20.])
plt.yticks([0,5,10,15,20])
plt.xlim([0,thetmax])
plt.xticks([0,50000,100000,150000,200000],[0,50,100,150,200])
plt.xlabel('Time (s)')
plt.ylabel('H')
i#plt.show()
plt.savefig('HdynamicsLearning.pdf', bbox_inches='tight')
plt.ylim([0,1.])
plt.yticks([0,0.4,0.8,1.2])
plt.xlim([0,tmax])
plt.xticks([0,10000,20000,30000,40000,50000],[0,10,20,30,40,50])
plt.xlabel('Time (s)')
plt.ylabel('H')
plt.savefig('HdynamicsLearningzoom.pdf', bbox_inches='tight')
#plt.show()
plt.close()


print 'HdynamicsLearningzoom.pdf',' is saved'



#colormap = plt.cm.Accent
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
#plt.plot(t,myH2[:],lw=2)
#plt.ylim([0,1.2])
#plt.yticks([0,0.4,0.8,1.2])
#plt.xlim([0,thetmax])
#plt.xticks([0,50000,100000,150000,200000],[0,50,100,150,200])
#plt.xlabel('Time (s)')
#plt.ylabel('H')
#i#plt.show()
#plt.savefig('HdynamicsLearning2.pdf', bbox_inches='tight')
#plt.xlim([0,tmax])
#plt.xticks([0,10000,20000,30000,40000,50000],[0,10,20,30,40,50])
#plt.xlabel('Time (s)')
#plt.ylabel('H')
#plt.savefig('HdynamicsLearningzoom2.pdf', bbox_inches='tight')
##plt.show()
#plt.close()
#
#
#print 'HdynamicsLearningzoom2.pdf',' is saved'
#
#
#