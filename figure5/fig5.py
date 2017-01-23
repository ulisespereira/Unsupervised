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
	pre_u=phi(x_hist[0],theta,uc)
	post_u=phi(x_hist[-1],theta,uc)
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
def F(u):
	return .5*(1.+np.tanh(af*(u-bf)))


def field(t,a,x_hist,W,H):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	conn_matrix=(W.T*H).T
	field_u=(1/tau)*(mystim.stim(t)+conn_matrix.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi(x_hist[-1],theta,uc)))#-a
	field_a=0.#in the paper we are not using adaptation during learning
	field_H=(H*(1.-(phi(post_u,theta,uc)/y0)))/tau_H
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W)
	return field_a,field_u,field_w,field_H

def fieldQuadratic(t,a,x_hist,W,H):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	conn_matrix=(W.T*H).T
	field_u=(1/tau)*(mystim.stim(t)+conn_matrix.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi(x_hist[-1],theta,uc)))#-a
	field_a=0.#in the paper we are not using adaptation during learning
	field_H=(H*(1.-(phi(post_u,theta,uc)/y0))-H*H)/tau_H
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W)
	return field_a,field_u,field_w,field_H

#This are a the parameters of the simulation
#This are a the parameters of the simulation

#open parameters of the model
n=10 #n pop
delay=15.3
tau=10.   #timescale of populations
tau_H=10000.#10000
af=0.1
bf=0.
y0=.05*np.ones(n)# 0.12
w_i=1.
w_inh=w_i/n
nu=1.
theta=0.
uc=1.
wmax=1.6
thres=0.6
beta=1.6
tau_a=10.
#parameters stimulation
dt=0.5
lagStim=100.
times=235
amp=2.5


delta=12.
period=13.


bf=10.
xf=0.7
a_post=bf
b_post=xf
a_pre=bf
b_pre=xf
tau_learning=400.#30000.

a1=6.
b1=-0.25
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
x0=0.1*np.ones((npts,n))
W0=[0.1*np.ones((n,n)) for i in range(npts)]
H0=[np.array([0.1 for i in range(n)]) for i in range(npts)]
#H0=[np.array([19.52158144,13.31267976,13.35448593,13.35612847,13.35535822,13.35451532,13.35366458,13.35281449,13.35258073,13.35252602]) for i in range(npts)]
#H0=[0.5*np.ones(n) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,thetmax)
theintegrator.fast=False
adapt,u,connectivity,W01,myH,t=theintegrator.DDE_Norm_Miller(field,a0,x0,W0,H0)
W0=[0.1*np.ones((n,n)) for i in range(npts)]
H0=[0.1*np.ones(n) for i in range(npts)]
adaptQ,uQ,connectivityQ,W01Q,myHQ,tQ=theintegrator.DDE_Norm_Miller(fieldQuadratic,a0,x0,W0,H0)

print 'Linear'
print myH[-1]
print connectivity[-1]
print 'Quadratic'
print myHQ[-1]
print connectivityQ[-1]


#----------------------------------------------------------------------
#------------Synaptic Weights------------------------------------------
#----------------------------------------------------------------------

rc={'axes.labelsize': 50, 'font.size': 40, 'legend.fontsize': 25, 'axes.titlesize': 30}
plt.rcParams.update(**rc)

for i in range(10):
		plt.plot(t,connectivity[:,i,i],'c',lw=4)
for i in range(0,9):
		plt.plot(t,connectivity[:,i+1,i],'y',lw=4)

print 'connectivitystimulationHzoom.pdf',' is saved'

for i in range(10):
		plt.plot(t,connectivityQ[:,i,i],'b--',lw=4,alpha=0.5)
for i in range(0,9):
		plt.plot(t,connectivityQ[:,i+1,i],'g--',lw=4,alpha=0.5)
#plt.xlim([0,thetmax])
#plt.xticks([0,50000,100000,150000,200000],[0,50,100,150,200])
#plt.ylim([0,1.2])
#plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')

plt.savefig('connectivitystimulationH.pdf', bbox_inches='tight')
#plt.xlim([0,tmax])
#plt.xticks([0,10000,20000,30000,40000,50000],[0,10,20,30,40,50])
#plt.ylim([0,1.2])
#plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitystimulationHzoom.pdf', bbox_inches='tight')
plt.close()

#------------------------------------------------------------------------
#-------------Homeostatic Variable --------------------------------------
#------------------------------------------------------------------------


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,myH[:],lw=4)
plt.plot(tQ,myHQ[:],'--',lw=4)
plt.ylim([0,5.])
plt.yticks([0,1,2,3,4,5])
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

#--------------------------------------------------------------------------
#-------------Printing Connectivity Matrices-------------------------------
#--------------------------------------------------------------------------


# matrix connectivity and homeostatic after stimulation 
linearW=np.transpose(np.multiply(np.transpose(connectivity[-1,:,:]),myH[-1,:]))
linearWsep=np.transpose(connectivity[-1,:,:])
QW=np.transpose(np.multiply(np.transpose(connectivityQ[-1,:,:]),myHQ[-1,:]))
QWsep=np.transpose(connectivityQ[-1,:,:])

vmax=1.6
#titles=['Linear'+r' $\matbb{W}$','Linear'+r' $\matbf{W}$','Modified'+r' $\matbb{W}$','Modified'+r' $\matbf{W}$']
fig, axes = plt.subplots()
im=axes.matshow(linearWsep, vmin=0, vmax=vmax)
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax,ticks=[0,vmax/2.,vmax])
plt.savefig('linearWsep.pdf', bbox_inches='tight')
plt.close()

fig, axes = plt.subplots()
im=axes.matshow(linearW, vmin=0, vmax=5)
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax,ticks=[0,2.5,5])
plt.savefig('linearW.pdf', bbox_inches='tight')
plt.close()

fig, axes = plt.subplots()
im=axes.matshow(QWsep, vmin=0, vmax=vmax)
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax,ticks=[0,vmax/2,vmax])
plt.savefig('QWsep.pdf', bbox_inches='tight')
plt.close()

fig, axes = plt.subplots()
im=axes.matshow(QW, vmin=0, vmax=vmax)
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax,ticks=[0,vmax/2,vmax])
plt.savefig('QW.pdf', bbox_inches='tight')
plt.close()


