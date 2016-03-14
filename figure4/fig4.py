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

def mytau(x): #time scale function synapses
	myresult=(1e50)*np.ones(len(x))
	myresult[x>thres]=tau_learning
	#print x>thres
	#print x
	#myresult=(1e8)*(1.+np.tanh(-50.*(x-thres)))+tau_learning
	#print myresult
	return myresult

def winf(x_hist):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	#parameters
	n=len(pre_u)
	return (wmax/4.)*np.outer((np.ones(n)+np.tanh(a_post*post_u+b_post)),(np.ones(n)+np.tanh(a_pre*pre_u+b_pre)))

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest

def tauWinv(x_hist):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	#return  np.add.outer(1/mytau(post_u),1/mytau(pre_u))
	return tau_learning*np.outer(1./mytau(post_u),1./mytau(pre_u))


def field(t,x_hist,W):
	field_u=(1/tau)*(mystim.stim(t)+W.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi_tanh(x_hist[-1])))
	field_w=np.multiply(tauWinv(x_hist),(-W+winf(x_hist)))
	return field_u,field_w

#script to save the parameters



#This are a the parameters of the simulation

#open parameters of the model
n=10 #n pop
delay=15.3 #multilpl:es of 9!
tau=10.   #timescale of populations
w_i=1.
nu=1.
theta=0.
uc=1.
wmax=3.5
thres=0.9
#parameters stimulation
dt=0.5
lagStim=100.


amp=20.
delta=15.3
period=40.
times=80

a_post=1.
b_post=-2.3
a_pre=1.
b_pre=-2.3
tau_learning=400.

a1=6.
b1=-0.25

w_inh=w_i/n
r1_matrix=np.ones((n,n))
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp

#integrato
npts=int(np.floor(delay/dt)+1)         # points delay
tmax=times*(lagStim+n*(period+delta))+40
#initial conditions
x0=0.01*np.ones((npts,n))
W0=[(0.1)*np.zeros((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False



rc={'axes.labelsize': 50, 'font.size': 40, 'legend.fontsize': 25, 'axes.titlesize': 30}
plt.rcParams.update(**rc)


#i
#-------------------------------------------------------------
#-------------------stimulation instability-------------------
#------------------------------------------------------------
amp=6.01
delta=5.
period=25.
times=40
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
tmax=times*(lagStim+n*(period+delta))+40
#initial conditions
x0=0.01*np.ones((npts,n))
W0=[(0.1)*np.ones((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False


u,Wdiag,Woffdiag,connectivity,W01,t=theintegrator.DDE(field,x0,W0)

## Dynamics 
mystim.inten=.1
colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi(u[:,:],theta,uc),lw=3)
elstim=np.array([sum(mystim.stim(x)) for x in t])
plt.plot(t,elstim,'k',lw=3)
plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xlim([0,400])
plt.yticks([0,0.4,0.8,1.2])
plt.xticks([0,200,400])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulation1.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'stimulation1.pdf is stored'

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi(u[:,:],theta,uc),lw=3)
elstim=np.array([sum(mystim.stim(x)) for x in t])
plt.plot(t,elstim,'k',lw=3)
plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xlim([7600,8000])
plt.xticks([7600,7800,8000])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulation2.pdf', bbox_inches='tight')
#plt.show()
plt.close()
print 'stimulation2.pdf is stored'

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi(u[:,:],theta,uc),lw=3)
elstim=np.array([sum(mystim.stim(x)) for x in t])
plt.plot(t,elstim,'k',lw=3)
plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xlim([13600,14000])
plt.xticks([13600,13800,14000])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulation3.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'stimulation3.pdf is stored'

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi(u[:,:],theta,uc),lw=3)
elstim=np.array([sum(mystim.stim(x)) for x in t])
plt.plot(t,elstim,'k',lw=3)
plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.xlim([0,tmax])
plt.yticks([0,0.4,0.8,1.2])
plt.xticks([0,5000,10000,15000],[0,5,10,15])
plt.ylim([0,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Rate')
plt.savefig('stimulationall.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'stimulationall.pdf is stored'
###dynamics synapses

for i in range(10):
		plt.plot(t,connectivity[:,i,i],'c',lw=3)
for i in range(0,9):
		plt.plot(t,connectivity[:,i+1,i],'y',lw=3)
for i in range(8):
		plt.plot(t,connectivity[:,i+2,i],'g',lw=3)
for i in range(9):
		plt.plot(t,connectivity[:,i,i+1],'r',lw=3)
for i in range(8):
		plt.plot(t,connectivity[:,i,i+2],'b',lw=3)
plt.xlim([0,tmax])
plt.xticks([0,5000,10000,15000],[0,5,10,15])
plt.yticks([0,1.,2.,3.])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitystimulation.pdf', bbox_inches='tight')
#plt.show()
plt.close()


print 'connectivitystimulation.pdf is stored'

data=[connectivity[0,:,:],connectivity[int(len(t)/3.),:,:],connectivity[int(2*len(t)/3.),:,:],connectivity[-1,:,:]]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=1.2)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulation.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'matrixstimulation.pdf is stored'
#-------------------------------------------------------
#-------------- sequences degradation ------------------
#-------------------------------------------------------


amp=1.5
times=10
delta=200.
period=7.
lagStim=1000
patterns=np.identity(n)
patterns=[patterns[:,0]]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
tmax=times*(lagStim+(period+delta))+4
x0=np.zeros(n)
#x0[0]=10.
x0=np.array([x0 for i in range(npts)])
W0=[(1.0)*np.identity(n)+0.85*np.eye(n,k=-1) for i in range(npts)]
theintegrator_test=myintegrator(delay,dt,n,tmax)
theintegrator_test.fast=False
u_test,Wdiag_test,Woffdiag_test,connectivity_test,W0_test,t_test=theintegrator_test.DDE(field,x0,W0)
mystim.inten=.1
elstim=np.array([sum(mystim.stim(x)) for x in t_test])

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=3)
plt.plot(t_test,elstim,'k',lw=3)
plt.fill_between(t_test,np.zeros(len(t_test)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xticks([0,4000,8000,12000],[0,4,8,12])
plt.yticks([0.4,0.8,1.2])
plt.xlim([0,tmax])
plt.xlabel('Time (s)')
plt.ylabel('Rate')
plt.savefig('sequenceall.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'sequenceall.pdf is stored'

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=3)
plt.plot(t_test,elstim,'k',lw=3)
plt.fill_between(t_test,np.zeros(len(t_test)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xlim([0,400.])
plt.xticks([0,200,400])
plt.yticks([0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencesfirst.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'sequencefirst.pdf is stored'
colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=3)
plt.plot(t_test,elstim,'k',lw=3)
plt.fill_between(t_test,np.zeros(len(t_test)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xlim([4800.,5200.])
plt.xticks([4800,5000,5200])
plt.yticks([0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencethird.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'sequencethird.pdf is stored'
colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=3)
plt.plot(t_test,elstim,'k',lw=3)
plt.fill_between(t_test,np.zeros(len(t_test)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xlim([9600,10000.])
plt.xticks([9600,9800,10000])
plt.yticks([0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequenceforth.pdf', bbox_inches='tight')
#plt.show()
plt.close()


print 'sequenceforth.pdf is stored'
###dynamics synapses
for i in range(10):
		plt.plot(t_test,connectivity_test[:,i,i],'c',lw=3)
for i in range(9):
		plt.plot(t_test,connectivity_test[:,i+1,i],'y',lw=3)

for i in range(8):
		plt.plot(t_test,connectivity_test[:,i+2,i],'g',lw=3)
for i in range(9):
		plt.plot(t_test,connectivity_test[:,i,i+1],'r',lw=3)
for i in range(8):
		plt.plot(t_test,connectivity_test[:,i,i+2],'b',lw=3)
plt.ylim([0,1.2])
plt.xlim([0,tmax])
plt.xticks([0,4000,8000,12000],[0,4,8,12])
plt.yticks([0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitydegradation.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'connectivitydegradation.pdf is stored'
#connectivity matrices

data=[connectivity_test[0,:,:],connectivity_test[int(len(t_test)/3.),:,:],connectivity_test[int(2*len(t_test)/3.),:,:],connectivity_test[-1,:,:]]

#figure.colorbar(mymatrix,cax=cbaxes)
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=1.2)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixsequences.pdf', bbox_inches='tight')
#plt.show()
plt.close()



print 'matrixsequences.pdf is stored'





