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
wmax=3.500
thres=1.5
beta=1.6
tau_a=10.
#parameters stimulation
dt=0.5
lagStim=100.
times=15
amp=3.


delta=8
period=20.


a_post=1.
b_post=-0.25
a_pre=1.0
b_pre=-0.25
tau_learning=400.

a1=6.
b1=-0.25


#-------------------------------------------------------------------
#-----------------Stimulation of Populations------------------------
#-------------------------------------------------------------------

# settin`g up the simulation 

r1_matrix=np.ones((n,n))
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp

#integrator
npts=int(np.floor(delay/dt)+1)         # points delay
tmax=times*(lagStim+n*(period+delta))+100.+mystim.delay_begin
thetmax=tmax+6*tau_H

#initial conditions
a0=np.zeros((npts,n))
x0=0.01*np.ones((npts,n))
W0=[0.0001*np.ones((n,n)) for i in range(npts)]
H0=[1.1*np.ones(n) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,thetmax)
theintegrator.fast=False
adapt,u,connectivity,W01,myH,t=theintegrator.DDE_Norm_Miller(field,a0,x0,W0,H0)


#-----------------------------------------------------------------------------------------
#-------------------------------- Dynamics-----------------------------------------------
#----------------------------------------------------------------------------------------

rc={'axes.labelsize': 30, 'font.size': 20, 'legend.fontsize': 28.0, 'axes.titlesize': 30}
plt.rcParams.update(**rc)

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi(u[:,:],theta,uc),lw=2)
mystim.inten=.1
elstim=np.array([sum(mystim.stim(x)) for x in t])
plt.plot(t,elstim,'k',lw=3)
plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xlim([0,400])
plt.yticks([0,0.4,0.8,1.2])
plt.xticks([0,200,400])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulationH1.pdf', bbox_inches='tight')
plt.show()


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi(u[:,:],theta,uc),lw=2)
mystim.inten=.1
elstim=np.array([sum(mystim.stim(x)) for x in t])
plt.plot(t,elstim,'k',lw=3)
plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,1.2])
plt.xlim([4550,4950])
plt.xticks([4550,4750,4950])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulationH2.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi(u[:,:],theta,uc),lw=2)
mystim.inten=.1
elstim=np.array([sum(mystim.stim(x)) for x in t])
plt.plot(t,elstim,'k',lw=3)
plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.xlim([5350,5750])
plt.xticks([5350,5550,5750])
plt.ylim([0,1.2])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulationH3.pdf', bbox_inches='tight')
plt.show()


#----------------------------------------------------------------------
#------------Synaptic Weights------------------------------------------
#----------------------------------------------------------------------



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


decayTime=-10.*np.log((thres/amp)*1./(1-np.exp(-period/10.)))
print decayTime+period-10*np.log(1.5)-delay
print (wmax/4.)*(1.+np.tanh(a_pre*amp+b_pre))*(1.+np.tanh(a_post*amp+b_post))*(1.-np.exp(-(decayTime+period-delay-10*np.log(1.5))/tau_learning))
print (wmax/4.)*(1.+np.tanh(a_pre*amp+b_pre))*(1.+np.tanh(a_post*amp+b_post))*(1.-np.exp(-10./tau_learning))
print wmax*(1.-np.exp(-5./tau_learning))
#plt.axhline(xmin=min(t),xmax=max(t),y=(wmax/4.)*(1.+np.tanh(a_post*(2.-np.exp(-period/tau))*amp+b_post))*(1+np.tanh(a_pre*amp*(1-np.exp(-period/tau))+b_pre)),linewidth=2,color='m',ls='dashed')
plt.xlim([0,thetmax])
plt.xticks([0,15000,30000,45000,60000],['0','15','30','45','60'])
plt.ylim([0,1.2])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitystimulationH.pdf', bbox_inches='tight')
plt.xlim([0,tmax])
plt.xticks([0,2000,4000,8000])
plt.ylim([0,1.2])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitystimulationHzoom.pdf', bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------
#-------------Homeostatic Variable --------------------------------------
#------------------------------------------------------------------------


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,myH[:],lw=2)
plt.ylim([0,1.2])
plt.yticks([0,0.4,0.8,1.2])
plt.xticks([0,15000,30000,45000,60000],['0','15','30','45','60'])
plt.xlabel('Time (s)')
plt.ylabel('H')
plt.savefig('HdynamicsLearning.pdf', bbox_inches='tight')
plt.xlim([0,tmax])
plt.xticks([0,2000,4000,8000])
plt.xlabel('Time (ms)')
plt.ylabel('H')
plt.savefig('HdynamicsLearningzoom.pdf', bbox_inches='tight')
plt.show()


#--------------------------------------------------------------------------------
#-----------The Connectivity Matrices--------------------------------------------
#--------------------------------------------------------------------------------

# connectivity matrix during the stimulation
data=[connectivity[0,:,:],connectivity[int((tmax/dt)/3.),:,:],connectivity[int(2*(tmax/dt)/3.),:,:],connectivity[int(tmax/dt),:,:]]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=1.2)
	# Make an axis for the colorbar on the right side
#cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
#fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulationH.pdf', bbox_inches='tight')
plt.show()


#matrix connectivity and homoestatic variable during stimulation
data=[np.transpose(np.multiply(np.transpose(connectivity[i,:,:]),myH[i,:])) for i in [0,int((tmax/dt)/3.),int((tmax/dt)*2./3.),int(tmax/dt)] ]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=1.2)
	# Make an axis for the colorbar on the right side
#cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
#fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulationHhom.pdf', bbox_inches='tight')
plt.show()

# matrix connectivity after stimulation
data=[connectivity[int(tmax/dt),:,:],connectivity[int(tmax/dt+((thetmax-tmax)/dt)/3.),:,:],connectivity[int(tmax/dt+2*((thetmax-tmax)/dt)/3.),:,:],connectivity[-1,:,:]]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=1.2)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulationHFinal.pdf', bbox_inches='tight')
plt.show()

# matrix connectivity and homeostatic after stimulation 
data=[np.transpose(np.multiply(np.transpose(connectivity[i,:,:]),myH[i,:])) for i in [int(tmax/dt),int(tmax/dt+((thetmax-tmax)/dt)/3.),int(tmax/dt+((thetmax-tmax)/dt)*2./3.),-1] ]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=1.2)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulationHhomFinal.pdf', bbox_inches='tight')
plt.show()


#-------------------------------------------------------------------------
#--------------Stability of NS--------------------------------------------
#------------------------------------------------------------------------


#new instance of the model
#with no stimulation
#and using W int as the matrix after learning
#to see is sequences arise

amp=0.5
times=10
delta=2000.
period=10.
lagStim=300
patterns=np.identity(n)
patterns=[patterns[:,0]]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
tmax=times*(lagStim+(period+delta))+4
x0=np.zeros(n)
#x0[0]=10.
a0=np.zeros((npts,n))
x0=np.array([x0 for i in range(npts)])
#W0=[connectivity[-1,:,:] for i in range(npts)]
W0=[0.6*np.eye(n)+0.9*np.eye(n,k=-1) for i in range(npts)]
H0=[myH[-1,:] for i in range(npts)]
theintegrator_test=myintegrator(delay,dt,n,tmax)
theintegrator_test.fast=False

print W0[1]
adapt_test,u_test,connectivity_test,W0_test,H_test,t_test=theintegrator_test.DDE_Norm_Miller(field,a0,x0,W0,H0)
#Plotting

#-------------------------------------------------------------------------------------
#------------- Sequential Dynamics ---------------------------------------------------
#-------------------------------------------------------------------------------------


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=2)
plt.ylim([0,1.2])
plt.xlim([0,tmax])
plt.xticks([0,15000,30000,45000,60000],['0','1.5','3','4.5','6'])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequenceallH.pdf', bbox_inches='tight')
plt.show()


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,H_test[:,:])
plt.ylim([0,1.2])
plt.xlim([0,tmax])
plt.xticks([0,15000,30000,45000,60000],['0','1.5','3','4.5','6'])
plt.xlabel('Time (ms)')
plt.ylabel('H')
plt.savefig('HdynamicsSequence.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=2)
plt.ylim([0,1.1])
plt.xlim([0,350.])
plt.xticks([0,100,200,300])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencesfirstH.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=2)
plt.ylim([0,1.1])
plt.xlim([5100,5500.])
plt.xticks([5100,5200,5300,5400,5500])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencessecondH.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=2)
plt.ylim([0,1.1])
plt.xlim([5600,6000.])
plt.xticks([5600,5700,5800,5900,6000])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencethirdH.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi(u_test[:,:],theta,uc),lw=2)
plt.ylim([0,1.1])
plt.xlim([6100,6550.])
plt.xticks([6100,6200,6300,6400,6500])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequenceforthH.pdf', bbox_inches='tight')
plt.show()

#----------------------------------------------------------------------
#------------Synaptic Weights------------------------------------------
#----------------------------------------------------------------------



for i in range(10):
		plt.plot(t_test,connectivity[:,i,i],'c',lw=3)
for i in range(0,9):
		plt.plot(t_test,connectivity[:,i+1,i],'y',lw=3)
for i in range(8):
		plt.plot(t_test,connectivity[:,i+2,i],'g',lw=3)
for i in range(9):
		plt.plot(t_test,connectivity[:,i,i+1],'r',lw=3)
for i in range(8):
		plt.plot(t_test,connectivity[:,i,i+2],'b',lw=3)


plt.xlim([0,thetmax])
plt.xticks([0,15000,30000,45000,60000],['0','1.5','3','4.5','6'])
plt.ylim([0,1.2])
plt.yticks([0,0.4,0.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitydegradationH.pdf', bbox_inches='tight')
plt.show()


#connectivity matrices

data=[connectivity_test[0,:,:],connectivity_test[int(len(t_test)/3.),:,:],connectivity_test[int(2*len(t_test)/3.),:,:],connectivity_test[-1,:,:]]

#figure.colorbar(mymatrix,cax=cbaxes)
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
#cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
#fig.colorbar(im, cax=cax)
plt.savefig('matrixsequencesH.pdf', bbox_inches='tight')
plt.show()

data=[np.transpose(np.multiply(np.transpose(connectivity_test[i,:,:]),H_test[i,:])) for i in [0,int((tmax/dt)/3.),int((tmax/dt)*2./3.),int(tmax/dt)] ]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=1.2)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixsecuencesHom.pdf', bbox_inches='tight')
plt.show()




