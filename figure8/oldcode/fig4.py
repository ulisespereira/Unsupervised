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
	return  np.outer(mytauInv(post_u),mytauInv(pre_u))
def F(u):
	return .5*(1.+np.tanh(af*(u-bf)))

def field(t,a,x_hist,W,H):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	conn_matrix=(W.T*H).T
	field_u=(1/tau)*(mystim.stim(t)+conn_matrix.dot(phi_tanh(x_hist[-1]))-x_hist[-1]-(w_inh/n)*np.dot(r1_matrix,phi_tanh(x_hist[-1])))#-a
	field_a=0.#(-a+beta*x_hist[-1])/tau_a
	field_H=(H*(1.-(post_u/y0))-H**2)/tau_H
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W)
	return field_a,field_u,field_w,field_H

#This are a the parameters of the simulation

#open parameters of the model
n=20 #n pop
delay=15.3
tau=10.   #timescale of populations
tau_H=1000.
af=0.1
bf=0.
y0=.12*np.ones(n)
w_i=0.4
nu=2.
theta=0.
uc=1/nu
wmax=340.8
thres=0.6
beta=1.6
tau_a=10.
#parameters stimulation
dt=0.1
delta=5.3
lagStim=100.
times=29
period=16.
amp=3.


a_post=1.
b_post=-0.25
a_pre=1.0
b_pre=-0.25
tau_learning=400.

a1=6.
b1=-0.25

# comment this lines if you don t want to load or save the parameters of the simulation
#name='failure_learning2.param'
#save_param(name)
#load_param(name)



w_inh=w_i/n
r1_matrix=np.ones((n,n))
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
#integrator
npts=int(np.floor(delay/dt)+1)         # points delay
tmax=times*(lagStim+n*(period+delta))+100.+mystim.delay_begin
thetmax=tmax+7000.
#initial conditions
a0=np.zeros((npts,n))
x0=0.01*np.ones((npts,n))
W0=[0.0001*np.ones((n,n)) for i in range(npts)]
H0=[1.1*np.ones(n) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,thetmax)
theintegrator.fast=False
adapt,u,connectivity,W01,myH,t=theintegrator.DDE_Norm_Miller(field,a0,x0,W0,H0)


################## Sitmulation ########################################
rc={'axes.labelsize': 30, 'font.size': 20, 'legend.fontsize': 28.0, 'axes.titlesize': 30}
plt.rcParams.update(**rc)


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi_tanh(u[:,:]),lw=2)
mystim.inten=.02
plt.plot(t,[mystim.stim(x) for x in t],'k',lw=2)
plt.ylim([0,1.1])
plt.xlim([0,thetmax])
plt.xticks([0,4000,8000,12000,16000])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulationallH.pdf', bbox_inches='tight')
plt.show()


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi_tanh(u[:,:]),lw=2)
mystim.inten=.02
plt.plot(t,[mystim.stim(x) for x in t],'k',lw=2)
plt.ylim([0,1.1])
plt.xlim([0,40+20*(delta+period)+70])
plt.xticks([0,200,400])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulationH1.pdf', bbox_inches='tight')
plt.show()


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi_tanh(u[:,:]),lw=2)
mystim.inten=.02
plt.plot(t,[mystim.stim(x) for x in t],'k',lw=2)
plt.ylim([0,1.1])
plt.xlim([8950,9500])
plt.xticks([8950,9150,9350,9500])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulationH2.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,phi_tanh(u[:,:]),lw=2)
mystim.inten=.02
plt.plot(t,[mystim.stim(x) for x in t],'k',lw=2)
plt.ylim([0,1.1])
plt.xlim([9500,10000])
plt.xticks([9500,9700,9900])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulationH3.pdf', bbox_inches='tight')
plt.show()

#
#print amp*(1-np.exp(-period/tau))
##
###dynamics synapses
for i in range(20):
		plt.plot(t,connectivity[:,i,i],'c',lw=2)
for i in range(0,19):
		plt.plot(t,connectivity[:,i+1,i],'y',lw=2)
for i in range(18):
		plt.plot(t,connectivity[:,i+2,i],'g',lw=2)
for i in range(19):
		plt.plot(t,connectivity[:,i,i+1],'r',lw=2)
for i in range(18):
		plt.plot(t,connectivity[:,i,i+2],'b',lw=2)

#plt.axhline(xmin=min(t),xmax=max(t),y=(wmax/4.)*(1.+np.tanh(a_post*(2.-np.exp(-period/tau))*amp+b_post))*(1+np.tanh(a_pre*amp*(1-np.exp(-period/tau))+b_pre)),linewidth=2,color='m',ls='dashed')
plt.xlim([0,thetmax])
plt.xticks([0,4000,8000,12000,16000])
plt.yticks([0,.2,.4,.6,.8])
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitystimulationH.pdf', bbox_inches='tight')
plt.show()


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,myH[:])
plt.ylim([0,1.])
plt.xticks([0,4000,8000,12000,16000])
plt.xlabel('Time (ms)')
plt.ylabel('H')
plt.savefig('HdynamicsLearning.pdf', bbox_inches='tight')
plt.show()

data=[connectivity[0,:,:],connectivity[int((tmax/dt)/3.),:,:],connectivity[int(2*(tmax/dt)/3.),:,:],connectivity[int(tmax/dt),:,:]]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
#cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
#fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulationH.pdf', bbox_inches='tight')
plt.show()


##outcomes sequence


data=[np.transpose(np.multiply(np.transpose(connectivity[i,:,:]),myH[i,:])) for i in [0,int((tmax/dt)/3.),int((tmax/dt)*2./3.),int(tmax/dt)] ]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
#cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
#fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulationHhom.pdf', bbox_inches='tight')
plt.show()


data=[connectivity[int(tmax/dt),:,:],connectivity[int(tmax/dt+((thetmax-tmax)/dt)/3.),:,:],connectivity[int(tmax/dt+2*((thetmax-tmax)/dt)/3.),:,:],connectivity[-1,:,:]]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulationHFinal.pdf', bbox_inches='tight')
plt.show()

data=[np.transpose(np.multiply(np.transpose(connectivity[i,:,:]),myH[i,:])) for i in [int(tmax/dt),int(tmax/dt+((thetmax-tmax)/dt)/3.),int(tmax/dt+((thetmax-tmax)/dt)*2./3.),-1] ]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulationHhomFinal.pdf', bbox_inches='tight')
plt.show()



#new instance of the model

#new instance of the model
#with no stimulation
#and using W int as the matrix after learning
#to see is sequences arise
#w_i=0.8
#w_inh=w_i/n
#beta=0.
#fig 3c 
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
W0=[connectivity[-1,:,:] for i in range(npts)]
#W0=[0.05*np.eye(n)+1.0*np.eye(n,k=-1) for i in range(npts)]
H0=[myH[-1,:] for i in range(npts)]
theintegrator_test=myintegrator(delay,dt,n,tmax)
theintegrator_test.fast=False


adapt_test,u_test,connectivity_test,W0_test,H_test,t_test=theintegrator_test.DDE_Norm_Miller(field,a0,x0,W0,H0)
#Plotting


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
#plt.xlim([0,8])

plt.xticks([0,2000,4000,6000,8000])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequenceallH.pdf', bbox_inches='tight')
plt.show()


colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,H_test[:,:])
#plt.ylim([0,1])
plt.xticks([0,2000,4000,6000,8000])
plt.xlabel('Time (ms)')
plt.ylabel('H')
plt.savefig('HdynamicsSequence.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xlim([0,350.])
plt.xticks([0,100,200,300])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencesfirstH.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xlim([5100,5500.])
plt.xticks([5100,5200,5300,5400,5500])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencessecondH.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xlim([5600,6000.])
plt.xticks([5600,5700,5800,5900,6000])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencethirdH.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xlim([6100,6550.])
plt.xticks([6100,6200,6300,6400,6500])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequenceforthH.pdf', bbox_inches='tight')
plt.show()

###dynamics synapses
for i in range(20):
		plt.plot(t_test,connectivity_test[:,i,i],'c',lw=2)
for i in range(19):
		plt.plot(t_test,connectivity_test[:,i+1,i],'y',lw=2)

for i in range(18):
		plt.plot(t_test,connectivity_test[:,i+2,i],'g',lw=2)
for i in range(19):
		plt.plot(t_test,connectivity_test[:,i,i+1],'r',lw=2)
for i in range(18):
		plt.plot(t_test,connectivity_test[:,i,i+2],'b',lw=2)
##plt.axhline(xmin=min(t),xmax=max(t),y=(wmax/4.)*(1.+np.tanh(a_post*(2.-np.exp(-period/tau))*amp+b_post))*(1+np.tanh(a_pre*amp*(1-np.exp(-period/tau))+b_pre)),linewidth=2,color='m',ls='dashed')
plt.ylim([0,0.8])
plt.xticks([0,2000,4000,6000,8000])
plt.yticks([0,.2,.4,.6,.8])
plt.xlabel('Time (ms)')
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
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixsecuencesHom.pdf', bbox_inches='tight')
plt.show()




