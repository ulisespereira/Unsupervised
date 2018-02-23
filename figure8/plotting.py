import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
from stimulus import *
from myintegrator import *
import cProfile
import json
import matplotlib.gridspec as gridspec
import cPickle as pickle

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
	field_H=(H*(1.-(phi(post_u,theta,uc)/y0))-H**2)/tau_H
	field_w=np.multiply(tauWinv(x_hist),winf(x_hist)-W)
	return field_a,field_u,field_w,field_H

#This are a the parameters of the simulation

#open parameters of the model
n=10 #n pop
delay=15.3
tau=10.   #timescale of populations
tau_H=2000.#200000.
af=0.1
bf=0.
y0=.05*np.ones(n)
w_i=4.3
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
lagStim=400.
times=235
amp=3.5


delta=7.
period=14.

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

# settin`g up the simulation 

r1_matrix=np.ones((n,n))
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp

#integrator
npts=int(np.floor(delay/dt)+1)         # points delay
tmax=times*(lagStim+n*(period+delta))+100.+mystim.delay_begin
thetmax=tmax+30*tau_H

#t = np.linspace(0,thetmax,10000)
u,connectivity,myH,t=pickle.load(open('dyn_stimulation.p','rb'))



#`t_u = t[0,120000]
#indexes = 2 * 450 * 1000
#connectivity  = connectivity[indexes,:]
#myH  = myH[indexes,:]
#u = u[0,]

#-----------------------------------------------------------------------------------------
#-------------------------------- Dynamics-----------------------------------------------
#----------------------------------------------------------------------------------------

#initial conditions

tmaxdyn=500
mystim.inten=0.
u_ret,connectivity_ret,myH_ret,t_ret = pickle.load(open('dyn_retrieval.p','rb'))
u_ret_PA,connectivity_ret_PA,myH_ret_PA,t_ret_PA = pickle.load(open('dyn_retrieval_PA.p','rb'))

#-------------------------------------------------------------------
#-----------------Stimulation of Populations------------------------
#-------------------------------------------------------------------

rc={'axes.labelsize': 32, 'font.size': 30, 'legend.fontsize': 25.0, 'axes.titlesize': 35}
plt.rcParams.update(**rc)
plt.rcParams['image.cmap'] = 'jet'

fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(3, 2,height_ratios=[3,3,2])
gs.update(wspace=0.44,hspace=0.03)
gs0 = gridspec.GridSpec(2, 2)
gs1 = gridspec.GridSpec(1, 2)
gs0.update(wspace=0.05,hspace=0.4,left=0.52,right=1.,top=0.8801,bottom=0.307)
gs1.update(wspace=0.05,hspace=0.4,left=0.1245,right=1.,top=0.21,bottom=0.05)
ax1A = plt.subplot(gs[0,0])
ax1B = plt.subplot(gs[1,0])
ax2A1 = plt.subplot(gs0[1,0])
ax2A2 = plt.subplot(gs0[1,1])
ax2B= plt.subplot(gs0[0,0])
ax2C= plt.subplot(gs0[0,1])
axSA= plt.subplot(gs1[0,0])
axPA= plt.subplot(gs1[0,1])


colormap = plt.cm.Accent

ax2B.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax2B.plot(t,phi(u[:,:],theta,uc),lw=3)
mystim.inten=.1
elstim=np.array([sum(mystim.stim(x)) for x in t])
ax2B.plot(t,elstim,'k',lw=3)
ax2B.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
ax2B.set_ylim([0,1.2])
ax2B.set_xlim([0,450])
ax2B.set_yticks([0.5,1])
ax2B.set_xticks([0,200,400])
ax2B.set_xticklabels([0.,.2,.4])
ax2B.set_xlabel('Time (s)')
ax2B.set_ylabel('Rate')
ax2B.set_title('(B)',x=1.028,y=1.04)


ax2C.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax2C.plot(t,phi(u[:,:],theta,uc),lw=3)
mystim.inten=.1
elstim=np.array([sum(mystim.stim(x)) for x in t])
ax2C.plot(t,elstim,'k',lw=3)
ax2C.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
ax2C.set_xlim([45650,46100])
ax2C.set_xticks([45700,45900,46100])
ax2C.set_xticklabels([45.7,45.9,46.1])
ax2C.set_ylim([0,1.2])
ax2C.set_yticks([])
ax2C.set_xlabel('Time (s)')
#ax2C.set_ylabel('Rate')

#----------------------------------------------------------------------
#------------Synaptic Weights------------------------------------------
#----------------------------------------------------------------------


for i in range(10):
		ax1A.plot(t,connectivity[:,i,i],'c',lw=3)
for i in range(0,9):
		ax1A.plot(t,connectivity[:,i+1,i],'y',lw=3)
for i in range(8):
		ax1A.plot(t,connectivity[:,i+2,i],'g',lw=3)
for i in range(9):
		ax1A.plot(t,connectivity[:,i,i+1],'r',lw=3)
for i in range(8):
		ax1A.plot(t,connectivity[:,i,i+2],'b',lw=3)

#
ax1A.set_xticks([])
ax1A.axvline(x=tmax,ymin=0,ymax=2.,linewidth=2,ls='--',color='gray',alpha=0.7)
#ax1A.set_xticklabels([0,50,100,150])
ax1A.set_ylim([0,1.8])
ax1A.set_xlim([0,400000])
ax1A.set_yticks([0,0.5,1.,1.5])
#ax1A.set_xlabel('Time (s)')
ax1A.set_ylabel('Synaptic Weights')
ax1A.set_title('(A)',y=1.04)

#------------------------------------------------------------------------
#-------------Homeostatic Variable --------------------------------------
#------------------------------------------------------------------------


ax1B.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax1B.plot(t,myH[:],lw=3)
ax1B.axvline(x=tmax,ymin=0,ymax=2.,linewidth=2,ls='--',color='gray',alpha=0.7)
ax1B.set_ylim([0,1.2])
ax1B.set_yticks([0.5,1.])
ax1B.set_xlim([0,400000])
ax1B.set_xticks([0,100000,200000,300000,400000])
ax1B.set_xticklabels([0,100,200,300,400])
ax1B.set_xlabel('Time (s)')
ax1B.set_ylabel('H')
#ax1B.savefig('HdynamicsLearning.pdf', bbox_inches='tight')
#ax1B.xlim([0,50000])
#ax1B.xticks([0,10000,20000,30000,40000,50000],[0,10,20,30,40,50])
#ax1B.xlabel('Time (s)')
#ax1B.ylabel('H')
#ax1B.savefig('HdynamicsLearningzoom.pdf', bbox_inches='tight')
#ax1B.show()
#ax1B.close()






vmax=wmax
dataStim=[np.transpose(np.multiply(np.transpose(connectivity[i,:,:]),myH[i,:])) for i in [0,int((tmax/dt)/3.),int((tmax/dt)*2./3.),int(tmax/dt)] ]
ax2A1.matshow(dataStim[2], vmin=0, vmax=vmax)
ax2A1.set_xticks([])
ax2A1.set_yticks([])
ax2A1.set_xlabel('During stimulation')
ax2A1.set_title('(C)',x=1.05,y=1.04)

dataAfterStim=[np.transpose(np.multiply(np.transpose(connectivity[i,:,:]),myH[i,:])) for i in [int(tmax/dt),int(tmax/dt+((thetmax-tmax)/dt)/3.),int(tmax/dt+((thetmax-tmax)/dt)*2./3.),-1] ]

ax2A2.matshow(dataAfterStim[2], vmin=0, vmax=vmax)
ax2A2.set_xticks([])
ax2A2.set_yticks([])
ax2A2.set_xlabel('After stimulation')
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0., vmax=vmax))
# fake up the array of the scalar mappable. Urgh...
sm._A = []
cax = fig.add_axes([1., 0.307, 0.02, 0.239]) # [left, bottom, width, height] 
myticks=[0.0,.5,1.,1.5]
cbar=fig.colorbar(sm, cax=cax,ticks=myticks,alpha=1.)
cbar.ax.tick_params(labelsize=30) 

axSA.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
axSA.plot(t_ret,phi(u_ret[:,:],theta,uc),lw=5)
axSA.set_ylim([0,1.2])
axSA.set_xlim([0,220])
axSA.set_xticks([0,100,200])
axSA.set_yticks([0.5,1])
axSA.set_xlabel('Time (ms)')
axSA.set_ylabel('Rate')
axSA.set_title('(D)',x=1.028,y=1.04)

axPA.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
axPA.plot(t_ret_PA,phi(u_ret_PA[:,:],theta,uc),lw=5)
axPA.set_ylim([0,1.2])
axPA.set_xlim([0,220])
axPA.set_xticks([0,100,200])
axPA.set_yticks([])
axPA.set_xlabel('Time (ms)')
#axPA.set_ylabel('Rate')
#axPA.set_title('(D)',y=1.04)



plt.savefig('fig6.pdf', bbox_inches='tight')







