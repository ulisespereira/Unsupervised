import matplotlib.gridspec as gridspec
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
	return (wmax/4.)*np.outer((np.ones(n)+np.tanh(a_post*(post_u-b_post))),(np.ones(n)+np.tanh(a_pre*(pre_u-b_pre))))

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest

def tauWinv(x_hist):
	pre_u=phi(x_hist[0],theta,uc)
	post_u=phi(x_hist[-1],theta,uc)
	#return  np.add.outer(1/mytau(post_u),1/mytau(pre_u))
	return tau_learning*np.outer(1./mytau(post_u),1./mytau(pre_u))


def field(t,x_hist,W):
	field_u=(1/tau)*(mystim.stim(t)+W.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi(x_hist[-1],theta,uc)))
	field_w=np.multiply(tauWinv(x_hist),(-W+winf(x_hist)))
	return field_u,field_w

#script to save the parameters



#This are a the parameters of the simulation

#open parameters of the model
n=10 #n pop
delay=15.3 #multilpl:es of 9!
tau=10.   #timescale of populations
w_i=1.#5.
nu=1.
theta=0.
uc=1.
wmax=2.5
thres=0.6
#parameters stimulation
dt=0.5
lagStim=100.


amp=10.
amp_dc=0.
delta=15.3
period=40.
times=240

bf=10.
xf=0.7

a_post=bf
b_post=xf
a_pre=bf
b_pre=xf
tau_learning=400.


w_inh=w_i/n
r1_matrix=np.ones((n,n))
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
mystim=stimulus(patterns,lagStim,delta,period,times)
#integrato
npts=int(np.floor(delay/dt)+1)         # points delay
tmax=times*(lagStim+n*(period+delta))+40

#initial conditions
x0=0.01*np.ones((npts,n))

W0=[(0.1)*np.zeros((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False



rc={'axes.labelsize': 32, 'font.size': 30, 'legend.fontsize': 25, 'axes.titlesize': 35}
plt.rcParams.update(**rc)
plt.rcParams['image.cmap'] = 'jet'



#-------------------------------------------------------------
#-------------------stimulation instability-------------------
#------------------------------------------------------------
amp=1.3
delta=10.
period=19.
times=40
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp-amp_dc
mystim.amp_dc=amp_dc
tmax=times*(lagStim+n*(period+delta))+40
#initial conditions
x0=0.01*np.ones((npts,n))
W0=[(0.1)*np.ones((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False


u,Wdiag,Woffdiag,connectivity,W01,t=theintegrator.DDE(field,x0,W0)

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3)
gs0 = gridspec.GridSpec(2, 2)
gs.update(wspace=0.3,hspace=0.43)
gs0.update(wspace=0.1,hspace=0.1,left=0.67,right=0.91,top=0.88,bottom=0.56)
#gs0.update(wspace=0.1,hspace=0.1)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3a = plt.subplot(gs0[0,0])
ax3b = plt.subplot(gs0[0,1])
ax3c = plt.subplot(gs0[1,0])
ax3d = plt.subplot(gs0[1,1])
#ax3=plt.subplot(gs[0,2])
ax4 = plt.subplot(gs[1,0])
ax5 = plt.subplot(gs[1,1])
ax6 = plt.subplot(gs[1,2])
## Dynamics 
mystim.inten=.1
colormap = plt.cm.Accent
ax4.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax4.plot(t,phi(u[:,:],theta,uc),lw=3)
elstim=np.array([sum(mystim.stim(x)) for x in t])
ax4.plot(t,elstim,'k',lw=3)
ax4.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
ax4.set_ylim([0,1.2])
ax4.set_xlim([0,400])
ax4.set_yticks([0.5,1.])
ax4.set_xticks([0,200,400])
ax4.set_xticklabels([0,.200,.400])
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Rate')
#plt.savefig('stimulation1.pdf', bbox_inches='tight')
#plt.show()
#plt.close()

print 'stimulation1.pdf is stored'

colormap = plt.cm.Accent
ax5.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax5.plot(t,phi(u[:,:],theta,uc),lw=3)
elstim=np.array([sum(mystim.stim(x)) for x in t])
ax5.plot(t,elstim,'k',lw=3)
ax5.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
ax5.set_ylim([0,1.2])
time_plot=14*(lagStim+n*(period+delta))
ax5.set_xlim([time_plot,time_plot+400])
ax5.set_xticks([time_plot,(time_plot+200),(time_plot+400)])
ax5.set_xticklabels([time_plot*1e-3,(time_plot+200)*1e-3,(time_plot+400)*1e-3])
ax5.set_yticks([])
ax5.set_ylim([0,1.2])
ax5.set_xlabel('Time (s)')
ax5.set_title('(D)',y=1.04)
#ax5.set_ylabel('Rate')
#plt.savefig('stimulation2.pdf', bbox_inches='tight')
#plt.show()
#plt.close()
print 'stimulation2.pdf is stored'

colormap = plt.cm.Accent
ax6.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax6.plot(t,phi(u[:,:],theta,uc),lw=3)
elstim=np.array([sum(mystim.stim(x)) for x in t])
ax6.plot(t,elstim,'k',lw=3)
ax6.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
ax6.set_ylim([0,1.2])
time_plot=18*(lagStim+n*(period+delta))
ax6.set_xlim([time_plot,time_plot+400])
ax6.set_ylim([0,1.2])
ax6.set_xticks([time_plot,(time_plot+200),(time_plot+400)])
ax6.set_xticklabels([time_plot*1e-3,(time_plot+200)*1e-3,(time_plot+400)*1e-3])
ax6.set_yticks([])
ax6.set_xlabel('Time (s)')
#ax6.set_ylabel('Rate')
#plt.savefig('stimulation3.pdf', bbox_inches='tight')
#plt.show()
#plt.close()

print 'stimulation3.pdf is stored'

colormap = plt.cm.Accent
ax1.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax1.plot(t,phi(u[:,:],theta,uc),lw=3)
elstim=np.array([sum(mystim.stim(x)) for x in t])
ax1.plot(t,elstim,'k',lw=3)
ax1.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
ax1.set_xlim([4000,8000])
ax1.set_yticks([0.5,1])
ax1.set_xticks([4000,6000,8000])
ax1.set_xticklabels(['4','6','8'])
ax1.set_ylim([0,1.2])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Rate')
ax1.set_title('(A)',y=1.04)
#plt.savefig('stimulationall.pdf', bbox_inches='tight')
#plt.show()
#plt.close()

print 'stimulationall.pdf is stored'
###dynamics synapses

for i in range(10):
		ax2.plot(t,connectivity[:,i,i],'c',lw=3)
for i in range(0,9):
		ax2.plot(t,connectivity[:,i+1,i],'y',lw=3)
for i in range(8):
		ax2.plot(t,connectivity[:,i+2,i],'g',lw=3)
for i in range(9):
		ax2.plot(t,connectivity[:,i,i+1],'r',lw=3)
for i in range(8):
		ax2.plot(t,connectivity[:,i,i+2],'b',lw=3)
ax2.set_xlim([0,10000])
ax2.set_xticks([0,5000,10000])
ax2.set_xticklabels(['0','5','10'])
ax2.set_yticks([0,1.,2.,3.])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Synaptic Weights')
ax2.set_title('(B)',y=1.04)

ax3a.matshow(connectivity[0,:,:],vmin=0,vmax=2.5)
ax3a.set_title('(C)',y=1.08,x=1.06)
ax3a.set_xticks([])
ax3a.set_yticks([])
ax3b.matshow(connectivity[int(3000./dt),:,:],vmin=0,vmax=1.2)
ax3b.set_xticks([])
ax3b.set_yticks([])
ax3c.matshow(connectivity[int(6000./dt),:,:],vmin=0,vmax=1.2)
ax3c.set_xticks([])
ax3c.set_yticks([])
ax3d.matshow(connectivity[int(9000./dt),:,:],vmin=0,vmax=1.2)
ax3d.set_xticks([])
ax3d.set_yticks([])
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0., vmax=1.2))
# fake up the array of the scalar mappable. Urgh...
sm._A = []
cax = fig.add_axes([0.92, 0.56, 0.02, 0.325]) # [left, bottom, width, height] 
myticks=[0.0,1]
cbar=fig.colorbar(sm, cax=cax,ticks=myticks,alpha=1.)
cbar.ax.tick_params(labelsize=30) 
#cbar.set_label(r'Capacity ($\alpha_c$)',size=42)
#plt.tight_layout()
#gs0.tight_layout(fig,rect=[0.65,0.56,0.89,0.9],h_pad=0.1,w_pad=0.1) #
fig.savefig('fig4.pdf', bbox_inches='tight')
#plt.show()
plt.close()






print 'connectivitystimulation.pdf is stored'

print t[int(2000./0.5)]

data=[connectivity[0,:,:],connectivity[int(3000./dt),:,:],connectivity[int(6000./dt),:,:],connectivity[int(9000./dt),:,:]]
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



