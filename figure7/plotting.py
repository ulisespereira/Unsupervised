import numpy as np
import matplotlib.pyplot as plt
from stimulus import *
from myintegrator import *
from functions import *
import matplotlib.gridspec as gridspec
import cPickle as pickle
#-------------------------------------------------------------------
#-------------------------------------------------------------------
#-----------------Stimulation of Populations------------------------
#-------------------------------------------------------------------

# settin`g up the simulation 
#times = 100
#delta  = 50
#period = 30
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp

#integrator
npts=int(np.floor(delay/dt)+1)         # points delay
tmax=times*(lagStim+n*(period+delta))+100.+mystim.delay_begin
thetmax=tmax+40000

#t = np.linspace(0,thetmax,100000)
u,uI,connectivity,WEI,t = pickle.load(open('dyn_stimulation_SA.p','rb'))


#-----------------------------------------------------------------------------------------
#-------------------------------- Dynamics-----------------------------------------------
#----------------------------------------------------------------------------------------

#initial conditions
tmaxdyn=500
mystim.inten=0.
theintegrator=myintegrator(delay,dt,n,tmaxdyn)
theintegrator.fast=False


#integration

u_ret,uI_ret,connectivity_ret,WEI_ret,t_ret = pickle.load(open('dyn_retrieval_SA.p','rb'))
u_ret_PA,uI_ret_PA,connectivity_ret_PA,WEI_ret_PA,t_ret_PA = pickle.load(open('dyn_retrieval_PA.p','rb'))
#-------------------------------------------------------------------
#-----------------Stimulation of Populations------------------------
#-------------------------------------------------------------------

rc={'axes.labelsize': 32, 'font.size': 30, 'legend.fontsize': 25.0, 'axes.titlesize': 35}
plt.rcParams.update(**rc)
plt.rcParams['image.cmap'] = 'jet'

fig = plt.figure(figsize=(19, 11))
gs = gridspec.GridSpec(2, 2)#height_ratios=[3,3,2])
gs.update(wspace=0.44,hspace=0.03)
gs0 = gridspec.GridSpec(2, 2)
gs0.update(wspace=0.05,hspace=0.4,left=0.54,right=1.,top=0.88,bottom=0.1106)
#gs1.update(wspace=0.05,hspace=0.4,left=0.1245,right=1.,top=0.21,bottom=0.05)

# Excitatory and Inhibitory weights 
ax1A = plt.subplot(gs[0,0])
ax1B = plt.subplot(gs[1,0])
#sequence
axSA = plt.subplot(gs0[1,0])
axPA = plt.subplot(gs0[1,1])
#stimulation
ax2B= plt.subplot(gs0[0,0])
ax2C= plt.subplot(gs0[0,1])



colormap = plt.cm.Accent

ax2B.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax2B.plot(t,phi(u[:,:],theta,uc),lw=3)
mystim.inten=.1
elstim=np.array([sum(mystim.stim(x)) for x in t])
ax2B.plot(t,elstim,'k',lw=3)
ax2B.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
ax2B.set_ylim([0,1.2])
ax2B.set_xlim([0,600])
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
ax2C.set_xlim([89475,90075])
ax2C.set_xticks([89500,89700,89900])
ax2C.set_xticklabels([89.5,89.7,89.9])
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


ax1A.set_xticks([])
ax1A.axvline(x=tmax,ymin=0,ymax=2.,linewidth=2,ls='--',color='gray',alpha=0.7)
#ax1A.set_xticklabels([0,50,100,150])
ax1A.set_ylim([0,1.8])
ax1A.set_xlim([0,250000])
ax1A.set_yticks([0,0.5,1.,1.5])
#ax1A.set_xlabel('Time (s)')
ax1A.set_ylabel('Synaptic Weights')
ax1A.set_title('(A)',y=1.04)

#------------------------------------------------------------------------
#-------------Homeostatic Variable --------------------------------------
#------------------------------------------------------------------------


ax1B.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
ax1B.plot(t,WEI[:],lw=3)
ax1B.axvline(x=tmax,ymin=0,ymax=2.,linewidth=2,ls='--',color='gray',alpha=0.7)
ax1B.set_ylim([0.,3.4])
ax1B.set_yticks([0.,1.,2.,3.])
ax1B.set_xlim([0,250000])
ax1B.set_xticks([0,50000,100000,150000,200000,250000])
ax1B.set_xticklabels([0,50,100,150,200,250])
ax1B.set_xlabel('Time (s)')
ax1B.set_ylabel(r'$W_{EI}$')


#plot sequence
axSA.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
axSA.plot(t_ret,phi(u_ret[:,:],theta,uc),lw=5)
axSA.set_ylim([0,1.2])
axSA.set_xlim([0,370])
axSA.set_xticks([0,100,200,300])
axSA.set_yticks([0.5,1])
axSA.set_xlabel('Time (ms)')
axSA.set_ylabel('Rate')
#axSA.set_title('(C)',y=1.04)
axSA.set_title('(C)',x=1.028,y=1.04)

# plot PA
axPA.set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
axPA.plot(t_ret_PA,phi(u_ret_PA[:,:],theta,uc),lw=5)
axPA.set_ylim([0,1.2])
axPA.set_xlim([0,370])
axPA.set_xticks([0,100,200,300])
axPA.set_yticks([])
axPA.set_xlabel('Time (ms)')

#plt.show()
plt.savefig('fig6.pdf', bbox_inches='tight')




