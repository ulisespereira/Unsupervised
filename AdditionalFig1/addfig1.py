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
	field_u=(1/tau)*(mystim.stim(t)+W.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi(x_hist[-1],theta,uc)))
	field_w=np.multiply(tauWinv(x_hist),(-W+winf(x_hist)))
	return field_u,field_w




#This are a the parameters of the simulation
#-------------------------------------------------------------------------------------
#------------------------Parameter Model----------------------------------------------
#-------------------------------------------------------------------------------------


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
lagStim=400.


amp=20.
delta=15.3
period=40.
times=240

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



#-------------------------------------------------------------
#-------------------stimulation Network-------------------
#------------------------------------------------------------
mywsum=[1.4]
delta=3.
period=22.
for wsum in mywsum:
	print 'We are in the ',wsum,' of ',mywsum[-1]
	amp_dc=0.
	amp=3.2-amp_dc
	times=600
	mystim=stimulus(patterns,lagStim,delta,period,times)
	mystim.inten=amp
	mystim.amp_dc=amp_dc
	tmax=times*(lagStim+n*(period+delta))+20000
	tmax_long=tmax
	#initial conditions
	x0=0.01*np.ones((npts,n))
	W0=[(wsum/n)*np.ones((n,n)) for i in range(npts)]
	theintegrator=myintegrator(delay,dt,n,tmax)
	theintegrator.fast=False

	u,Wdiag,Woffdiag,connectivity,W01,t=theintegrator.DDE_Norm_additive(field,x0,W0)


	#retrieval
	amp=0.
	times=100
	mystim=stimulus(patterns,lagStim,delta,period,times)
	mystim.inten=amp
	mystim.amp_dc=0.
	tmax=500
	#initial conditions
	x0=np.zeros((npts,n))
	x0[:,0]=1.
	W0=[connectivity[-1] for i in range(npts)]
	theintegrator=myintegrator(delay,dt,n,tmax)
	theintegrator.fast=False
	u_ret,Wdiag_ret,Woffdiag_ret,connectivity_ret,W01_ret,t_ret=theintegrator.DDE_Norm_additive(field,x0,W0)


	#-------------------------------------------------------------------------------------
	#----------------Plotting--------------------------------------------------------------
	#---------------------------------------------------------------------------------------

	rc={'axes.labelsize': 50, 'font.size': 40, 'legend.fontsize': 25, 'axes.titlesize': 30}
	plt.rcParams.update(**rc)
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
	plt.savefig('learning_stim1_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png',bbox_inches='tight', dpi=1200)
	#plt.show()
	plt.close()


	colormap = plt.cm.Accent
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
	plt.plot(t,phi(u[:,:],theta,uc),lw=3)
	elstim=np.array([sum(mystim.stim(x)) for x in t])
	plt.plot(t,elstim,'k',lw=3)
	plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
	plt.ylim([0,1.2])
	time_plot=10*(lagStim+n*(period+delta))
	plt.xlim([time_plot,time_plot+400])
	plt.xticks([time_plot,time_plot+200,time_plot+400])
	plt.yticks([0,0.4,0.8,1.2])
	plt.xlabel('Time (ms)')
	plt.ylabel('Rate')
	plt.savefig('learning_stim2_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png', bbox_inches='tight',dpi=1200)
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
	time_plot=100*(lagStim+n*(period+delta))
	plt.xlim([time_plot,time_plot+400])
	plt.xticks([time_plot,time_plot+200,time_plot+400])
	plt.yticks([0,0.4,0.8,1.2])
	plt.xlabel('Time (ms)')
	plt.ylabel('Rate')
	plt.savefig('learning_stim3_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png', bbox_inches='tight',dpi=1200)
	#plt.show()
	plt.close()

	colormap = plt.cm.Accent
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
	plt.plot(t_ret,phi(u_ret[:,:],theta,uc),lw=3)
	plt.ylim([0,1.2])
	plt.xlim([0,500])
	plt.yticks([0,0.4,0.8,1.2])
	plt.xlabel('Time (ms)')
	plt.ylabel('Rate')
	plt.savefig('retrival_dyn_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png', bbox_inches='tight',dpi=1200)
	#plt.show()
	plt.close()


	colormap = plt.cm.Accent
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
	plt.plot(t,phi(u[:,:],theta,uc),lw=3)
	elstim=np.array([sum(mystim.stim(x)) for x in t])
	plt.plot(t,elstim,'k',lw=3)
	plt.fill_between(t,np.zeros(len(t)),elstim,alpha=0.5,edgecolor='k', facecolor='darkgrey')
	plt.xlim([0,500000])
	plt.yticks([0,0.4,0.8,1.2])
	#plt.xticks([0,1000,2000,3000,4000,5000,6000],[0,1,2,3,4,5,6])
	plt.ylim([0,1.2])
	plt.xlabel('Time (s)')
	plt.ylabel('Rate')
	plt.savefig('learning_stimall_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png', bbox_inches='tight',dpi=1200)
	#plt.show()
	plt.close()

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
	plt.xlim([0,tmax_long])
	plt.xticks([0,100000,200000,300000,400000],[0,100,200,300,400])
	plt.yticks([0,0.4,0.8,1.2])
	plt.ylim([0,1.2])
	plt.xlabel('Time (s)')
	plt.ylabel('Synaptic Weights')
	plt.savefig('learning_conndyn_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png', bbox_inches='tight',dpi=1200)
	#plt.show()
	plt.close()


	###dynamics synapses

	for i in range(10):
			plt.plot(t_ret,connectivity_ret[:,i,i],'c',lw=3)
	for i in range(0,9):
			plt.plot(t_ret,connectivity_ret[:,i+1,i],'y',lw=3)
	for i in range(8):
			plt.plot(t_ret,connectivity_ret[:,i+2,i],'g',lw=3)
	for i in range(9):
			plt.plot(t_ret,connectivity_ret[:,i,i+1],'r',lw=3)
	for i in range(8):
			plt.plot(t_ret,connectivity_ret[:,i,i+2],'b',lw=3)
	plt.xlim([0,500])
	#plt.xticks([0,5000,10000,15000,20000,25000,30000],[0,5,10,15,20,25,30])
	#plt.yticks([0,1.,2.,3.])
	plt.xlabel('Time (s)')
	plt.ylabel('Synaptic Weights')
	plt.savefig('retrival_conndyn_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png', bbox_inches='tight')
	#plt.show()
	plt.close()



	data=[connectivity[0,:,:],connectivity[int(tmax_long/(dt)),:,:]]#connectivity[int((2*tmax_long)/(dt*3)),:,:],connectivity[int(tmax_long/dt),:,:]]
	fig, axes = plt.subplots(nrows=1, ncols=2)
	for dat, ax in zip(data, axes.flat):
		    # The vmin and vmax arguments specify the color limit
		im = ax.matshow(dat, vmin=0, vmax=1.4)
		# Make an axis for the colorbar on the right side
	cax = fig.add_axes([0.93, 0.24, 0.03, 0.54])
	fig.colorbar(im, cax=cax,ticks=[0,1.4])
	plt.savefig('learning_matrix_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png', bbox_inches='tight')
	#plt.show()
	plt.close()



	data=[connectivity_ret[0,:,:],connectivity_ret[int(150./dt),:,:],connectivity_ret[int(200./dt),:,:],connectivity_ret[int(500./dt),:,:]]
	fig, axes = plt.subplots(nrows=2, ncols=2)
	for dat, ax in zip(data, axes.flat):
		    # The vmin and vmax arguments specify the color limit
		im = ax.matshow(dat, vmin=0, vmax=wsum)
		# Make an axis for the colorbar on the right side
	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	fig.colorbar(im, cax=cax,ticks=[0,1.35])
	plt.savefig('retrival_matrix_sumrow_'+str(wsum)+'_delta_'+str(delta)+'_T_'+str(period)+'.png', bbox_inches='tight')
	#plt.show()
	plt.close()




