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
wmax=3.
thres=1.5
#parameters stimulation
dt=0.5
lagStim=100.
times=15.
amp=3.


delta=8.
period=20.

a_post=1.
b_post=-.25
a_pre=1.
b_pre=-.25
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
W0=[(0.00001)*np.zeros((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False



rc={'axes.labelsize': 50, 'font.size': 40, 'legend.fontsize': 32, 'axes.titlesize': 30}
plt.rcParams.update(**rc)






#----------------------------------------------------------
# Transfer function, stationary leanring function and tau
#-----------------------------------------------------------
figure=plt.figure(figsize=(40,10))

learningrule1=figure.add_subplot(131)
current=np.linspace(-2.5,2.5,200)
tf,=learningrule1.plot(current,phi(current,theta,uc),'b',lw=4,label=r'$\phi(u)$')
learnmax,=learningrule1.plot(current,0.5*(1+np.tanh(a_post*current+b_post)),'g',lw=4,label=r'$f(u)=g(u)$')
#learnmax=learningrule1.plot(current,0.5*(1+np.tanh(-50.*(current-thres))),'m',lw=3,label=r'$\tau_{Pre}(u)=\tau_{Post}(u)$')
learningrule1.axvline(x=thres, ymin=-1., ymax = 2., linewidth=4,color='m',ls='dashed')
learningrule1.legend( (tf,learnmax),(r'$\phi(u)$',r'$f(u)=g(u)$'), loc = (0.1, 0.8) )
learningrule1.set_ylim([0,1.2])
learningrule1.set_yticks([0,0.4,0.8,1.2])
learningrule1.set_xlim([-2.5,2.5])
learningrule1.set_xlabel(r'$u$')
#learningrule1.legend(loc='upper left')
learningrule2=figure.add_subplot(132)
current1=np.linspace(-2.5,2.5,200)
current2=np.linspace(-2.5,2.5,200)
learningrule2.contourf(current1,current2,tauWinv([current1,current2]),10,alpha=0.5,cmap=plt.cm.autumn,origin='lower')
learningrule2.set_xlabel(r'$u_{Pre}$')
learningrule2.set_ylabel(r'$u_{Post}$')
learningrule3=figure.add_subplot(133)
current1=np.linspace(-2.5,2.5,200)
current2=np.linspace(-2.5,2.5,200)
learningrule3.contourf(current1,current2,winf([current1,current2]),10,alpha=0.5,cmap=plt.cm.autumn,origin='lower')
learningrule3.set_xlabel(r'$u_{Pre}$')
learningrule3.set_ylabel(r'$u_{Post}$')
plt.savefig('transferfunction.pdf', bbox_inches='tight')
plt.show()
#plt.show()

print 'tranferfunction.pdf is stored'

#---------------------------------------------------------------
# Qualitative  T vs  Delta diagaram
#---------------------------------------------------------------

myTv=np.linspace(0,2*delay,200)
myDeltav=np.linspace(0,2*delay,200)
valDelay=np.array([delay for i in range(200)])

plt.plot(myDeltav,valDelay,'k')
plt.plot(valDelay,myTv,'k')
plt.plot(myDeltav,(-myDeltav+delay)/2.,'k')
plt.fill_between(myDeltav[0:100],np.zeros(100),valDelay[0:100],alpha=0.5,edgecolor='k', facecolor='red')
plt.fill_between(myDeltav[100:200],np.zeros(100),valDelay[100:200],alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.fill_between(myDeltav[0:100],delay*np.ones(100),2*valDelay[0:100],alpha=0.5,edgecolor='k', facecolor='red')
plt.fill_between(myDeltav[100:200],delay*np.ones(100),2*valDelay[100:200],alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.fill_between(myDeltav[0:100],np.zeros(100),0.5*(delay-myDeltav[0:100]),alpha=0.5,edgecolor='k', facecolor='darkgrey')
plt.show()


#

#------------------------------------------------------
# New bifurcation diagram
#------------------------------------------------------

def myT(x):
	mywmax=(wmax/4.)*(1.+np.tanh(a_pre*amp+b_pre))*(1.+np.tanh(a_post*amp+b_post))
	return delay-(tau_learning/times)*np.log(1-x/mywmax)

def myDelta(x):
	mywmax=(wmax/4.)*(1.+np.tanh(a_pre*amp+b_pre))*(1.+np.tanh(a_post*amp+b_post))
	return delay+(tau_learning/times)*np.log(1-x/mywmax)

bifcurve=np.load('mybifcurve.npy')
mys=np.linspace(0,1.5,100)
myw=np.linspace(0,1.5,100)
upperBsequences=np.array([1+w_i*(1+0.)/n for j in range(0,len(bifcurve[:,1]))])

plt.plot(myDelta(bifcurve[:,0]),myT(bifcurve[:,1]),'k')
plt.plot(myDelta(mys),np.array([myT(1+(w_i+0.)/n) for i in range(0,100)]),c='k',lw=1)
plt.fill_between(myDelta(bifcurve[:,0]),myT(bifcurve[:,1]),myT(upperBsequences),alpha=0.5,edgecolor='k', facecolor='red')
plt.fill_between(myDelta(bifcurve[:,0]),myT(np.zeros(len(bifcurve[:,1]))),myT(bifcurve[:,1]),alpha=0.5, facecolor='darkgrey',linewidth=0)
plt.fill_between(myDelta(np.linspace(0,bifcurve[-1,0],100)),myT(np.zeros(100)),myT((1.+(w_i+0.)/n)*np.ones(100)),alpha=0.5,edgecolor='k', facecolor='darkgrey',linewidth=0)


colormap = plt.cm.winter 


alph=0.15
for i in range(1,n):
	for j in range(0,i):
		myline2=np.linspace(w_i*(j+0.)/n,w_i*(j+1.)/n,100)
		myconstant1=np.array([1+w_i*(i+0.)/n for l in range(0,100)])
		plt.fill_between(myDelta(myline2),myT(myconstant1),myT(myconstant1+w_i/n),alpha=alph,edgecolor='grey', facecolor=colormap((j+0.)/n)[0:3])
	alph=alph+(0.95-0.15)/9
for i in range(1,n):
	myline1=np.linspace(w_i*(i+0.)/n,1,100)
	myconstant1=np.array([1+w_i*(i+0.)/n for l in range(0,100)])
	plt.fill_between(myDelta(myline1),myT(myconstant1),myT(myconstant1+w_i/n),alpha=0.1*i,edgecolor='grey', facecolor='green')

myconstant1=np.array([2. for j in range(0,100)])
myconstant2=np.array([2.1 for j in range(0,100)])

alph=0.15
for j in range(0,n):
	myline2=np.linspace(w_i*(j+0.)/n,w_i*(j+1.)/n,100)
	plt.fill_between(myDelta(myline2),myT(myconstant1),myT(myconstant2),alpha=1.,edgecolor='grey', facecolor=colormap((j+0.)/n)[0:3])


plt.xlim([myDelta(1.),myDelta(0.)])
plt.ylim([myT(0),myT(2.1)])
#plt.yticks([1.5,1.6,1.7])
#plt.xticks([0.4,0.5,0.6])
plt.xlabel(r'$s$',fontsize='28')
plt.ylabel(r'$w$',fontsize='28')
plt.savefig('bifurcationdiagramLog.pdf', bbox_inches='tight')
plt.show()

print 'bifurcationdiagram.pdf stored'



#-------------------------------------------------------------
#-------------------stimulation degradation-------------------
#-------------------------------------------------------------

u,Wdiag,Woffdiag,connectivity,W01,t=theintegrator.DDE(field,x0,W0)

## Dynamics 
colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t,u[:,:],lw=2)
mystim.inten=.02
plt.plot(t,[mystim.stim(x) for x in t],'k',lw=2)
#plt.ylim([0,1.1])
plt.xlim([0,tmax])
#plt.xticks([0,200,400,600])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('stimulation1.pdf', bbox_inches='tight')
plt.show()


###dynamics synapses

timeWTheoric=np.linspace(40,tmax-40,times+1)
wTheoric=[0]
wk=0.
for l in range(int(times)):
	wk=wk+(wmax/4.)*(1.+np.tanh(a_pre*(amp+wk)+b_pre))*(1.+np.tanh(a_post*(amp+wk)+b_post))*(1.-np.exp(-(period-delay)/tau_learning))
	print wk
	wTheoric.append(wk)
print timeWTheoric, wTheoric
plt.plot(timeWTheoric,wTheoric)
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
plt.xlim([0,tmax])
plt.xticks([0,4000,8000,12000])
plt.yticks([0,.2,.4,.6,.8])
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitystimulation.pdf', bbox_inches='tight')
plt.show()


data=[connectivity[0,:,:],connectivity[int(len(t)/3.),:,:],connectivity[int(2*len(t)/3.),:,:],connectivity[-1,:,:]]
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixstimulation.pdf', bbox_inches='tight')
plt.show()

#################################################
######### sequences degradation #################

#fig 3c 
amp=1
times=15
delta=200.
period=10.
lagStim=1000
patterns=np.identity(n)
patterns=[patterns[:,0]]
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
tmax=times*(lagStim+(period+delta))+4
x0=np.zeros(n)
#x0[0]=10.
x0=np.array([x0 for i in range(npts)])
W0=[(1.0)*np.identity(n)+0.8*np.eye(n,k=-1) for i in range(npts)]
theintegrator_test=myintegrator(delay,dt,n,tmax)
theintegrator_test.fast=False
u_test,Wdiag_test,Woffdiag_test,connectivity_test,W0_test,t_test=theintegrator_test.DDE(field,x0,W0)

#Plotting
##outcomes sequence

# Have a look at the colormaps here and decide which one you'd like:
# http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xticks([0,2000,4000,6000,8000])
#plt.xlim([0,8])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequenceall.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xlim([0,300.])
plt.xticks([0,100,200,300])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencesfirst.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xlim([5100,5500.])
plt.xticks([5100,5200,5300,5400,5500])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencessecond.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xlim([5600,6000.])
plt.xticks([5600,5700,5800,5900,6000])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequencethird.pdf', bbox_inches='tight')
plt.show()

colormap = plt.cm.Accent
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.plot(t_test,phi_tanh(u_test[:,:]),lw=2)
plt.ylim([0,1.1])
plt.xlim([6100,6450.])
plt.xticks([6100,6200,6300,6400])
plt.xlabel('Time (ms)')
plt.ylabel('Rate')
plt.savefig('sequenceforth.pdf', bbox_inches='tight')
plt.show()

###dynamics synapses
for i in range(10):
		plt.plot(t_test,connectivity_test[:,i,i],'c',lw=2)
for i in range(9):
		plt.plot(t_test,connectivity_test[:,i+1,i],'y',lw=2)

for i in range(8):
		plt.plot(t_test,connectivity_test[:,i+2,i],'g',lw=2)
for i in range(9):
		plt.plot(t_test,connectivity_test[:,i,i+1],'r',lw=2)
for i in range(8):
		plt.plot(t_test,connectivity_test[:,i,i+2],'b',lw=2)
##plt.axhline(xmin=min(t),xmax=max(t),y=(wmax/4.)*(1.+np.tanh(a_post*(2.-np.exp(-period/tau))*amp+b_post))*(1+np.tanh(a_pre*amp*(1-np.exp(-period/tau))+b_pre)),linewidth=2,color='m',ls='dashed')
plt.ylim([0,1.2])
plt.xticks([0,2000,4000,6000,8000])
plt.yticks([0,.2,.4,.6,.8])
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic Weights')
plt.savefig('connectivitydegradation.pdf', bbox_inches='tight')
plt.show()

#connectivity matrices

data=[connectivity_test[0,:,:],connectivity_test[int(len(t_test)/3.),:,:],connectivity_test[int(2*len(t_test)/3.),:,:],connectivity_test[-1,:,:]]

#figure.colorbar(mymatrix,cax=cbaxes)
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
plt.savefig('matrixsequences.pdf', bbox_inches='tight')
plt.show()








