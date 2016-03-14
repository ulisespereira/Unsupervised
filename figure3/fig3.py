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
times=105.

amp=4.


delta=8.
period=20.

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






#----------------------------------------------------------
# Transfer function, stationary leanring function and tau
#-----------------------------------------------------------
figure=plt.figure(figsize=(25,10))

learningrule1=figure.add_subplot(121)
current=np.linspace(-2.5,7,400)
tf,=learningrule1.plot(current,phi(current,theta,uc),'b',lw=4,label=r'$\phi(u)$')
learnmax,=learningrule1.plot(current,0.5*(1+np.tanh(a_post*current+b_post)),'g',lw=4,label=r'$f(u)=g(u)$')
#learnmax=learningrule1.plot(current,0.5*(1+np.tanh(-50.*(current-thres))),'m',lw=3,label=r'$\tau_{Pre}(u)=\tau_{Post}(u)$')
learningrule1.axvline(x=thres, ymin=-1., ymax = 2., linewidth=4,color='darkgrey',ls='dashed')
learningrule1.legend( (tf,learnmax),(r'$\phi(u)$',r'$f(u)=g(u)$'), loc = (0.01, 0.8) )
learningrule1.set_ylim([0,1.2])
learningrule1.set_xlim([-2,6])
learningrule1.set_yticks([0,0.4,0.8,1.2])
learningrule1.set_xticks([-2,0,2,4,6])
learningrule1.set_xlabel(r'$u$')
#learningrule1.legend(loc='upper left')
learningrule3=figure.add_subplot(122)
current1=np.linspace(-1.,6,200)
current2=np.linspace(-1.,6,200)
myplot=learningrule3.contourf(current1,current2,winf([current1,current2]),10,alpha=0.5,cmap=plt.cm.autumn,origin='lower')
learningrule3.axvline(x=0.9, ymin=1.9/7., ymax = 1, linewidth=4,color='darkgrey',ls='dashed')
learningrule3.axhline(y=0.9, xmin=1.9/7., xmax = 1, linewidth=4,color='darkgrey',ls='dashed')
learningrule3.set_xlabel(r'$u_{Pre}$')
learningrule3.set_ylabel(r'$u_{Post}$')
plt.colorbar(myplot,ticks=[0,1,2,3])
figure.savefig('transferfunction.pdf', bbox_inches='tight')
#plt.show()
plt.close(figure)

print 'tranferfunction.pdf is stored'

#---------------------------------------------------------------
# Qualitative  T vs  Delta diagaram
#---------------------------------------------------------------

myTv=np.linspace(0,2*delay,200)
myDeltav=np.linspace(0,2*delay,200)
valDelay=np.array([delay for i in range(200)])

myalpha=0.3
plt.plot(myDeltav,valDelay,'k')
plt.plot(valDelay,myTv,'k')
plt.plot(myDeltav,(-myDeltav+delay)/2.,'k')
plt.fill_between(myDeltav[0:100],0.5*(delay-myDeltav[0:100]),valDelay[0:100],alpha=myalpha,edgecolor='k', facecolor='green')
plt.fill_between(myDeltav[100:200],np.zeros(100),valDelay[100:200],alpha=myalpha,edgecolor='k', facecolor='yellow')
plt.fill_between(myDeltav[0:100],delay*np.ones(100),2*valDelay[0:100],alpha=myalpha,edgecolor='k', facecolor='red')
plt.fill_between(myDeltav[100:200],delay*np.ones(100),2*valDelay[100:200],alpha=myalpha,edgecolor='k', facecolor='blue')
plt.fill_between(myDeltav[0:100],np.zeros(100),0.5*(delay-myDeltav[0:100]),alpha=0.2,edgecolor='k', facecolor='darkgrey')
plt.ylim([0,2*delay])
plt.xlim([0,2*delay])
plt.xticks([0,15,30])
plt.yticks([15,30])
plt.xlabel(r'$\Delta$')
plt.ylabel(r'$T$')
plt.savefig('qualitativediagram.pdf', bbox_inches='tight')
plt.close()
#plt.show()

print 'qualitativediagram.pdf is stored'

timesmax=300
#----------------------------------------------------
#------- Examples Qualitative Diagram ---------------
#----------------------------------------------------
amp=5
timesmax=300
# region 1
delta=14.5
period=15.5
times=timesmax
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
tmax=times*(lagStim+n*(period+delta))+40
#initial conditions
x0=0.01*np.ones((npts,n))
W0=[(0.1)*np.ones((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False
u,Wdiag,Woffdiag,connectivity,W01,t=theintegrator.DDE(field,x0,W0)



timeWTheoric=np.linspace(40,tmax-40,times+1)
# Ploting the theoretical line for recurrent connections learned
wTheoric=[0]
wk=0.
for l in range(int(times)):
	wk=wk+(wmax/4.)*(1.+np.tanh(a_pre*(amp+wk)+b_pre))*(1.+np.tanh(a_post*(amp+wk)+b_post))*(1.-np.exp(-(period-delay)/tau_learning))
	wTheoric.append(wk)
#plt.plot(timeWTheoric,wTheoric,'c--',lw=3)
#Ploting the theoretical line for the feed forward connections learned
wTheoric=[0]
wk=0.
for l in range(int(times)):
	wk=wk+(wmax/4.)*(1.+np.tanh(a_pre*(amp+wk)+b_pre))*(1.+np.tanh(a_post*(amp+wk)+b_post))*(1.-np.exp(-(delay-delta)/tau_learning))
	wTheoric.append(wk)

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
plt.xticks([0,40000,80000,120000],['0','40','80','120'])
plt.yticks([0,.4,.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('dynamicsweights1.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'dynamicsweights1.pdf is stored'

# region 2
amp=4
delta=120.
period=15.5
times=timesmax
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
tmax=times*(lagStim+n*(period+delta))+40
#initial conditions
x0=0.01*np.ones((npts,n))
W0=[(0.1)*np.ones((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False
u,Wdiag,Woffdiag,connectivity,W01,t=theintegrator.DDE(field,x0,W0)



timeWTheoric=np.linspace(40,tmax-40,times+1)
# Ploting the theoretical line for recurrent connections learned
wTheoric=[0]
wk=0.
for l in range(int(times)):
	wk=wk+(wmax/4.)*(1.+np.tanh(a_pre*(amp+wk)+b_pre))*(1.+np.tanh(a_post*(amp+wk)+b_post))*(1.-np.exp(-(period-delay)/tau_learning))
	wTheoric.append(wk)
#plt.plot(timeWTheoric,wTheoric,'y--',lw=3)
#Ploting the theoretical line for the feed forward connections learned
wTheoric=[0]
wk=0.
for l in range(int(times)):
	wk=wk+(wmax/4.)*(1.+np.tanh(a_pre*(amp+wk)+b_pre))*(1.+np.tanh(a_post*(amp+wk)+b_post))*(1.-np.exp(-(delay-delta)/tau_learning))
	wTheoric.append(wk)
#plt.plot(timeWTheoric,wTheoric,'c--',lw=3)




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
#plt.xticks([0,5000,10000])
plt.xticks([0,100000,200000,300000,400000],[0,100,200,300,400])
#plt.xticks([0,40000,80000,120000],['0','40','80','120'])
plt.yticks([0,.4,.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('dynamicsweights2.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'dynamicsweights2.pdf is stored'
# region 3
delta=50.
period=1.
times=timesmax
mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
tmax=times*(lagStim+n*(period+delta))+40
#initial conditions
x0=0.01*np.ones((npts,n))
W0=[(0.1)*np.ones((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False
u,Wdiag,Woffdiag,connectivity,W01,t=theintegrator.DDE(field,x0,W0)

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
#plt.xticks([0,4000,8000])
plt.xticks([0,40000,80000,120000],['0','40','80','120'])
plt.yticks([0,.4,.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('dynamicsweights3.pdf', bbox_inches='tight')
#plt.show()
plt.close()

print 'dynamicsweights3.pdf is stored'
# region 4
amp=3.
delta=2.
period=14.
times=2*timesmax

mystim=stimulus(patterns,lagStim,delta,period,times)
mystim.inten=amp
tmax=times*(lagStim+n*(period+delta))+40
#initial conditions
x0=0.01*np.ones((npts,n))
W0=[(0.1)*np.ones((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False
u,Wdiag,Woffdiag,connectivity,W01,t=theintegrator.DDE(field,x0,W0)

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
plt.xlim([0,120000])
#plt.xticks([0,2600,5200])
plt.xticks([0,40000,80000,120000],['0','40','80','120'])
plt.yticks([0,.4,.8,1.2])
plt.xlabel('Time (s)')
plt.ylabel('Synaptic Weights')
plt.savefig('dynamicsweights4.pdf', bbox_inches='tight')
#plt.show()
plt.close()


print 'dynamicsweights4.pdf is stored'



