import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
from stimulus import *
from myintegrator import *
import cProfile
import json
import scipy.integrate as integrate
import pickle
from mpl_toolkits.axes_grid.inset_locator import inset_axes
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
	myresult[x>thres]=(1/tau_learning)#*0.5*(1+np.tanh(a_tau*(x[x>thres]+b_tau)))
	return myresult

def winf(x_hist):
	pre_u=phi(x_hist[0],theta,uc)
	post_u=phi(x_hist[-1],theta,uc)
	mynu=5.5
	mytheta=-0.8
	#parameters
	n=len(pre_u)
	vec_pre=0.5*(np.ones(n)+np.tanh(a_pre*(pre_u-b_pre)))
	return (wmax/2.)*np.outer((np.ones(n)+np.tanh(a_post*(post_u-b_post))),vec_pre)
	#return (wmax/2.)*np.outer(vec_post,vec_pre)

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest

def tauWinv(x_hist):
	pre_u=x_hist[0]
	post_u=x_hist[-1]
	n=len(pre_u)
	return  tau_learning*np.outer(mytauInv(post_u),mytauInv(pre_u))

def F(u):
	if theta<=u and u<=uc:
		r=nu*(u-theta)
	elif u<=theta:
		r=0
	elif uc<=u:
		r=nu*(uc-theta)
	return np.sqrt(wmax)*.5*(1.+np.tanh(a_post*(r-b_post)))


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
#This are a the parameters of the simulation

#open parameters of the model
n=10 #n pop
delay=15.3
tau=10.   #timescale of populations
tau_H=200000.
af=0.1
bf=0.
y0=.12*np.ones(n)
w_i=1.
w_inh=w_i/n
nu=1.
theta=0.
uc=1.
wmax=1.8
thres=0.6
beta=1.6
tau_a=10.
#parameters stimulation
dt=0.5
lagStim=100.
times=100
amp=5.5


bf=10.
xf=0.7
a_post=bf
b_post=xf
a_pre=bf
b_pre=xf
tau_learning=400.

x=np.linspace(0,6,100)


a1=6.
b1=-0.25


w0=0.01

#---------------------------------------------------------------
#----------------Learning vector Field--------------------------
#---------------------------------------------------------------




def tau_u0_theta(T):
	return -tau*np.log(1.-(thres/amp))

def tau_umax_theta(T):
	return -tau*np.log((thres/amp)*(1./(1.-np.exp(-T/tau))))
def tau_theta(T):
	return T-tau_u0_theta(T)+tau_umax_theta(T)

#approximation pupulation dynamics
def myu(t,T,tstart):
	ttilda=t-tstart
	if tau_u0_theta(T)<=ttilda and ttilda<=T:
		return amp*(1.-np.exp(-ttilda/tau))
	elif ttilda>T:
		return amp*(1.-np.exp(-T/tau))*np.exp(-(ttilda-T)/tau)
	elif tau_u0_theta(T)>ttilda:
		return amp*(1.-np.exp(-ttilda/tau))
	elif ttilda<0:
		print 'holi',ttilda

def recurrentTheo(T,k):
	tstart=0.
	myt=tstart+T+tau_umax_theta(T)
	wdyn=[w0]
	wk=w0
	for i in range(k):
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart))*np.exp((x-delay-tstart-tau_u0_theta(T))/tau_learning) 
		myintegral=lambda y:integrate.quad(df,tstart+delay+tau_u0_theta(T),y,epsabs=1e-5)
		val,err=myintegral(myt)
		wk=np.exp(-(myt-(delay+tau_u0_theta(T)+tstart))/tau_learning)*(wk+val*(1./tau_learning))
		wdyn.append(wk)
	return wdyn

def feedforwardTheo(T,delta,k):
	tstart=0.
	myt=tstart+delay-delta+tau_umax_theta(T)
	wdyn=[w0]
	wk=w0
	for i in range(k):
		df=lambda x:F(myu(x,T,tstart))*F(myu(x-delay,T,tstart-delta-T))*np.exp((x-tau_u0_theta(T)-tstart)/tau_learning) 
		myintegral=lambda y:integrate.quad(df,tau_u0_theta(T)+tstart,y,epsabs=1e-5)
		val,err=myintegral(myt)
		wk=np.exp(-(myt-(tstart+tau_u0_theta(T)))/tau_learning)*(wk+val*(1./tau_learning))
		wdyn.append(wk)
	return wdyn

#-------------------------------------------------------------------
#-----------------Stimulation of Populations------------------------
#-------------------------------------------------------------------

# setting up the simulation 


r1_matrix=np.ones((n,n))
patterns=np.identity(n)
patterns=[patterns[:,i] for i in range(n)]
npts=int(np.floor(delay/dt)+1)         # points delay
#initial conditions
a0=np.zeros((npts,n))
x0=0.01*np.ones((npts,n))
W0=[w0*np.ones((n,n)) for i in range(npts)]
H0=[0.01*np.ones(n) for i in range(npts)]
norbits=2
nperiod=2

#delta_T=[[21.,9.5],[18.,15.],[23.,12.],[28.,16.]]
delta_T=[[20.,8.5],[5.,13.],[7.,14.],[50.,40.]]
allRecurrent=[]
allFF=[]
allRecurrentTheo=[]
allFFTheo=[]
for param in delta_T:
	elT=param[0]
	eldelta=param[1]
	valueRecurrent=[w0]
	valueFF=[w0]
	mystim=stimulus(patterns,lagStim,eldelta,elT,times)
	mystim.inten=amp
	times=150
	#integrator
	tmax=times*(lagStim+n*(elT+eldelta))+mystim.delay_begin+5000.
	thetmax=tmax
	theintegrator=myintegrator(delay,dt,n,thetmax)
	theintegrator.fast=False
	tau_learning=400.	
	w_i=1.
	w_inh=w_i/n
	adapt,u,connectivity,W01,myH,t=theintegrator.DDE_Norm_Miller(field,a0,x0,W0,H0)
	for k in range(times):
		eltiempo=int((mystim.delay_begin+k*n*(elT+eldelta)+lagStim*(k-1)+lagStim/2.)/dt)
		valueRecurrent.append(connectivity[eltiempo,5,5])
		valueFF.append(connectivity[eltiempo,6,5])

	a0_1=np.zeros((npts,n))
	x0_1=np.zeros((npts,n))
	x0_1[:,0]=np.ones(npts)
	W0_1=[connectivity[-1,:,:] for i in range(npts)]
	H0_1=[np.ones(n) for i in range(npts)]
		
	a0_2=np.zeros((npts,n))
	x0_2=np.ones((npts,n))
	W0_2=[connectivity[-1,:,:] for i in range(npts)]
	H0_2=[np.ones(n) for i in range(npts)]
	
	rc={'axes.labelsize': 50, 'font.size': 50, 'legend.fontsize': 50., 'axes.titlesize': 50}
		
	w_i=1.
	w_inh=w_i/n
	mystim.inten=0.
	tau_learning=30000.
	thetmax=5000.
	theintegrator=myintegrator(delay,dt,n,thetmax)
	theintegrator.fast=False
	adapt1,u1,connectivity1,W011,myH1,t1=theintegrator.DDE_Norm_Miller(field,a0_1,x0_1,W0_1,H0_1)
	#adapt2,u2,connectivity2,W012,myH2,t2=theintegrator.DDE_Norm_Miller(field,a0_2,x0_2,W0_2,H0_2)
	figure=plt.figure(figsize=(25,10))
	colormap = plt.cm.Accent
	#dynamics
	dynamics=figure.add_subplot(111)
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
	dynamics.plot(t1,phi(u1,theta,uc),lw=6)
	#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
	dynamics.tick_params(labelsize=55)
	dynamics.set_yticks([0,0.4,0.8,1.2])
	dynamics.set_xticks([0,100,200,300,400,500])
	dynamics.set_xlim([0,500])
	dynamics.set_ylim([0,1.2])
	dynamics.set_xlabel('Time (ms)',fontsize=75)
	dynamics.set_ylabel('Rate',fontsize=75)

	#inset_axes = inset_axes(dynamics,width="50%",height=1.0,loc=1)
	#a = plt.plot(t1,phi(u1,theta,uc))
	#plt.title('Probability')
	#plt.xticks([])
	#plt.yticks([])
	
#	#dynamics
#	dynamics2=figure.add_subplot(312)
#	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
#	dynamics2.plot(t1,myH1,lw=2.)
#	dynamics2.set_xlim([0,1000])
#	dynamics2.set_ylim([0,1.2])
#	dynamics2.set_xticks([0,200,400,600,800,1000],['0','1','2','3','4','5'])
#	dynamics2.set_yticks([0,0.4,0.8,1.2])
#	dynamics2.set_xlabel('Time (ms)')
#	dynamics2.set_ylabel('H',fontsize=18)
#	
#	dynamics3=figure.add_subplot(313)
#	for i in range(10):
#		dynamics3.plot(t1,connectivity1[:,i,i],'c',lw=3)
#	for i in range(0,9):
#		dynamics3.plot(t1,connectivity1[:,i+1,i],'y',lw=3)	
#	
#	dynamics3.set_xlim([0,1000])
#	dynamics3.set_ylim([0,2.0])
#	dynamics3.set_yticks([0,0.5,1.,1.5,2.])
#	dynamics3.set_xticks([0,200,400,600,800,1000],['0','1','2','3','4','5'])
#	dynamics3.set_xlabel('Time (ms)',fontsize=18)
#	dynamics3.set_ylabel('Synaptic Weights',fontsize=18)
#
	name='dynamics_'+str(elT)+'_'+str(eldelta)+'.pdf'
	plt.savefig(name, bbox_inches='tight')
	plt.close()
	#plt.show()
	
	figure2=plt.figure(figsize=(25,10))
	dynamicszoom=figure2.add_subplot(111)
	plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
	dynamicszoom.plot(t1,phi(u1,theta,uc),lw=6)
	dynamicszoom.set_xlim([0,300])
	dynamicszoom.set_ylim([0,1.2])
	dynamicszoom.tick_params(labelsize=70)
	dynamicszoom.set_xticks([0,100,200,300])
	dynamicszoom.set_yticks([0,0.4,0.8,1.2])
	dynamicszoom.set_xlabel('Time (ms)',fontsize=80)
	dynamicszoom.set_ylabel('Rate',fontsize=80)
	name='dynamics_'+str(elT)+'_'+str(eldelta)+'_zoom.pdf'
	plt.savefig(name, bbox_inches='tight')
	plt.close()
		
	print('------------------------------------------------')
	print 'Orbit with delta',eldelta,elT
	tau_learning=400.
	allRecurrentTheo.append(recurrentTheo(elT,100))
	allFFTheo.append(feedforwardTheo(elT,eldelta,100))
	allRecurrent.append(valueRecurrent)
	allFF.append(valueFF)
#
#the_filename1='recurrent_dynamics.dat'
#with open(the_filename1, 'wb') as f:
#	    pickle.dump(allRecurrent, f)
#
#the_filename2='ff_dynamics.dat'
#with open(the_filename2, 'wb') as f:
#	    pickle.dump(allFF, f)

#------------------------------------------------------------------
#---------------Bifurcation Diagram--------------------------------
#------------------------------------------------------------------
# This par of the code is to build a bifurcation diagram 
# that depends on the stimulation parameters T and delta -> period,delta



rc={'axes.labelsize': 30, 'font.size': 22, 'legend.fontsize': 28.0, 'axes.titlesize': 30}


bifcurve=np.load('mybifcurve2.npy')
mys=np.linspace(0,2.0,100)
myw=np.linspace(0,2.0,100)

colormap = plt.cm.afmhot
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])


#with open(the_filename1, 'rb') as f:
#	    allRecurrent = pickle.load(f)

#with open(the_filename2, 'rb') as f:
#	    allFF = pickle.load(f)
mymarkers=['o','+','h','s']

for i in range(norbits*nperiod):
	plt.plot(allFF[i],allRecurrent[i],'ko',alpha=0.6,lw=1,marker=mymarkers[i],markersize=12.)

for i in range(norbits*nperiod):
	plt.plot(allFFTheo[i],allRecurrentTheo[i],color='r',alpha=0.7,lw=4)


upperBsequences=np.array([1+w_i*(1+0.)/n for j in range(0,len(bifcurve[:,1]))])
plt.plot(bifcurve[:,0],bifcurve[:,1],'k')
plt.plot(mys,np.array([1+(w_i+0.)/n for i in range(0,100)]),c='k',lw=1)
plt.fill_between(bifcurve[:,0],bifcurve[:,1],upperBsequences,alpha=0.5,edgecolor='k', facecolor='red',linewidth=0)
plt.fill_between(np.linspace(bifcurve[0,0],2,100),np.zeros(100),(1.+(w_i+0.)/n)*np.ones(100),alpha=0.5,edgecolor='red', facecolor='red',linewidth=0)
plt.fill_between(bifcurve[:,0],np.zeros(len(bifcurve[:,1])),bifcurve[:,1],alpha=0.5, facecolor='darkgrey',linewidth=0)
plt.fill_between(np.linspace(0,bifcurve[-1,0],100),np.zeros(100),(1.+(w_i+0.)/n)*np.ones(100),alpha=0.5,edgecolor='k', facecolor='darkgrey',linewidth=0)


colormap = plt.cm.winter 


alph=0.15
for i in range(1,n):
	for j in range(0,i):
		myline2=np.linspace(w_i*(j+0.)/n,w_i*(j+1.)/n,100)
		myconstant1=np.array([1+w_i*(i+0.)/n for l in range(0,100)])
		plt.fill_between(myline2,myconstant1,myconstant1+w_i/n,alpha=alph,edgecolor='grey', facecolor=colormap((j+0.)/n)[0:3])
	alph=alph+(0.95-0.15)/9
for i in range(1,n):
	myline1=np.linspace(w_i*(i+0.)/n,2.,100)
	myconstant1=np.array([1+w_i*(i+0.)/n for l in range(0,100)])
	plt.fill_between(myline1,myconstant1,myconstant1+w_i/n,alpha=0.1*i,edgecolor='grey', facecolor='green')

myconstant1=np.array([2. for j in range(0,100)])
myconstant2=np.array([2. for j in range(0,100)])

alph=0.15
for j in range(0,n):
	myline2=np.linspace(w_i*(j+0.)/n,w_i*(j+1.)/n,100)
	plt.fill_between(myline2,myconstant1,myconstant2,alpha=1.,edgecolor='grey', facecolor=colormap((j+0.)/n)[0:3])


plt.xlim([0.,2.])
plt.ylim([0,2.])
plt.yticks([0.5,1,1.5,2],fontsize='30')
plt.xticks([0,1.,2.],fontsize='30')
plt.xlabel(r'$s$',fontsize='35')
plt.ylabel(r'$w$',fontsize='35')
plt.savefig('bifdiagramTvsDel1.pdf', bbox_inches='tight')
#plt.show()






