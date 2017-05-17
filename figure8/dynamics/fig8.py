import numpy as np
import matplotlib.gridspec as gridspec
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

fig = plt.figure(figsize=(30, 12))
gs = gridspec.GridSpec(1, 2)
gs.update(wspace=0.2,hspace=0.3)
gs0 = gridspec.GridSpec(2, 2)
gs0.update(wspace=0.13,hspace=0.1,left=0.55,right=0.95,top=0.9,bottom=0.1)
axbif = plt.subplot(gs[0,0])
ax1 = plt.subplot(gs0[0,0])
ax2= plt.subplot(gs0[0,1])
ax3 = plt.subplot(gs0[1,0])
ax4 = plt.subplot(gs0[1,1])

rc={'axes.labelsize':55, 'font.size': 45, 'legend.fontsize': 25, 'axes.titlesize': 35}
plt.rcParams.update(**rc)

list_axis=[ax1,ax2,ax3,ax4]
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
delta_T=[[50,40],[20.,8.5],[7,14],[5.,13.]]

# storing dynamics weights
allRecurrent=[]
allFF=[]
allRecurrentTheo=[]
allFFTheo=[]
# storing rate dynamics
mydyn=[]

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
	
		
	w_i=1.
	w_inh=w_i/n
	mystim.inten=0.
	tau_learning=30000.
	thetmax=5000.
	theintegrator=myintegrator(delay,dt,n,thetmax)
	theintegrator.fast=False
	adapt1,u1,connectivity1,W011,myH1,t1=theintegrator.DDE_Norm_Miller(field,a0_1,x0_1,W0_1,H0_1)
	mydyn.append(u1)
		
	print('------------------------------------------------')
	print 'Orbit with delta',eldelta,elT
	tau_learning=400.
	allRecurrentTheo.append(recurrentTheo(elT,100))
	allFFTheo.append(feedforwardTheo(elT,eldelta,100))
	allRecurrent.append(valueRecurrent)
	allFF.append(valueFF)


colormap = plt.cm.Accent
for l in range(len(mydyn)):
	#dynamics
	list_axis[l].set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,n)]))
	list_axis[l].plot(t1,phi(mydyn[l],theta,uc),lw=6)
	#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
	list_axis[l].set_ylim([0,1.2])
	if l==0:
		list_axis[l].set_yticks([0.5,1.])
		list_axis[l].set_xticks([])			
		list_axis[l].set_ylabel('Rate')
		list_axis[l].text(420, 1.05, 'PA')
	elif l==1:
		list_axis[l].set_yticks([])
		list_axis[l].set_xticks([])			
		list_axis[l].text(333, 1.05, 'PA/SA')
	elif l==2:
		list_axis[l].set_yticks([0.5,1.])
		list_axis[l].set_xticks([0,200,400])
		list_axis[l].set_xticklabels(['0','0.2','0.4'])
		list_axis[l].set_xlabel('Time (s)')
		list_axis[l].set_ylabel('Rate')
		list_axis[l].text(420, 1.05, 'SA')
	elif l==3:
		list_axis[l].set_yticks([])
		list_axis[l].set_xticks([0,200,400])			
		list_axis[l].set_xticklabels(['0','0.2','0.4'])
		list_axis[l].set_xlabel('Time (s)')
		list_axis[l].text(385, 1.05, 'dSA')
	list_axis[l].set_xlim([0,500])



#------------------------------------------------------------------
#---------------Bifurcation Diagram--------------------------------
#------------------------------------------------------------------
# This par of the code is to build a bifurcation diagram 
# that depends on the stimulation parameters T and delta -> period,delta



bifcurve=np.load('mybifcurve2.npy')
mys=np.linspace(0,2.0,100)
myw=np.linspace(0,2.0,100)

colormap = plt.cm.afmhot
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])


mymarkers=['o','+','h','s']
sizemarker=360
axbif.text(0.04, 1.7, 'PA')
axbif.text(1.61, 1.68, 'PA/SA')
axbif.text(1.55, 0.95, 'SA')
axbif.text(1.52, 0.1, 'dSA')

for i in range(norbits*nperiod):
	if i==1:
		axbif.scatter(allFF[i],allRecurrent[i],marker=mymarkers[i],s=sizemarker,facecolors='k')
	else:
		axbif.scatter(allFF[i],allRecurrent[i],marker=mymarkers[i],s=sizemarker,facecolors='none',edgecolors='k')

for i in range(norbits*nperiod):
	axbif.plot(allFFTheo[i],allRecurrentTheo[i],color='r',alpha=0.7,lw=7)


upperBsequences=np.array([1+w_i*(1+0.)/n for j in range(0,len(bifcurve[:,1]))])
axbif.plot(bifcurve[:,0],bifcurve[:,1],'k')
axbif.plot(mys,np.array([1+(w_i+0.)/n for i in range(0,100)]),c='k',lw=1)
axbif.fill_between(bifcurve[:,0],bifcurve[:,1],upperBsequences,alpha=0.5,edgecolor='k', facecolor='red',linewidth=0)
axbif.fill_between(np.linspace(bifcurve[0,0],2,100),np.zeros(100),(1.+(w_i+0.)/n)*np.ones(100),alpha=0.5,edgecolor='red', facecolor='red',linewidth=0)
axbif.fill_between(bifcurve[:,0],np.zeros(len(bifcurve[:,1])),bifcurve[:,1],alpha=0.5, facecolor='darkgrey',linewidth=0)
axbif.fill_between(np.linspace(0,bifcurve[-1,0],100),np.zeros(100),(1.+(w_i+0.)/n)*np.ones(100),alpha=0.5,edgecolor='k', facecolor='darkgrey',linewidth=0)


colormap = plt.cm.winter 


alph=0.15
for i in range(1,n):
	for j in range(0,i):
		myline2=np.linspace(w_i*(j+0.)/n,w_i*(j+1.)/n,100)
		myconstant1=np.array([1+w_i*(i+0.)/n for l in range(0,100)])
		axbif.fill_between(myline2,myconstant1,myconstant1+w_i/n,alpha=alph,edgecolor='grey', facecolor=colormap((j+0.)/n)[0:3])
	alph=alph+(0.95-0.15)/9
for i in range(1,n):
	myline1=np.linspace(w_i*(i+0.)/n,2.,100)
	myconstant1=np.array([1+w_i*(i+0.)/n for l in range(0,100)])
	axbif.fill_between(myline1,myconstant1,myconstant1+w_i/n,alpha=0.1*i,edgecolor='grey', facecolor='green')

myconstant1=np.array([2. for j in range(0,100)])
myconstant2=np.array([2. for j in range(0,100)])

alph=0.15
for j in range(0,n):
	myline2=np.linspace(w_i*(j+0.)/n,w_i*(j+1.)/n,100)
	axbif.fill_between(myline2,myconstant1,myconstant2,alpha=1.,edgecolor='grey', facecolor=colormap((j+0.)/n)[0:3])


axbif.set_xlim([0.,2.])
axbif.set_ylim([0,2.])
axbif.set_yticks([1,2])
axbif.set_xticks([0,1.,2.])
axbif.set_xlabel(r'Feed-forward ($s$)')
axbif.set_ylabel(r'Recurrent ($w$)')

fig.savefig('fig8.pdf', bbox_inches='tight')
#plt.show()






