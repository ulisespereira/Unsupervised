import numpy as np
from scipy import sparse
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt
import math as mt
from stimulus import *
from myintegrator import *
import cProfile
import json
import matplotlib.gridspec as gridspec


plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# this is the transfer function 
def phi(x,theta,uc):
	myresult=nu*(x-theta)
	myresult[x<theta]=0.
	myresult[x>uc]=nu*(uc-theta)
	return myresult


def mytau(x): #time scale function synapses
	myresult=(1e50)*np.ones(len(x))
	myresult[x>thres]=tau_learning
	#print x>thres
	#print x
	#myresult=(1e8)*(1.+np.tanh(-50.*(x-thres)))+tau_learning
	#print myresult
	return myresult

def winf(x_hist):
	pre_u=phi(x_hist[0],theta,uc)
	post_u=phi(x_hist[-1],theta,uc)
	#parameters
	n=len(pre_u)
	return (wmax/4.)*np.outer((np.ones(n)+np.tanh(a_post*(post_u-b_post))),(np.ones(n)+np.tanh(a_pre*(pre_u-b_pre))))

#function for the field
#x_hist is the 'historic' of the x during the delay period the zero is the oldest and the -1 is the newest

def tauWinv(x_hist):
	pre_u=phi(x_hist[0],theta,uc)
	post_u=phi(x_hist[-1],theta,uc)

	tau_inv =   np.add.outer(1/mytau(post_u),1/mytau(pre_u))
	tau_inv[tau_inv == 2. / tau_learning] = 1./tau_learning
	return tau_inv
	#return tau_learning*np.outer(1./mytau(post_u),1./mytau(pre_u))



def field(t,x_hist,W):
	field_u=(1/tau)*(mystim.stim(t)+W.dot(phi(x_hist[-1],theta,uc))-x_hist[-1]-w_inh*np.dot(r1_matrix,phi(x_hist[-1],theta,uc)))
	field_w=np.multiply(tauWinv(x_hist),(-W+winf(x_hist)))
	return field_u,field_w

#script to save the parameters

# computing boundary plot

#This are a the parameters of the simulation

#open parameters of the model
n=10 #n pop
delay=15.3 #multilpl:es of 9!
tau=10.   #timescale of populations
w_i=1.
nu=1.
theta=0.
uc=1.
wmax=1.5
thres=0.6
#parameters stimulation
dt=0.5
lagStim=100.
times=105.

amp=4.


delta=8.
period=20.

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
mystim.inten=amp

#integrato
npts=int(np.floor(delay/dt)+1)         # points delay
tmax=times*(lagStim+n*(period+delta))+40
#initial conditions
x0=0.01*np.ones((npts,n))
W0=[(0.1)*np.zeros((n,n)) for i in range(npts)]
theintegrator=myintegrator(delay,dt,n,tmax)
theintegrator.fast=False



rc={'axes.labelsize': 40, 'font.size': 35, 'legend.fontsize': 25, 'axes.titlesize': 30}
plt.rcParams.update(**rc)





#----------------------------------------------------------
# Transfer function, stationary leanring function and tau
#-----------------------------------------------------------
figure=plt.figure(figsize=(28,7))
figure.subplots_adjust(wspace=.4) # vertical space bw figures
#figure.subplots_adjust(hspace=.3) # vertical space bw figures
lw=7
learningrule1=figure.add_subplot(131)
current=np.linspace(-2.5,7,400)
tf,=learningrule1.plot(current,phi(current,theta,uc),'k',lw=lw,label=r'$\phi(u)$')
#learnmax=learningrule1.plot(current,0.5*(1+np.tanh(-50.*(current-thres))),'m',lw=3,label=r'$\tau_{Pre}(u)=\tau_{Post}(u)$')
learningrule1.axhline(y=thres, xmin=-2., xmax = 2., linewidth=lw,color='darkgrey',ls='dashed')
#learningrule1.legend(tf,r'$\phi(u)$',r'$f(u)=g(u)$'), loc = (0.01, 0.8) )
learningrule1.set_ylim([0,1.2])
learningrule1.set_xlim([-1.2,1.2])
learningrule1.set_yticks([0.5,1])
learningrule1.set_xticks([-1,0,1])
learningrule1.set_xlabel(r'Current ($u$)',fontsize=45)
learningrule1.set_ylabel(r'Rate ($\phi(u)$)',fontsize=45)
learningrule1.set_title('(A)',fontsize=45,y=1.06)
#figure.savefig('transferfunction.pdf', bbox_inches='tight')

#figure=plt.figure()
learningrule2=figure.add_subplot(132)
Fr=np.linspace(0,1,400)
learnmax,=learningrule2.plot(Fr,0.5*(1+np.tanh(a_post*(Fr-b_post))),'k',lw=lw,label=r'$f(u)=g(u)$')
#learnmax=learningrule1.plot(current,0.5*(1+np.tanh(-50.*(current-thres))),'m',lw=3,label=r'$\tau_{Pre}(u)=\tau_{Post}(u)$')
learningrule2.axvline(x=thres, ymin=-1., ymax = 2., linewidth=lw,color='darkgrey',ls='dashed')
#learningrule2.legend( (tf,learnmax),(r'$\phi(u)$',r'$f(u)=g(u)$'), loc = (0.01, 0.8) )
learningrule2.set_ylim([0,1.2])
learningrule2.set_xlim([0,1.])
learningrule2.set_yticks([0.5,1.])
learningrule2.set_xticks([0,0.5,1])
learningrule2.set_xlabel(r'Rate ($r$)',fontsize=45)
learningrule2.set_ylabel(r'$f(r)$',fontsize=45)
learningrule2.set_title('(B)',fontsize=45,y=1.06)
#figure.savefig('LR.pdf', bbox_inches='tight')
#plt.close(figure)

#figure=plt.figure()
#learningrule1.legend(loc='upper left')
#curve f(x)g(y)=0.5 wmax
current_plot=np.linspace(.68,1,200)
def fun(x): 
	return (1/2.)* (1. + np.tanh(a_post*(x-b_post)))
x0 = 0.8
f_g_eq_0_5 = []
for x in current_plot:
	y = lambda r:fun(r)-.5/fun(x)
	sol = root(y, x0)
	f_g_eq_0_5.append(sol.x[0])


learningrule3=figure.add_subplot(133)
current1=np.linspace(0,1,200)
current2=np.linspace(0,1,200)
myplot=learningrule3.contourf(current1,current2,winf([current1,current2])*(1./wmax),10,alpha=0.5,cmap=plt.cm.autumn,origin='lower')
learningrule3.plot(current_plot, f_g_eq_0_5,color = 'g',lw = lw, ls = '--')
learningrule3.axvline(x=thres,ymin = 0, ymax=(thres-0.3)/0.7, linewidth=lw,color='darkgrey',ls='dashed')
learningrule3.axhline(y=thres,xmin=0, xmax=(thres-0.3)/(0.7), linewidth=lw,color='darkgrey',ls='dashed')
learningrule3.set_xlabel(r'Rate pre ($r_{j}$)')
learningrule3.set_ylabel(r'Rate post ($r_{i}$)')
learningrule3.set_ylim([0.3,1.])
learningrule3.set_xlim([0.3,1.])
learningrule3.set_xticks([0.4,0.7,1])
learningrule3.set_yticks([0.4,0.7,1])
learningrule3.text(.39, .48, r'No', fontsize=35)
learningrule3.text(.298, .41, r'Plasticity', fontsize=35)
learningrule3.text(0.65,.41, r'Plasticity', fontsize=35)
cbar=plt.colorbar(myplot,ticks=[0,0.5,1.])
cbar.ax.set_ylabel(r'$w_{max}$',fontsize=45)
learningrule3.set_title('(C)',fontsize=45,y=1.06)
figure.savefig('LRheat.pdf', bbox_inches='tight')
#plt.show()
plt.close(figure)
plt.close()
print 'tranferfunction.pdf is stored'
#---------------------------------------------------------------
# Qualitative  T vs  Delta diagaram
#---------------------------------------------------------------

rc={'axes.labelsize': 40, 'font.size': 30, 'legend.fontsize': 25, 'axes.titlesize': 30}
plt.rcParams.update(**rc)

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
plt.xticks([delay,2*delay],[r'$D$',r'$2D$'])
plt.yticks([delay,2*delay],[r'$D$',r'$2D$'])
plt.xlabel(r'$\Delta$ (ms)',fontsize=32)
plt.ylabel(r'$T$ (ms)',fontsize=32)
plt.savefig('qualitativediagram.pdf', bbox_inches='tight')
plt.close()
#plt.show()

print 'qualitativediagram.pdf is stored'







def step_function(x,offset,start,period):
	fun  = np.zeros(len(x)) + offset
	fun[(start<=x) * (x<=(start + period))] = 0.8 + offset 
	return fun



#----------------------------------------------------
#------- Examples Qualitative Diagram ---------------
#----------------------------------------------------

rc={'axes.labelsize': 40, 'font.size': 35, 'legend.fontsize': 25, 'axes.titlesize': 30}
plt.rcParams.update(**rc)



fig=plt.figure(figsize=(26,12))
gs = gridspec.GridSpec(2, 3)
gs.update(wspace=0.1,hspace=0.1)
dynamics1 = plt.subplot(gs[0,1])
dynamics2= plt.subplot(gs[0,2])
dynamics4 = plt.subplot(gs[1,1])
dynamics3= plt.subplot(gs[1,2])

gs1 = gridspec.GridSpec(1, 1)
gs1.update(wspace=0.05,hspace=0.05,left=0.,right=.3,top=0.88,bottom=0.12)
ax1 = plt.subplot(gs1[0,0])

mydel = 20
x = np.linspace(0,140,1000)
I1 = step_function(x,10,10,20)
I2 = step_function(x,8.7,10  + 20 + mydel,20)
off1 = 10 * np.ones(1000)
off2 = 8.7 * np.ones(1000)

ax1.plot(x,I1,color = 'k',lw = 3)
ax1.plot(x,I2,color = 'k',lw = 3)
ax1.fill_between(x,off1,I1,where  = off1<I1,alpha=0.2,edgecolor='k', facecolor='darkgrey')
ax1.fill_between(x,off2,I2,where  = off2<I2,alpha=0.2,edgecolor='k', facecolor='darkgrey')
#ax1.plot(x,step_function(x,7.2,10  + 40 + 2 * mydel,20),color = 'k',lw= 3)
ax1.axis('off')
ymin = 8.6
ymax = 11.1
ax1.axvline(x = 10+20,ymin =(9.5-ymin)/(ymax - ymin),ymax = (10 - ymin)/(ymax - ymin), ls = '--',color = 'gray')
ax1.axvline(x = 10+20 + mydel,ymin =(9.5-ymin)/(ymax - ymin),ymax = (10 - ymin)/(ymax - ymin), ls = '--',color = 'gray')
#ax1.axvline(x = 10 + 40 + mydel,ymin =(8.1-ymin)/(ymax - ymin),ymax = (8.5 - ymin)/(ymax - ymin), ls = '--',color = 'gray')
#ax1.axvline(x = 10+40 + 2 * mydel,ymin =(8.1-ymin)/(ymax - ymin),ymax = (8.5 - ymin)/(ymax - ymin), ls = '--',color = 'gray')
ax1.set_ylim([ymin,ymax])
ax1.annotate("", xy=(10, 10.2), xytext=(30, 10.2),arrowprops=dict(color= 'red',arrowstyle="<->"),color = 'r')
ax1.annotate("", xy=(30, 9.6), xytext=(30 + mydel, 9.6),arrowprops=dict(color='red',arrowstyle="<->"))
#font = {'family': 'serif','color':  'darkred','weight': 'normal','size': 50,
					        #}
ax1.text(10+5, 10.3, r'$T$', fontsize = 50)
ax1.text(10+20 + 5, 9.7, r'$\Delta$', fontsize = 50)
ax1.text(86, 10.38, r'Stimulus', fontsize = 30)
ax1.text(78, 10.2, r'1$^{st}$ population', fontsize = 30)
ax1.text(86, 9.08, r'Stimulus', fontsize = 30)
ax1.text(78, 8.9, r'2$^{nd}$ population', fontsize = 30)
#ax1.text(20, 7.4, r'Stim. Pop. 3', fontsize = 30)
#ax1.axhline(y = 10 + 40 + mydel,ymin =(8.1-ymin)/(ymax - ymin),ymax = (8.5 - ymin)/(ymax - ymin), ls = '--',color = 'gray')
ax1.set_title('(A)',y=1.03,fontsize = 60)

amp=1.6#1.25
timesmax=30
# region 1
delta=8.
period=18.
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
		dynamics1.plot(t,connectivity[:,i,i],'c',lw=3)
for i in range(0,9):
		dynamics1.plot(t,connectivity[:,i+1,i],'y',lw=3)
for i in range(8):
		dynamics1.plot(t,connectivity[:,i+2,i],'g',lw=3)
for i in range(9):
		dynamics1.plot(t,connectivity[:,i,i+1],'r',lw=3)
for i in range(8):
		dynamics1.plot(t,connectivity[:,i,i+2],'b',lw=3)
dynamics1.text(1000, 0.5, r'$\Delta<D< T$', fontsize = 40)
dynamics1.set_xlim([0,10000])
dynamics1.set_ylim([0,.6])
dynamics1.set_xticks([])#[0,5000,10000],['0','5','10'])
dynamics1.set_yticks([.3,0.6])
#plt.xlabel('Time (s)')
dynamics1.set_ylabel('Synaptic Weights')
dynamics1.set_title('(B)',x =1.05,y = 1.06,fontsize = 60)
#plt.savefig('dynamicsweights1.pdf', bbox_inches='tight',transparent=True)
#plt.show()
#plt.close()

# region 2
amp=1.8
delta=80.
period=20.5
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
		dynamics2.plot(t,connectivity[:,i,i],'c',lw=3)
for i in range(0,9):
		dynamics2.plot(t,connectivity[:,i+1,i],'y',lw=3)
for i in range(8):
		dynamics2.plot(t,connectivity[:,i+2,i],'g',lw=3)
for i in range(9):
		dynamics2.plot(t,connectivity[:,i,i+1],'r',lw=3)
for i in range(8):
		dynamics2.plot(t,connectivity[:,i,i+2],'b',lw=3)
dynamics2.text(1000, 0.5, r'$D<\Delta, T$', fontsize = 40)
dynamics2.set_xlim([0,10000])
#plt.xticks([0,5000,10000])
dynamics2.set_xticks([])#0,5000,10000],['0','5','10','15'])
#plt.xticks([0,40000,80000,120000],['0','40','80','120'])
dynamics2.set_yticks([])#0,.3,.6])
dynamics2.set_ylim([0,.6])
#plt.xlabel('Time (s)')
#plt.ylabel('Synaptic Weights')
#plt.savefig('dynamicsweights2.pdf', bbox_inches='tight',transparent=True)
#plt.show()
#plt.close()


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
		dynamics3.plot(t,connectivity[:,i,i],'c',lw=3)
for i in range(0,9):
		dynamics3.plot(t,connectivity[:,i+1,i],'y',lw=3)
for i in range(8):
		dynamics3.plot(t,connectivity[:,i+2,i],'g',lw=3)
for i in range(9):
		dynamics3.plot(t,connectivity[:,i,i+1],'r',lw=3)
for i in range(8):
		dynamics3.plot(t,connectivity[:,i,i+2],'b',lw=3)
dynamics3.text(1000, 0.5, r'$ T<D<\Delta$', fontsize = 40)
dynamics3.set_xlim([0,10000])
dynamics3.set_ylim([0,0.6])
#dynamics3.xticks([0,4000,8000])
dynamics3.set_xticks([0,5000,10000])
dynamics3.set_xticklabels(['0','5','10'])
dynamics3.set_yticks([])
dynamics3.set_xlabel('Time (s)')
#plt.ylabel('Synaptic Weights')
#plt.savefig('dynamicsweights3.pdf', bbox_inches='tight',transparent=True)
#plt.show()
#plt.close()
#print 'dynamicsweights3.pdf is stored'

# region 4
#amp=3.
delta=9.
period=7.
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
		dynamics4.plot(t,connectivity[:,i,i],'c',lw=3)
for i in range(0,9):
		dynamics4.plot(t,connectivity[:,i+1,i],'y',lw=3)
for i in range(8):
		dynamics4.plot(t,connectivity[:,i+2,i],'g',lw=3)
for i in range(9):
		dynamics4.plot(t,connectivity[:,i,i+1],'r',lw=3)
for i in range(8):
		dynamics4.plot(t,connectivity[:,i,i+2],'b',lw=3)
dynamics4.text(1000, 0.5, r'$\Delta, T<D$', fontsize = 40)
dynamics4.set_xlim([0,10000])
#dynamics4.xticks([0,2600,5200])
dynamics4.set_xticks([0,5000,10000])
dynamics4.set_xticklabels(['0','5','10'])
dynamics4.set_yticks([.3,.6])
dynamics4.set_ylim([0,.6])
dynamics4.set_xlabel('Time (s)')
dynamics4.set_ylabel('Synaptic Weights')
plt.savefig('dynamicsweights.pdf', bbox_inches='tight',transparent=True)
#plt.savefig('dynamicsweights.pdf', bbox_inches='tight')
#plt.show()
#plt.close()


print 'dynamicsweights4.pdf is stored'



