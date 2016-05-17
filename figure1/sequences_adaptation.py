import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
from myintegrator import *

# this is the transfer function 
def phi(x,theta,uc):
	myphi=nu*(x-theta)
	myphi[x>uc]=nu*(uc-theta)
	myphi[theta>x]=0.
	return myphi

def phi_brunel(x,theta,uc):
	myphi=nu_brunel*((x-theta)/uc)**2
	myphi[x>(uc+theta)]=2*nu_brunel*np.sqrt((x[x>(uc+theta)]-theta)/uc-3./4.)
	myphi[theta>x]=0.
	return myphi

def phi_tanh(x):
	return 0.5*(1+np.tanh(a1*(x+b1)))

# this is the connectivity matrix of the network
def net(wmax,sdel,n,k):
	mysdel=np.concatenate([sdel[i]*np.ones(n/k) for i in range(k)])
	mysdel=mysdel[0:-1]
	mywmax=np.concatenate([wmax[i]*np.ones(n/k) for i in range(k)])
	diagonals=[mywmax,mysdel]
	return sparse.diags(diagonals,[0,-1])



def net_matrix(wmax_min,wmax_max,sdel_min,sdel_max,n,k):
	sdel=np.linspace(sdel_min,sdel_max,k)
	wmax=np.linspace(wmax_min,wmax_max,k)
	mysdel=np.concatenate([sdel[i]*np.ones(n/k) for i in range(k)])
	mysdel=mysdel[0:-1]
	mywmax=np.concatenate([wmax[i]*np.ones(n/k) for i in range(k)])
	diagonals=[mywmax,mysdel]
	vuelta=np.zeros((n,n))
	vuelta[0,-1]=0.1
	return np.diag(diagonals[0],0)+np.diag(diagonals[1],-1)-w_inh*nu*np.ones((n,n))#+vuelta

#fields approximations
def field_tanh(x,a,I,t):
	n=len(x)
	noise=np.zeros(n)
	noise[0]=0#np.random.normal(0,.4)
	thefield_e=(1./tau)*(net(wmax,sdel,n,k).dot(phi_tanh(x))-x-w_inh*I*np.ones(n)-a+noise)
	thefield_a=(1./tau_a)*(-a+beta*x)
	thefield_I=0.#(2./tau)*(amp*np.sin(2*np.pi*w*t)+Ic-I+np.ones(n).dot(phi_tanh(x)))
	return thefield_e,thefield_a,thefield_I

#fields approximations
def field_pw(x,a,I,t):
	n=len(x)
	noise=np.zeros(n)
	noise[0]=0#np.random.normal(0,.4)
	thefield_e=(1./tau)*(net(wmax,sdel,n,k).dot(phi(x,theta,uc))-x-w_inh*I*np.ones(n)-a+noise)
	thefield_a=(1./tau_a)*(-a+beta*x)
	thefield_I=0#(2./tau)*(amp*np.sin(2*np.pi*w*t)+Ic-I+np.ones(n).dot(phi(x,theta,uc)))
	return thefield_e,thefield_a,thefield_I

#fields approximations
def field_brunel(x,a,I,t):
	n=len(x)
	noise=np.zeros(n)
	noise[0]=0#np.random.normal(0,.4)
	thefield_e=(1./tau)*(net(wmax,sdel,n,k).dot(phi_brunel(x,theta_brunel,uc_brunel))-x-w_inh*I*np.ones(n)-a+noise)
	thefield_a=(1./tau_a)*(-a+beta*x)
	thefield_I=0.#(2./tau)*(amp*np.sin(2*np.pi*w*t)+Ic-I+np.ones(n).dot(phi_brunel(x,theta_brunel,uc_brunel)))
	return thefield_e,thefield_a,thefield_I

n=20
k=1
w_i=0.
w_inh=w_i/n
tau=10.
nu=2.
nu_brunel=0.4*nu
theta=-0.0
theta_brunel=-0.1
uc=1/nu
uc_brunel=uc
mys=0.45
sdelmax=mys
sdelmin=mys
wmaxmin=0.25
wmaxmax=0.25

amp=0
w=0.05
Ic=0.

beta=0.8
tau_a=80.

#print(1./(nu+sdel))
a1=6.
b1=-0.25
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)
r1_matrix=np.ones((n,n))#np.outer(np.ones(n),np.random.normal(1,sigma_wi/n,n))

u0=np.zeros(n)
u0[0]=1.
a0=np.zeros(n)
I0=0.
tsim=500
#approx
utanh,atanh,Itanh,timetanh=rk4(field_tanh,u0,a0,I0,0.1,tsim)
upw,apw,Ipw,timepw=rk4(field_pw,u0,a0,I0,0.1,tsim)
ubrunel,abrunel,Ibrunel,timebrunel=rk4(field_brunel,u0,a0,I0,0.1,tsim)

rc={'axes.labelsize': 30, 'font.size': 20, 'legend.fontsize': 23.0, 'axes.titlesize': 30}
plt.rcParams.update(**rc)

#connectivity matrix
W01=net_matrix(wmaxmin,wmaxmax,sdelmin,sdelmax,n,k)
plt.matshow(W01)
plt.xlabel('Connectivity Matrix')
cax = plt.axes([0.95, 0.1, 0.03, 0.8])
plt.colorbar(cax=cax)
plt.savefig('connectivityAdaptation.pdf', bbox_inches='tight')
plt.show()




#dynamics

figure=plt.figure()
colormap = plt.cm.Accent
tanh=figure.add_subplot(311)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
tanh.plot(timetanh,phi_tanh(utanh[:,0:n]),lw=2)
tanh.set_xlim([0,450.])
tanh.set_ylim([0,3.])
tanh.set_xticks([0,100,200,300,400])
tanh.set_yticks([0,1.,2,3.])
tanh.set_xlabel('Time (ms)')
tanh.set_ylabel('Rate')
brunel=figure.add_subplot(312)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
brunel.plot(timebrunel,phi_brunel(ubrunel[:,0:n],theta_brunel,uc_brunel),lw=2)
brunel.set_xlim([0,450.])
brunel.set_ylim([0,3.])
brunel.set_xticks([0,100,200,300,400])
brunel.set_yticks([0,1.,2,3.])
brunel.set_xlabel('Time (ms)')
brunel.set_ylabel('Rate')
pwlinear=figure.add_subplot(313)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
pwlinear.plot(timepw,phi(upw[:,0:n],theta,uc),lw=2)
pwlinear.set_xlim([0,450.])
pwlinear.set_ylim([0,3.])
pwlinear.set_xticks([0,100,200,300,400])
pwlinear.set_yticks([0,1.,2,3.])
pwlinear.set_xlabel('Time (ms)')
pwlinear.set_ylabel('Rate')
plt.savefig('sequencesAdaptation.pdf', bbox_inches='tight')
plt.show()

