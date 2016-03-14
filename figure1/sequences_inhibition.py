import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
''' This is a simulation of a population net with shared inhibiton and three different transfer functions'''


# this is the ii transfer function 
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

def field_tanh(x,t):
	return net(wmax,sdel,n,k).dot(phi_tanh(x))-x-w_inh*np.dot(r1_matrix,phi_tanh(x))
def field_pw(x,t):
	return net(wmax,sdel,n,k).dot(phi(x,theta,uc))-x-w_inh*np.dot(r1_matrix,phi(x,theta,uc))

def field_brunel(x,t):
	return net(wmax,sdel,n,k).dot(phi_brunel(x,theta,uc))-x-w_inh*np.dot(r1_matrix,phi_brunel(x,theta,uc))
#field true 
def field_true_tanh(x,t):
	n=len(x)
	thefield=np.zeros(n)
	thefield[0:n-1]=(1/tau)*(net(wmax,sdel,n-1,k).dot(phi_tanh(x[0:n-1]))-x[0:n-1]-w_inh*x[-1]*np.ones(n-1))
	thefield[-1]=(2./tau)*(-x[-1]+np.ones(n-1).dot(phi_tanh(x[0:n-1])))
	return thefield

def field_true_pw(x,t):
	n=len(x)
	thefield=np.zeros(n)
	thefield[0:n-1]=(1/tau)*(net(wmax,sdel,n-1,k).dot(phi(x[0:n-1],theta,uc))-x[0:n-1]-w_inh*x[-1]*np.ones(n-1))
	thefield[-1]=(2./tau)*(-x[-1]+np.ones(n-1).dot(phi(x[0:n-1],theta,uc)))
	return thefield

def field_true_brunel(x,t):
	n=len(x)
	thefield=np.zeros(n)
	thefield[0:n-1]=(1/tau)*(net(wmax,sdel,n-1,k).dot(phi_brunel(x[0:n-1],theta_brunel,uc_brunel))-x[0:n-1]-w_inh*x[-1]*np.ones(n-1))
	thefield[-1]=(2./tau)*(-x[-1]+np.ones(n-1).dot(phi_brunel(x[0:n-1],theta_brunel,uc_brunel)))
	return thefield



def rk4(f,y0,dt,T):
	mysol=[]
	mytime=[]
	t=0
	un=y0
	mytime.append(t)
	mysol.append(un)
	while t<=T:
		k1=f(un,t)
		k2=f(un+(dt/2)*k1,t+dt/2)
		k3=f(un+(dt/2)*k2,t+dt/2)
		k4=f(un+dt*k3,t+dt)
		un=un+(dt/6)*(k1+2*k2+2*k3+k4)
		t=t+dt
		mysol.append(un)
		mytime.append(t)
		print(t)
	return np.array(mysol),mytime


n=20
k=1
w_i=1.6
w_inh=w_i/n
nu=2.
tau=10.
nu_brunel=0.4*nu
theta=-0.0
theta_brunel=-0.1
uc=1/nu
uc_brunel=uc
sdelmax=0.6
sdelmin=0.6
wmaxmin=0.35
wmaxmax=0.35
#print(1./(nu+sdel))
a1=6.
b1=-0.25
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)
r1_matrix=np.ones((n,n))#np.outer(np.ones(n),np.random.normal(1,sigma_wi/n,n))

y0=theta*np.ones(n)
y0[0]=1.
y0_true=np.zeros(n+1)
y0_true[0]=1.



rc={'axes.labelsize': 30, 'font.size': 20, 'legend.fontsize': 28.0, 'axes.titlesize': 30}
plt.rcParams.update(**rc)
#approx
#true
ytanh_true,timetanh_true=rk4(field_true_pw,y0_true,0.1,500)
ypw_true,timepw_true=rk4(field_true_pw,y0_true,0.1,500)
ybrunel_true,timebrunel_true=rk4(field_true_brunel,y0_true,0.1,500)

#figure

#connectivity matrix
W01=net_matrix(wmaxmin,wmaxmax,sdelmin,sdelmax,n,k)
plt.matshow(W01,vmin=-0.2,vmax=0.6)
plt.xlabel('Effective Connectivity (II)')
cax = plt.axes([0.95, 0.1, 0.03, 0.8])
plt.colorbar(cax=cax,ticks=[-0.2,0,0.2,0.4,0.6])
plt.savefig('connectivityInhib.pdf', bbox_inches='tight')
plt.show()


W01=net_matrix(wmaxmin,wmaxmax,sdelmin,sdelmax,n,k)
w_i=0.
W02=net_matrix(wmaxmin,wmaxmax,sdelmin,sdelmax,n,k)
data=[W01,W02]
fig, axes = plt.subplots(nrows=2, ncols=1)
for dat, ax in zip(data, axes.flat):
	    # The vmin and vmax arguments specify the color limit
	im = ax.matshow(dat, vmin=0, vmax=.8)
	# Make an axis for the colorbar on the right side
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax,ticks=[-0.2,0,0.2,0.4,0.6])
plt.savefig('matrixsequences.pdf', bbox_inches='tight')
plt.show()
#transfer function

myu=np.linspace(-.5,1.5,200)
thepw,=plt.plot(myu,phi(myu,theta,uc),'b',lw=2.5,label='line 1')
thetanh,=plt.plot(myu,phi_tanh(myu),'g',lw=2.5,label='line 2')
thebrunel,=plt.plot(myu,phi_brunel(myu,theta_brunel,uc_brunel),'r',lw=2.5,label='line 3')
plt.xlim([-.2,0.8])
plt.ylim([0,2.0])
plt.xlabel('Current')
plt.ylabel('Transfer Function')
plt.legend([thepw,thetanh,thebrunel],['PL','S','PNL'],loc='upper left')
plt.savefig('transferFunctions.pdf', bbox_inches='tight')
plt.show()


#dynamics

figure=plt.figure()
colormap = plt.cm.Accent
tanh=figure.add_subplot(311)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
tanh.plot(timetanh_true,phi_tanh(ytanh_true[:,0:n]),lw=2)
tanh.set_xlim([0,450.])
tanh.set_ylim([0,3.])
tanh.set_xticks([0,100,200,300,400])
tanh.set_yticks([0,1.,2,3.])
tanh.set_xlabel('Time (ms)')
tanh.set_ylabel('Rate')
brunel=figure.add_subplot(312)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
brunel.plot(timebrunel_true,phi_brunel(ybrunel_true[:,0:n],theta_brunel,uc_brunel),lw=2)
brunel.set_xlim([0,450.])
brunel.set_ylim([0,3.])
brunel.set_xticks([0,100,200,300,400])
brunel.set_yticks([0,1.,2,3.])
brunel.set_xlabel('Time (ms)')
brunel.set_ylabel('Rate')
pwlinear=figure.add_subplot(313)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
pwlinear.plot(timepw_true,phi(ypw_true[:,0:n],theta,uc),lw=2)
pwlinear.set_xlim([0,450.])
pwlinear.set_ylim([0,3.])
pwlinear.set_xticks([0,100,200,300,400])
pwlinear.set_yticks([0,1.,2,3.])
pwlinear.set_xlabel('Time (ms)')
pwlinear.set_ylabel('Rate')
plt.savefig('sequencesInhibition.pdf', bbox_inches='tight')
plt.show()


