import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
from tempfile import TemporaryFile

''' This is a code fo find numerically the boundery between dSA and SA'''


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

def netapprox(wmax,sdel,n,k):
	mysdel=np.concatenate([(sdel[i]-w_i/n)*np.ones(n/k) for i in range(k)])
	mysdel=mysdel[0:-1]
	mywmax=np.concatenate([(wmax[i]-w_i/n)*np.ones(n/k) for i in range(k)])
	myinh1=-(w_i/n)*np.ones(n-2)
	myinh2=-(w_i/n)*np.ones(n-1)
	myinh3=-(w_i/n)*np.ones(n-3)
	diagonals=[mywmax,mysdel,myinh2,myinh1,myinh1]
	return sparse.diags(diagonals,[0,-1,1,-2,2])



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
	thefield[0:n-1]=net(wmax,sdel,n-1,k).dot(phi_tanh(x[0:n-1]))-x[0:n-1]-w_inh*x[-1]*np.ones(n-1)
	thefield[-1]=2.*(-x[-1]+np.ones(n-1).dot(phi_tanh(x[0:n-1])))
	return thefield

def field_true_pw(x,t):
	n=len(x)
	thefield=np.zeros(n)
	thefield[0:n-1]=net(wmax,sdel,n-1,k).dot(phi(x[0:n-1],theta,uc))-x[0:n-1]-w_inh*x[-1]*np.ones(n-1)
	thefield[-1]=2.*(-x[-1]+np.ones(n-1).dot(phi(x[0:n-1],theta,uc)))
	return thefield

def field_true_pw_approx(x,t):
	n=len(x)
	thefield=np.zeros(n)
	thefield[0:n-1]=netapprox(wmax,sdel,n-1,k).dot(phi(x[0:n-1],theta,uc))-x[0:n-1]
	thefield[-1]=2.*(-x[-1]+np.ones(n-1).dot(phi(x[0:n-1],theta,uc)))
	return thefield

def field_true_brunel(x,t):
	n=len(x)
	thefield=np.zeros(n)
	thefield[0:n-1]=net(wmax,sdel,n-1,k).dot(phi_brunel(x[0:n-1],theta_brunel,uc_brunel))-x[0:n-1]-w_inh*x[-1]*np.ones(n-1)
	thefield[-1]=2.*(-x[-1]+np.ones(n-1).dot(phi_brunel(x[0:n-1],theta_brunel,uc_brunel)))
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
		#print(t)
	return np.array(mysol),mytime


n=10
k=1
w_i=2.#1.0
w_inh=w_i/n
nu=1.
nu_brunel=0.4*nu
theta=-0.0
theta_brunel=-0.1
uc=1./nu
uc_brunel=uc
w=0.5
s=0.9
sdelmax=s
sdelmin=s
wmaxmin=w
wmaxmax=w
a1=6.
b1=-0.25
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)
r1_matrix=np.ones((n,n))#np.outer(np.ones(n),np.random.normal(1,sigma_wi/n,n))

y0=theta*np.ones(n)
y0[0]=1.
y0_true=1.2*np.zeros(n+1)
y0_true[0]=1.2
#y0_true[1]=1.2
#y0_true[2]=1.2
#y0_true[3]=1.2

tsim=100.
#approx
#ypw,timepw=rk4(field_pw,y0,0.1,tsim)
##true
##ypw_true_approx,timepw_true_approx=rk4(field_true_pw_approx,y0_true,0.1,tsim)
#
numpoints=200
bifdiagram=[]
gridw=np.linspace(0.01,2.,numpoints)
grids=np.linspace(0.01,2.,numpoints)
bifcurve=[]
for wgrid in gridw:
	wmax=np.linspace(wgrid,wgrid,k)
	bifdiagram_w=[0]	
	print(wgrid)
	i=0
	for sgrid in grids:
		sdel=np.linspace(sgrid,sgrid,k)
		ypw_true,timepw_true=rk4(field_true_pw,y0_true,0.2,tsim)
		if max(ypw_true[:,1])<=max(ypw_true[:,9]) and ypw_true[-1,9]<0.98:
			bifdiagram_w.append(1)
		
			if bifdiagram_w[i]==0 and bifdiagram_w[i+1]==1:
				bifcurve.append([sgrid,wgrid])
		else:
			bifdiagram_w.append(0)
		i=i+1
	bifdiagram.append(bifdiagram_w[1:numpoints])

print(bifcurve)
bifcurve=np.array(bifcurve)
print bifcurve[:,0]
np.save('mybifcurve2.npy', bifcurve)
bifcurve=np.load('mybifcurve2.npy')
plt.matshow(np.array(bifdiagram))
plt.show()
#

s=0.6
w=1.01
sdelmin=s
sdelmax=s
wmaxmax=w
wmaxmin=w
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)
ypw_true,timepw_true=rk4(field_true_pw,y0_true,0.1,tsim)
#figure
figure=plt.figure()


colormap = plt.cm.Accent

#dynamics
dynamics=figure.add_subplot(121)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
dynamics.plot(timepw_true,phi(ypw_true[:,0:n],theta,uc),lw=2)
#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
dynamics.set_xlim([0,tsim])
dynamics.set_ylim([0,1.2])
dynamics.set_xlabel('Time')
dynamics.set_ylabel('Rate')


#dynamics
dynamics2=figure.add_subplot(122)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
dynamics2.plot(timepw_true,phi(ypw_true[:,0:n],theta,uc),lw=2.)
dynamics2.set_xlim([0,tsim])
dynamics2.set_ylim([0,1.2])
dynamics2.set_xlabel('Time')
dynamics2.set_ylabel('Rate')

plt.show()


mys=np.linspace(0,1.5,100)
myw=np.linspace(0,1.5,100)


plt.plot(mys,mys+1,c='k',lw=2)
plt.plot(mys,np.array([1+(w_i+0.)/n for i in range(0,100)]),c='k',lw=2)
plt.plot(bifcurve[:,0],bifcurve[:,1],'bo')
for i in range(1,n+1):
	myline1=np.linspace(0,w_i*(i+0.)/n,100)
	myconstant1=np.array([1+w_i*(i+0.)/n for j in range(0,100)])
	plt.plot(myline1,myconstant1,c='k',lw=1)
	plt.fill_between(myline1,myconstant1,myconstant1+w_i/n,alpha=0.5,edgecolor='#CC4F1B', facecolor='#FF9848')
plt.xlim([0,1.0])
plt.ylim([0,2.5])
plt.show()




