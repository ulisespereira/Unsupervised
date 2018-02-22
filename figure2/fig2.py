import numpy as np
from scipy import sparse
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
import math as mt
from tempfile import TemporaryFile

''' This is a simulation of a population net with shared inhibiton and three different transfer functions'''


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

#field true 

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
w_i=2.0
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

y0_true=1.*np.zeros(n+1)
y0_true[0]=1.
#y0_true[1]=1.2
#y0_true[2]=1.2
#y0_true[3]=1.2


bifcurve=np.load('mybifcurve.npy')

#figure

rc={'axes.labelsize': 30, 'font.size': 20, 'legend.fontsize': 28.0, 'axes.titlesize': 30}
plt.rcParams.update(**rc)


#-----------------------------------------
#Full Bifurcation Diagram
#-----------------------------------------

mys=np.linspace(0,2.,100)
myw=np.linspace(0,2.,100)
upperBsequences=np.array([1+w_i*(1+0.)/n for j in range(0,len(bifcurve[:,1]))])

plt.plot(bifcurve[:,0],bifcurve[:,1],'k')
plt.plot(mys,np.array([1+(w_i+0.)/n for i in range(0,100)]),c='k',lw=1)
plt.fill_between(bifcurve[:,0],bifcurve[:,1],upperBsequences,alpha=0.5,edgecolor='k', facecolor='red')
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

smax=2. # the upper bound for s
for i in range(1,n):
	myline1=np.linspace(w_i*(i+0.)/n,smax,100)
	myconstant1=np.array([1+w_i*(i+0.)/n for l in range(0,100)])
	plt.fill_between(myline1,myconstant1,myconstant1+w_i/n,alpha=0.1*i,edgecolor='grey', facecolor='green')

myconstant1=np.array([2. for j in range(0,100)])
myconstant2=np.array([2.1 for j in range(0,100)])

alph=0.15
for j in range(0,n):
	myline2=np.linspace(w_i*(j+0.)/n,w_i*(j+1.)/n,100)
	plt.fill_between(myline2,myconstant1,myconstant2,alpha=1.,edgecolor='grey', facecolor=colormap((j+0.)/n)[0:3])


plt.xlim([0.,2.])
plt.ylim([0,2])
plt.yticks([0.5,1,1.5,2.])
plt.xticks([0.,0.5,1.,1.5,2.])
plt.xlabel(r'Feed-forward ($s$)',fontsize='26')
plt.ylabel(r'Recurrent ($w$)',fontsize='26')
plt.savefig('bifurcationdiagram.pdf', bbox_inches='tight')

print 'bifurcationdiagram.pdf stored'


#---------------------------------------
# Dynamics DNS 
#---------------------------------------

tsim=15.
s=1.5
w=0.2
sdelmin=s
sdelmax=s
wmaxmax=w
wmaxmin=w
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)

y0_true=1.*np.zeros(n+1)
y0_true[0]=1.
ypw_true1,timepw_true1=rk4(field_true_pw,y0_true,0.1,tsim)
y0_true=1.*np.ones(n+1)
y0_true[-1]=0.
ypw_true2,timepw_true2=rk4(field_true_pw,y0_true,0.1,tsim)

figure=plt.figure()
colormap = plt.cm.Accent
#dynamics
dynamics=figure.add_subplot(111)
#dynamics=figure.add_subplot(211)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
dynamics.plot(10.*np.array(timepw_true1),phi(ypw_true1[:,0:n],theta,uc),lw=4)
#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
dynamics.set_yticks([0.5,1.])
dynamics.set_xticks([0,50,100,150])
dynamics.set_xlim([0,10*tsim])
dynamics.set_ylim([0,1.2])
dynamics.set_xlabel('Time (ms)',fontsize=40)
dynamics.set_ylabel('Rate',fontsize=40)
plt.tick_params(labelsize=35)
#dynamics
#dynamics2=figure.add_subplot(212)
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
#dynamics2.plot(10.*np.array(timepw_true2),phi(ypw_true2[:,0:n],theta,uc),lw=2.)
#dynamics2.set_yticks([0,0.4,0.8,1.2])
#dynamics2.set_xlim([0,10*tsim])
#dynamics2.set_ylim([0,1.2])
#dynamics2.set_xlabel('Time (ms)')
#dynamics2.set_ylabel('Rate')
plt.savefig('DNS.pdf', bbox_inches='tight')

print 'DNS.pdf stored'

#plt.show()

#------------------------------------------
# Dynamics NS
#-----------------------------------------

tsim=40.
s=1.39
w=1.01
sdelmin=s
sdelmax=s
wmaxmax=w
wmaxmin=w
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)

y0_true=1.*np.zeros(n+1)
y0_true[0]=1.
ypw_true1,timepw_true1=rk4(field_true_pw,y0_true,0.1,tsim)
y0_true=1.*np.ones(n+1)
y0_true[-1]=0.
ypw_true2,timepw_true2=rk4(field_true_pw,y0_true,0.1,tsim)

figure=plt.figure()
colormap = plt.cm.Accent
#dynamics
#dynamics=figure.add_subplot(211)
dynamics=figure.add_subplot(111)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
dynamics.plot(10.*np.array(timepw_true1),phi(ypw_true1[:,0:n],theta,uc),lw=4)
#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
dynamics.set_xticks([0.,200.,400])
dynamics.set_yticks([0.5,1.])
dynamics.set_yticks([])
dynamics.set_xlim([0,10*tsim])
dynamics.set_ylim([0,1.2])
dynamics.set_xlabel('Time (ms)',fontsize=40)
plt.tick_params(labelsize=35)
#dynamics.set_ylabel('Rate')


#dynamics
#dynamics2=figure.add_subplot(212)
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
#dynamics2.plot(10.*np.array(timepw_true2),phi(ypw_true2[:,0:n],theta,uc),lw=2.)
#dynamics2.set_xlim([0,10*tsim])
#dynamics2.set_ylim([0,1.2])
#dynamics2.set_yticks([0,0.4,0.8,1.2])
#dynamics2.set_xlabel('Time (ms)')
#dynamics2.set_ylabel('Rate')
plt.savefig('NS.pdf', bbox_inches='tight')
print 'NS.pdf stored'
#plt.show()


#------------------------------------------------------
# Dynamics NS ends in Fixed Point 1 Neuron Hight Rate 
#------------------------------------------------------

tsim=60.
s=1.
w=1.3
sdelmin=s
sdelmax=s
wmaxmax=w
wmaxmin=w
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)

y0_true=1.*np.zeros(n+1)
y0_true[0]=1.
ypw_true1,timepw_true1=rk4(field_true_pw,y0_true,0.1,tsim)
y0_true=1.*np.ones(n+1)
y0_true[-1]=0.
ypw_true2,timepw_true2=rk4(field_true_pw,y0_true,0.1,tsim)

figure=plt.figure()
colormap = plt.cm.Accent
#dynamics
#dynamics=figure.add_subplot(211)
dynamics=figure.add_subplot(111)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
dynamics.plot(10.*np.array(timepw_true1),phi(ypw_true1[:,0:n],theta,uc),lw=4)
#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
#dynamics.set_yticks([0.5,1.])
dynamics.set_yticks([])
dynamics.set_xticks([0.,300,600.])
dynamics.set_xlim([0,10*tsim])
dynamics.set_ylim([0,1.2])
dynamics.set_xlabel('Time (ms)',fontsize=40)
plt.tick_params(labelsize=35)
#dynamics.set_ylabel('Rate')


#dynamics
#dynamics2=figure.add_subplot(212)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
#dynamics2.plot(10.*np.array(timepw_true2),phi(ypw_true2[:,0:n],theta,uc),lw=2.)
#dynamics2.set_xlim([0,10*tsim])
#dynamics2.set_ylim([0,1.2])
#dynamics2.set_yticks([0,0.4,0.8,1.2])
#dynamics2.set_xlabel('Time (ms)')
#dynamics2.set_ylabel('Rate')
plt.savefig('NStoPA1.pdf', bbox_inches='tight')
print 'NStoPA1.pdf stored'
#plt.show()


#---------------------------------------------------------------------------
# Dynamics of  Fixed Point with at least and at most 2 Neurons in Hight Rate
#---------------------------------------------------------------------------


tsim=20.
s=0.15
w=1.3
sdelmin=s
sdelmax=s
wmaxmax=w
wmaxmin=w
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)

y0_true=1.*np.zeros(n+1)
y0_true[0]=1.
ypw_true1,timepw_true1=rk4(field_true_pw,y0_true,0.1,tsim)
y0_true=1.*np.ones(n+1)
y0_true[-1]=0.
ypw_true2,timepw_true2=rk4(field_true_pw,y0_true,0.1,tsim)

figure=plt.figure()
colormap = plt.cm.Accent
#dynamics
#dynamics=figure.add_subplot(211)
dynamics=figure.add_subplot(111)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
dynamics.plot(10.*np.array(timepw_true1),phi(ypw_true1[:,0:n],theta,uc),lw=4)
#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
dynamics.set_yticks([0.5,1])
dynamics.set_xticks([0.,100,200])
dynamics.set_xlim([0,10*tsim])
dynamics.set_ylim([0,1.2])
plt.tick_params(labelsize=35)
#dynamics.set_xlabel('Time (ms)')
dynamics.set_ylabel('Rate',fontsize=40)


#dynamics
#dynamics2=figure.add_subplot(212)
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
#dynamics2.plot(10.*np.array(timepw_true2),phi(ypw_true2[:,0:n],theta,uc),lw=2.)
#dynamics2.set_xlim([0,10*tsim])
#dynamics2.set_ylim([0,1.2])
#dynamics2.set_yticks([0,0.4,0.8,1.2])
#dynamics2.set_xlabel('Time (ms)')
#dynamics2.set_ylabel('Rate')
plt.savefig('PA1_1.pdf', bbox_inches='tight')
#plt.show()
print 'PA1_1.pdf stored'

#------------------------------------------------------------------
# Dynamics At Least and At Most Fixed Point 5 Neurons in Hight Rate
#------------------------------------------------------------------


tsim=20.
s=0.7
w=1.9
sdelmin=s
sdelmax=s
wmaxmax=w
wmaxmin=w
sdel=np.linspace(sdelmin,sdelmax,k)
wmax=np.linspace(wmaxmin,wmaxmax,k)

y0_true=1.*np.zeros(n+1)
y0_true[0]=1.
ypw_true1,timepw_true1=rk4(field_true_pw,y0_true,0.1,tsim)
y0_true=1.*np.ones(n+1)
y0_true[-1]=0.
ypw_true2,timepw_true2=rk4(field_true_pw,y0_true,0.1,tsim)

figure=plt.figure()
colormap = plt.cm.Accent
#dynamics
#dynamics=figure.add_subplot(211)
dynamics=figure.add_subplot(111)
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
dynamics.plot(10.*np.array(timepw_true1),phi(ypw_true1[:,0:n],theta,uc),lw=4)
#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
dynamics.set_xticks([0,100,200])
dynamics.set_yticks([0.5,1.])
dynamics.set_xlim([0,10*tsim])
dynamics.set_ylim([0,1.2])
plt.tick_params(labelsize=35)
#dynamics.set_xlabel('Time (ms)')
dynamics.set_ylabel('Rate',fontsize=40)


#dynamics
#dynamics2=figure.add_subplot(212)
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
#dynamics2.plot(10.*np.array(timepw_true2),phi(ypw_true2[:,0:n],theta,uc),lw=2.)
#dynamics2.set_xlim([0,10*tsim])
#dynamics2.set_ylim([0,1.2])
#dynamics2.set_yticks([0,0.4,0.8,1.2])
#dynamics2.set_xlabel('Time (ms)')
#dynamics2.set_ylabel('Rate')
plt.savefig('PA4_4.pdf', bbox_inches='tight')
#plt.show()

print 'PA4_4.pdf stored'









