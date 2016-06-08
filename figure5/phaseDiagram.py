import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt



tau=10.   #timescale of populations
tau_H=100.#10000
WI=0.1
a1=2.
b1=-1.5
y0=0.07
w=0.5

def phi_tanh(x):
	return 0.5*(1+np.tanh(a1*(x+b1)))

def nullcline_u(u):
	return u/(w*phi_tanh(u))+WI/w

def nullcline_H_Amended(u):
	return 1.-u/y0


def fieldSimpleModel(x,t):
	field_u=(1/tau)*(-x[0]+w*x[1]*phi_tanh(x[0])-WI*phi_tanh(x[0]))
	field_H=x[1]*(1.-(x[0]/y0))/tau_H
	return np.array([field_u,field_H])


def fieldSimpleModelAmended(x,t):
	field_u=(1/tau)*(-x[0]+w*x[1]*phi_tanh(x[0])-WI*phi_tanh(x[0]))
	field_H=(x[1]*(1.-(x[0]/y0))-x[1]*x[1])/tau_H
	return np.array([field_u,field_H])

umin=-0.1
umax=2
Hmin=0.1
Hmax=400
u_null=np.linspace(umin,umax,500)
#----------------------------------------------------------------------
#-------------------- Model--------------------------------------------
#----------------------------------------------------------------------
plt.axvline(y0, color='red',lw=3)
w=0.1
plt.plot(u_null,nullcline_u(u_null),color='b',lw=4,label=r'$w=0.1$')
Hstar_0_1=nullcline_u(y0)
plt.plot(y0,Hstar_0_1,'ob',markersize=12,alpha=1)
myh0=np.linspace(Hstar_0_1*0.7,Hstar_0_1*1.8,30)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e3, 1e4)
	sol = odeint(fieldSimpleModel,x0,t)
	plt.plot(sol[:,0],sol[:,1],color='b',alpha=0.2)
w=0.5
plt.plot(u_null,nullcline_u(u_null),color='g',lw=3,label=r'$w=0.5$')
Hstar_0_5=nullcline_u(y0)
plt.plot(y0,Hstar_0_5,'og',markersize=12,alpha=0.9)
myh0=np.linspace(Hstar_0_5*0.7,Hstar_0_5*2,30)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e3, 1e4)
	sol = odeint(fieldSimpleModel,x0,t)
	plt.plot(sol[:,0],sol[:,1],color='g',alpha=0.2)
w=1.2
plt.plot(u_null,nullcline_u(u_null),color='m',lw=3,label=r'$w=1.2$')
Hstar_1_2=nullcline_u(y0)
plt.plot(y0,Hstar_1_2,'om',markersize=12,alpha=0.9)
myh0=np.linspace(Hstar_1_2*0.7,Hstar_1_2*2,30)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e3, 1e4)
	sol = odeint(fieldSimpleModel,x0,t)
	plt.plot(sol[:,0],sol[:,1],color='m',alpha=0.2)


plt.xlim([umin,umax])
plt.ylim([Hmin,Hmax])
plt.xticks([0,1,2],size=30)
plt.yticks([0,100,200,300,400],size=30)
plt.legend(loc='upper right',fontsize=25)
plt.xlabel(r'$u$',size=40)
plt.ylabel(r'$H$',size=40)
plt.savefig('SimpleModel.pdf', bbox_inches='tight')
#plt.show(plot1) 
plt.close()

#--------------------------------------------------------------------
#----------------- Model Amended-------------------------------------
#---------------------------------------------------------------------

plt.plot(u_null,nullcline_H_Amended(u_null), color='red',lw=4)
w=0.1
plt.plot(u_null,nullcline_u(u_null),color='b',lw=4,label=r'$w=0.1$')
sol=root(fieldSimpleModelAmended,np.array([0.01,0.8]),args=(0))
H_star=sol.x[1]
u_star=sol.x[0]
plt.plot(u_star,H_star,'ob',markersize=12,alpha=1)
myh0=np.linspace(Hstar_0_1*0.7,Hstar_0_1*1.8,30)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e3, 1e4)
	sol = odeint(fieldSimpleModelAmended,x0,t)
	plt.plot(sol[:,0],sol[:,1],color='b',alpha=0.2)
w=0.5
plt.plot(u_null,nullcline_u(u_null),color='g',lw=3,label=r'$w=0.5$')
sol=root(fieldSimpleModelAmended,np.array([0.01,0.8]),args=(0))
H_star=sol.x[1]
u_star=sol.x[0]
plt.plot(u_star,H_star,'og',markersize=12,alpha=1)
myh0=np.linspace(Hstar_0_5*0.7,Hstar_0_5*1.8,30)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e3, 1e4)
	sol = odeint(fieldSimpleModelAmended,x0,t)
	plt.plot(sol[:,0],sol[:,1],color='g',alpha=0.2)
w=1.2
plt.plot(u_null,nullcline_u(u_null),color='m',lw=3,label=r'$w=1.2$')
sol=root(fieldSimpleModelAmended,np.array([0.01,0.8]),args=(0))
H_star=sol.x[1]
u_star=sol.x[0]
plt.plot(u_star,H_star,'om',markersize=12,alpha=1)
myh0=np.linspace(Hstar_1_2*0.7,Hstar_1_2*2,30)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e3, 1e4)
	sol = odeint(fieldSimpleModelAmended,x0,t)
	plt.plot(sol[:,0],sol[:,1],color='m',alpha=0.2)


plt.xlim([umin,umax])
plt.ylim([Hmin,Hmax])
plt.xticks([0,1,2],size=30)
plt.yticks([0,100,200,300,400],size=30)
plt.xlabel(r'$u$',size=40)
plt.ylabel(r'$H$',size=40)
plt.savefig('SimpleModelAmmended.pdf', bbox_inches='tight')
plt.xlim([0.,0.005])
plt.ylim([0,1.5])
plt.xticks([0.0025,0.005],size=30)
plt.yticks([0,0.5,1,1.5],size=30)
plt.xlabel(r'$u$',size=40)
plt.ylabel(r'$H$',size=40)
plt.savefig('SimpleModelAmmendedZoom.pdf', bbox_inches='tight')
plt.close() 











# nullcline








# nullcline


