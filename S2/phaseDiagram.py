import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt



tau=10.   #timescale of populations
tau_H=100.#10000
WI=0.1
a1=2.
b1=-1.5
y0=0.005
w=0.5

def phi_tanh(x):
	return 0.5*(1+np.tanh(a1*(x+b1)))

def nullcline_u(u):
	return u/(w*phi_tanh(u))+WI/w

def nullcline_H_Amended(u):
	return 1.-phi_tanh(u)/y0


def fieldSimpleModel(x,t):
	field_u=(1/tau)*(-x[0]+w*x[1]*phi_tanh(x[0])-WI*phi_tanh(x[0]))
	field_H=x[1]*(1.-(phi_tanh(x[0])/y0))/tau_H
	return np.array([field_u,field_H])


def fieldSimpleModelAmended(x,t):
	field_u=(1/tau)*(-x[0]+w*x[1]*phi_tanh(x[0])-WI*phi_tanh(x[0]))
	field_H=(x[1]*(1.-(phi_tanh(x[0])/y0))-x[1]*x[1])/tau_H
	return np.array([field_u,field_H])

umin=-0.1
umax=2
Hmin=0.1
Hmax=75
u_null=np.linspace(umin,umax,500)


#----------------------------------------------------------------------
#-------------------- Model--------------------------------------------
#----------------------------------------------------------------------

rc={'axes.labelsize': 32, 'font.size': 30, 'legend.fontsize': 25, 'axes.titlesize': 30}
plt.rcParams.update(**rc)


ustar=np.arctanh(2*y0-1)/a1-b1
plt.axvline(ustar, color='red',lw=4)

w=0.5
plt.plot(u_null,nullcline_u(u_null),color='g',lw=4,label=r'$w=0.5$')
Hstar_0_5=nullcline_u(ustar)
plt.plot(ustar,Hstar_0_5,'og',markersize=12,alpha=0.9)
myh0=np.linspace(Hstar_0_5*0.9,Hstar_0_5*0.9,1)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e5, 1e6)
	sol = odeint(fieldSimpleModel,x0,t)
	plt.plot(sol[:,0],sol[:,1],':k',alpha=.5,lw=6)
w=2.2
plt.plot(u_null,nullcline_u(u_null),color='m',lw=4,label=r'$w=2.2$')
Hstar_1_2=nullcline_u(ustar)
plt.plot(ustar,Hstar_1_2,'om',markersize=12,alpha=0.9)
myh0=np.linspace(Hstar_1_2*0.9,Hstar_1_2*0.9,1)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e5, 1e6)
	sol = odeint(fieldSimpleModel,x0,t)
	plt.plot(sol[:,0],sol[:,1],'--k',alpha=.5,lw=6)


plt.xlim([umin,umax])
plt.ylim([Hmin,Hmax])
plt.xticks([0,1,2])
plt.yticks([0,50,100])
plt.legend(loc='upper right',fontsize=25)
plt.xlabel(r'$u$')
plt.ylabel(r'$H$')
plt.savefig('SimpleModel.pdf', bbox_inches='tight',transparent=True)
plt.close()

#--------------------------------------------------------------------
#----------------- Model Amended-------------------------------------
#---------------------------------------------------------------------

plt.plot(u_null,nullcline_H_Amended(u_null), color='red',lw=4)
w=0.5
plt.plot(u_null,nullcline_u(u_null),color='g',lw=4,label=r'$w=0.5$')
sol=root(fieldSimpleModelAmended,np.array([0.01,0.8]),args=(0))
H_star=sol.x[1]
u_star=sol.x[0]
plt.plot(u_star,H_star,'og',markersize=20,alpha=1)
myh0=np.linspace(Hstar_0_5*2.,Hstar_0_5*1.8,1)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e3, 1e4)
	sol = odeint(fieldSimpleModelAmended,x0,t)
	plt.plot(sol[:,0],sol[:,1],':k',lw=6,alpha=0.5)
w=1.2
plt.plot(u_null,nullcline_u(u_null),color='m',lw=4,label=r'$w=1.2$')
sol=root(fieldSimpleModelAmended,np.array([0.01,0.8]),args=(0))
H_star=sol.x[1]
u_star=sol.x[0]
plt.plot(u_star,H_star,'om',markersize=20,alpha=1)
myh0=np.linspace(Hstar_1_2*2.,Hstar_1_2*2,1)
for H0 in myh0: 
	x0=np.array([0.5,H0])
	t = np.linspace(0, 1e3, 1e4)
	sol = odeint(fieldSimpleModelAmended,x0,t)
	plt.plot(sol[:,0],sol[:,1],'--k',lw=6,alpha=0.5)


plt.xlim([umin,umax])
plt.ylim([Hmin,Hmax])
plt.xticks([0,1,2])
plt.yticks([0,50,100])
plt.xlabel(r'$u$')
plt.ylabel(r'$H$')
plt.legend(loc='lower right',fontsize=25)
plt.savefig('SimpleModelAmmended.pdf', bbox_inches='tight',transparent=True)
plt.xlim([-0.0001,0.005])
plt.ylim([0,1.2])
plt.xticks([0.0025,0.005])
plt.yticks([0,0.5,1])
plt.xlabel(r'$u$')
plt.ylabel(r'$H$')
plt.savefig('SimpleModelAmmendedZoom.pdf', bbox_inches='tight',transparent=True)
plt.close() 



myw=np.logspace(-2,3,400)
myHAmended=[]
myH=[]
i=0
for thew in myw:
	w=thew
	solAmmended=root(fieldSimpleModelAmended,np.array([0.01,.8]),args=(0))
	myHAmended.append(solAmmended.x[1])
	myH.append(nullcline_u(ustar))
#	print i,solAmmended.x[1]
	i=i+1

plt.loglog(myw,np.array(myHAmended),color='b',lw=4,label='Modified')
plt.loglog(myw,myH,lw=4,color='orange',label='Linear')
#plt.yscale('log')
plt.xlim([0.08,100])
plt.ylim([0.01,1000])
plt.xlabel(r'$w$')
plt.ylabel(r'$H^*$')
plt.xticks([0.1,1,10,100])
plt.yticks([0.1,1,10,100,1000])
plt.legend(loc='upper right',fontsize=23)
plt.savefig('Hvsw.pdf', bbox_inches='tight',transparent=True)
plt.yscale('linear')
plt.ylim([0.0,1.5])
plt.yticks([0.5,1.,1.5])
plt.savefig('HvswZoom.pdf', bbox_inches='tight',transparent=True)
#plt.show()




