import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import math as mt
from stimulus import *
from myintegrator import *
import scipy.integrate as integrate
import pickle
from functions import *
#import matplotlib as mpl
#-------------------------------------------------------------------
#-----------------Stimulation of Populations------------------------
#-------------------------------------------------------------------

# setting up the simulation 
#initial conditions
net=Network()
x0=0.01*np.ones((net.npts,net.n))
W0=[net.w0*np.ones((net.n,net.n)) for i in range(net.npts)]
H0=[0.01*np.ones(net.n) for i in range(net.npts)]
norbits=2
nperiod=2

delta_T=[[7.,14.],[50.,40.],[5.,13.],[20.,8.5]]

# storing dynamics weights
the_allRecurrent=[]
the_allFF=[]
the_allRecurrentTheo=[]
the_allFFTheo=[]
# storing rate dynamics
the_mydyn=[]
for wi in [1.,2.]:
	# storing dynamics weights
	allRecurrent=[]
	allFF=[]
	allRecurrentTheo=[]
	allFFTheo=[]
	# storing rate dynamics
	mydyn=[]
	for param in delta_T:
		net.w_i=wi # setting up the inhibition level
		elT=param[0]
		eldelta=param[1]
		net.tau_learning=400.
		net.setStim(net.amp,eldelta,elT,net.times) # setting up the stimulus
		valueRecurrent=[net.w0]
		valueFF=[net.w0]
		#integrator
		tmax=net.times*(net.lagStim+net.n*(elT+eldelta))+net.mystim.delay_begin+5000.
		thetmax=tmax
		theintegrator=myintegrator(net.delay,net.dt,net.n,thetmax)
		theintegrator.fast=False
		u,connectivity,W01,myH,t=theintegrator.DDE_Norm_Miller(net.field,x0,W0,H0,net.mystim.stim)
		for k in range(net.times):
			eltiempo=int((net.mystim.delay_begin+k*net.n*(elT+eldelta)+net.lagStim*(k-1)+net.lagStim/2.)/net.dt)
			valueRecurrent.append(connectivity[eltiempo,5,5])
			valueFF.append(connectivity[eltiempo,6,5])

		a0_1=np.zeros((net.npts,net.n))
		x0_1=np.zeros((net.npts,net.n))
		x0_1[:,0]=np.ones(net.npts)
		W0_1=[connectivity[-1,:,:] for i in range(net.npts)]
		H0_1=[np.ones(net.n) for i in range(net.npts)]
			
		net.setStim(0,net.delta,net.T,net.times)
		net.tau_learning=3e60
		thetmax=1000.
		theintegrator=myintegrator(net.delay,net.dt,net.n,thetmax)
		theintegrator.fast=False
		u1,connectivity1,W011,myH1,t1=theintegrator.DDE_Norm_Miller(net.field,x0_1,W0_1,H0_1,net.mystim.stim)
		net.setStim(5.5,net.delta,net.T,net.times) # no division by 0
		mydyn.append(u1)

		print('------------------------------------------------')
		print 'Orbit with delta',eldelta,elT
		net.tau_learning=400
		allRecurrentTheo.append(net.recurrentTheo(elT,100))
		allFFTheo.append(net.feedforwardTheo(elT,eldelta,100))
		allRecurrent.append(valueRecurrent)
		allFF.append(valueFF)
	#storing results simulations	
	the_allRecurrent.append(allRecurrent)
	the_allFF.append(allFF)
	the_allRecurrentTheo.append(allRecurrentTheo)
	the_allFFTheo.append(allFFTheo)
	the_mydyn.append(mydyn)
#------------------------------------------------------------------------
#-----Plotting-----------------------------------------------------------
#------------------------------------------------------------------------

fig = plt.figure(figsize=(30, 24))
# principal figure
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.2,hspace=0.3)

#figure low inh
gs0 = gridspec.GridSpec(2, 2)
gs0.update(wspace=0.13,hspace=0.1,left=0.57,right=0.95,top=0.8805,bottom=0.545)
axbif_low = plt.subplot(gs[0,0])
ax1_low = plt.subplot(gs0[0,0])
ax2_low= plt.subplot(gs0[0,1])
ax3_low = plt.subplot(gs0[1,0])
ax4_low = plt.subplot(gs0[1,1])
list_axis_low=[ax1_low,ax2_low,ax3_low,ax4_low]

#figure high inh
gs1 = gridspec.GridSpec(2, 2)
gs1.update(wspace=0.13,hspace=0.1,left=0.57,right=0.95,top=0.445,bottom=0.1096)
axbif_high = plt.subplot(gs[1,0])
ax1_high = plt.subplot(gs1[0,0])
ax2_high= plt.subplot(gs1[0,1])
ax3_high = plt.subplot(gs1[1,0])
ax4_high = plt.subplot(gs1[1,1])
list_axis_high=[ax1_high,ax2_high,ax3_high,ax4_high]

#rc={'axes.labelsize':55, 'font.size': 45, 'legend.fontsize': 25, 'axes.titlesize': 35}
#plt.rcParams.update(**rc)
stick=45
sfont=50
stitle=55
xpostitle=1.1 # x position title
ypostitle=1.035 # y pos title
#labels
yposlabels=1.03
xposlabels_1col=170
xposlabels_2col=310

colormap = plt.cm.Accent
# low  inhibition
for l in range(len(the_mydyn[0])):
	#dynamics
	list_axis_low[l].set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,net.n)]))
	list_axis_low[l].plot(t1,net.phi(the_mydyn[0][l]),lw=6)
	#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
	list_axis_low[l].set_ylim([0,1.2])
	if l==0:
		list_axis_low[l].set_yticks([0.5,1.])
		list_axis_low[l].set_xticks([])			
		list_axis_low[l].set_ylabel('Rate',fontsize=sfont)
		list_axis_low[l].text(xposlabels_1col+150,yposlabels, 'SA',fontsize=sfont)
		list_axis_low[l].set_xlim([0,450])
		list_axis_low[l].tick_params(axis='both', which='major', labelsize=stick)
	elif l==1:
		list_axis_low[l].set_yticks([])
		list_axis_low[l].set_xticks([])			
		list_axis_low[l].text(xposlabels_2col,yposlabels, 'PA/SA',fontsize=sfont)
		list_axis_low[l].set_xlim([0,500])
	elif l==2:
		list_axis_low[l].set_yticks([0.5,1.])
		list_axis_low[l].set_xticks([0,200,400])
		#list_axis_low[l].set_xticklabels(['0','0.2','0.4'])
		list_axis_low[l].set_xlabel('Time (ms)',fontsize=sfont)
		list_axis_low[l].set_ylabel('Rate',fontsize=sfont)
		list_axis_low[l].text(xposlabels_1col+150,yposlabels, 'dSA',fontsize=sfont)
		list_axis_low[l].set_xlim([0,450])
		list_axis_low[l].tick_params(axis='both', which='major', labelsize=stick)
	elif l==3:
		list_axis_low[l].set_yticks([])
		list_axis_low[l].set_xticks([0,200,400])			
		#list_axis_low[l].set_xticklabels(['0','0.2','0.4'])
		list_axis_low[l].set_xlabel('Time (ms)',fontsize=sfont)
		list_axis_low[l].text(xposlabels_2col,yposlabels, 'PA',fontsize=sfont)
		list_axis_low[l].set_xlim([0,500])
		list_axis_low[l].tick_params(axis='both', which='major', labelsize=stick)
# high  inhibition
for l in range(len(the_mydyn[1])):
	#dynamics
	list_axis_high[l].set_prop_cycle(plt.cycler('color',[colormap(i) for i in np.linspace(0, 0.9,net.n)]))
	list_axis_high[l].plot(t1,net.phi(the_mydyn[1][l]),lw=6)
	#dynamics.plot(timepw_true_approx,phi(ypw_true_approx[:,0:n],theta,uc),lw=2,color='b')
	list_axis_high[l].set_ylim([0,1.2])
	if l==0:
		list_axis_high[l].set_yticks([0.5,1.])
		list_axis_high[l].set_xticks([])			
		list_axis_high[l].set_ylabel('Rate',fontsize=sfont)
		list_axis_high[l].text(xposlabels_1col,yposlabels, 'SA',fontsize=sfont)
		list_axis_high[l].set_xlim([0,250])
		list_axis_high[l].tick_params(axis='both', which='major', labelsize=stick)
	elif l==1:
		list_axis_high[l].set_yticks([])
		list_axis_high[l].set_xticks([])			
		list_axis_high[l].text(xposlabels_2col,yposlabels, 'PA/SA',fontsize=sfont)
		list_axis_high[l].set_xlim([0,500])
	elif l==2:
		list_axis_high[l].set_yticks([0.5,1.])
		list_axis_high[l].set_xticks([0,100,200])
		list_axis_high[l].set_xlabel('Time (ms)',fontsize=sfont)
		list_axis_high[l].set_ylabel('Rate',fontsize=sfont)
		list_axis_high[l].text(xposlabels_1col,yposlabels,'dSA',fontsize=sfont)
		list_axis_high[l].set_xlim([0,250])
		list_axis_high[l].tick_params(axis='both', which='major', labelsize=stick)
	elif l==3:
		list_axis_high[l].set_yticks([])
		list_axis_high[l].set_xticks([0,200,400])			
		#list_axis_high[l].set_xticklabels(['0','200','400'])
		list_axis_high[l].set_xlabel('Time (ms)',fontsize=sfont)
		list_axis_high[l].text(xposlabels_2col,yposlabels, 'PA',fontsize=sfont)
		list_axis_high[l].set_xlim([0,500])
		list_axis_high[l].tick_params(axis='both', which='major', labelsize=stick)


#------------------------------------------------------------------
#---------------Bifurcation Diagram--------------------------------
#------------------------------------------------------------------
# This par of the code is to build a bifurcation diagram 
# that depends on the stimulation parameters T and delta -> period,delta


mywi=1. # value wi
bifcurve=np.load('mybifcurve2.npy')
mys=np.linspace(0,2.0,100)
myw=np.linspace(0,2.0,100)

colormap = plt.cm.afmhot
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,net.n)])

mymarkers=['o','+','h','s']
sizemarker=360
axbif_low.text(0.04, 1.7, 'PA',fontsize=sfont)
axbif_low.text(1.61, 1.72, 'PA/SA',fontsize=sfont)
axbif_low.text(1.55, 0.95, 'SA',fontsize=sfont)
axbif_low.text(1.52, 0.1, 'dSA',fontsize=sfont)

for i in range(norbits*nperiod):
	if i==1:
		axbif_low.scatter(the_allFF[0][i],the_allRecurrent[0][i],marker=mymarkers[i],s=sizemarker,facecolors='k')
	else:
		axbif_low.scatter(the_allFF[0][i],the_allRecurrent[0][i],marker=mymarkers[i],s=sizemarker,facecolors='none',edgecolors='k')

for i in range(norbits*nperiod):
	axbif_low.plot(the_allFFTheo[0][i],the_allRecurrentTheo[0][i],color='r',alpha=0.7,lw=7)


upperBsequences=np.array([1+mywi*(1+0.)/net.n for j in range(0,len(bifcurve[:,1]))])
axbif_low.plot(bifcurve[:,0],bifcurve[:,1],'k')
axbif_low.plot(mys,np.array([1+(mywi+0.)/net.n for i in range(0,100)]),c='k',lw=1)
axbif_low.fill_between(bifcurve[:,0],bifcurve[:,1],upperBsequences,alpha=0.5,edgecolor='k', facecolor='red',linewidth=0)
axbif_low.fill_between(np.linspace(bifcurve[0,0],2,100),np.zeros(100),(1.+(mywi+0.)/net.n)*np.ones(100),alpha=0.5,edgecolor='red', facecolor='red',linewidth=0)
axbif_low.fill_between(bifcurve[:,0],np.zeros(len(bifcurve[:,1])),bifcurve[:,1],alpha=0.5, facecolor='darkgrey',linewidth=0)
axbif_low.fill_between(np.linspace(0,bifcurve[-1,0],100),np.zeros(100),(1.+(mywi+0.)/net.n)*np.ones(100),alpha=0.5,edgecolor='k', facecolor='darkgrey',linewidth=0)


colormap = plt.cm.winter 


alph=0.15
for i in range(1,net.n):
	for j in range(0,i):
		myline2=np.linspace(mywi*(j+0.)/net.n,mywi*(j+1.)/net.n,100)
		myconstant1=np.array([1+mywi*(i+0.)/net.n for l in range(0,100)])
		axbif_low.fill_between(myline2,myconstant1,myconstant1+mywi/net.n,alpha=alph,edgecolor='grey', facecolor=colormap((j+0.)/net.n)[0:3])
	alph=alph+(0.95-0.15)/9
for i in range(1,net.n):
	myline1=np.linspace(mywi*(i+0.)/net.n,2.,100)
	myconstant1=np.array([1+mywi*(i+0.)/net.n for l in range(0,100)])
	axbif_low.fill_between(myline1,myconstant1,myconstant1+mywi/net.n,alpha=0.1*i,edgecolor='grey', facecolor='green')

myconstant1=np.array([2. for j in range(0,100)])
myconstant2=np.array([2. for j in range(0,100)])

alph=0.15
for j in range(0,net.n):
	myline2=np.linspace(mywi*(j+0.)/net.n,mywi*(j+1.)/net.n,100)
	axbif_low.fill_between(myline2,myconstant1,myconstant2,alpha=1.,edgecolor='grey', facecolor=colormap((j+0.)/net.n)[0:3])


axbif_low.set_xlim([0.,2.])
axbif_low.set_ylim([0,2.])
axbif_low.set_yticks([1,2])
axbif_low.set_xticks([0,1.,2.])
axbif_low.set_xlabel(r'Feed-forward ($s$)',fontsize=sfont)
axbif_low.set_ylabel(r'Recurrent ($w$)',fontsize=sfont)
axbif_low.tick_params(axis='both', which='major', labelsize=stick)
axbif_low.set_title('(A)',fontsize=stitle,x=xpostitle,y=ypostitle)

#------------------------------------------------------------------
#---------------Bifurcation Diagram High Inhibition-----------------
#------------------------------------------------------------------
# This par of the code is to build a bifurcation diagram 
# that depends on the stimulation parameters T and delta -> period,delta

mywi=2. # value wi
bifcurve=np.load('mybifcurve.npy')
mys=np.linspace(0,2.0,100)
myw=np.linspace(0,2.0,100)

colormap = plt.cm.afmhot
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,net.n)])

mymarkers=['o','+','h','s']
sizemarker=360
axbif_high.text(0.04, 1.7, 'PA',fontsize=sfont)
axbif_high.text(1.61, 1.72, 'PA/SA',fontsize=sfont)
axbif_high.text(1.55, 0.95, 'SA',fontsize=sfont)
axbif_high.text(1.52, 0.1, 'dSA',fontsize=sfont)

for i in range(norbits*nperiod):
	if i==1:
		axbif_high.scatter(the_allFF[1][i],the_allRecurrent[1][i],marker=mymarkers[i],s=sizemarker,facecolors='k')
	else:
		axbif_high.scatter(the_allFF[1][i],the_allRecurrent[1][i],marker=mymarkers[i],s=sizemarker,facecolors='none',edgecolors='k')

for i in range(norbits*nperiod):
	axbif_high.plot(the_allFFTheo[1][i],the_allRecurrentTheo[1][i],color='r',alpha=0.7,lw=7)


upperBsequences=np.array([1+mywi*(1+0.)/net.n for j in range(0,len(bifcurve[:,1]))])
axbif_high.plot(bifcurve[:,0],bifcurve[:,1],'k')
axbif_high.plot(mys,np.array([1+(mywi+0.)/net.n for i in range(0,100)]),c='k',lw=1)
axbif_high.fill_between(bifcurve[:,0],bifcurve[:,1],upperBsequences,alpha=0.5,edgecolor='k', facecolor='red',linewidth=0)
axbif_high.fill_between(np.linspace(bifcurve[0,0],2,100),np.zeros(100),(1.+(mywi+0.)/net.n)*np.ones(100),alpha=0.5,edgecolor='red', facecolor='red',linewidth=0)
axbif_high.fill_between(bifcurve[:,0],np.zeros(len(bifcurve[:,1])),bifcurve[:,1],alpha=0.5, facecolor='darkgrey',linewidth=0)
axbif_high.fill_between(np.linspace(0,bifcurve[-1,0],100),np.zeros(100),(1.+(mywi+0.)/net.n)*np.ones(100),alpha=0.5,edgecolor='k', facecolor='darkgrey',linewidth=0)


colormap = plt.cm.winter 


alph=0.15
for i in range(1,net.n):
	for j in range(0,i):
		myline2=np.linspace(mywi*(j+0.)/net.n,mywi*(j+1.)/net.n,100)
		myconstant1=np.array([1+mywi*(i+0.)/net.n for l in range(0,100)])
		axbif_high.fill_between(myline2,myconstant1,myconstant1+mywi/net.n,alpha=alph,edgecolor='grey', facecolor=colormap((j+0.)/net.n)[0:3])
	alph=alph+(0.95-0.15)/9
for i in range(1,net.n):
	myline1=np.linspace(mywi*(i+0.)/net.n,2.,100)
	myconstant1=np.array([1+mywi*(i+0.)/net.n for l in range(0,100)])
	axbif_high.fill_between(myline1,myconstant1,myconstant1+mywi/net.n,alpha=0.1*i,edgecolor='grey', facecolor='green')

myconstant1=np.array([2. for j in range(0,100)])
myconstant2=np.array([2. for j in range(0,100)])

alph=0.15
for j in range(0,net.n):
	myline2=np.linspace(mywi*(j+0.)/net.n,mywi*(j+1.)/net.n,100)
	axbif_high.fill_between(myline2,myconstant1,myconstant2,alpha=1.,edgecolor='grey', facecolor=colormap((j+0.)/net.n)[0:3])


axbif_high.set_xlim([0.,2.])
axbif_high.set_ylim([0,2.])
axbif_high.set_yticks([1,2])
axbif_high.set_xticks([0,1.,2.])
axbif_high.set_xlabel(r'Feed-forward ($s$)',fontsize=sfont)
axbif_high.set_ylabel(r'Recurrent ($w$)',fontsize=sfont)
axbif_high.tick_params(axis='both', which='major', labelsize=stick)
axbif_high.set_title('(B)',fontsize=stitle,x=xpostitle,y=ypostitle)


fig.savefig('fig8.pdf', bbox_inches='tight')
#plt.show()






