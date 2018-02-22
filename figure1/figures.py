import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import cPickle as pickle 
from functions import *
''' This plots the bifurcation diagram in fig1 and transfer function for methods.'''



#ploting the bifurcation diagram 


#loading data
dynamics_bif = pickle.load(open('dynamics_bif.p','rb'))
colormap = plt.cm.Accent

# size ticks
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18


figure=plt.figure(figsize=(10,10))
figure.subplots_adjust(hspace = .3,wspace=.3) # vertical space bw figures

label_size = 22 # size labels
lw_alp = 1# lw gray dashed linear
alp = 1. # opacity gray dashed linear

n=14
# bifurcation diagram
mys = np.linspace(0,1.5,100)
myw = np.linspace(0,1.5,100)
upperBsequences = np.ones(100)
lowerBsequences = np.zeros(100)

fig1 = figure.add_subplot(221)
fig1.plot(mys,1-mys,c='k',lw=1)
fig1.fill_between(mys,1.-mys,lowerBsequences,alpha=0.5,edgecolor='k', facecolor='darkgrey',linewidth=0)
fig1.fill_between(mys,1.-mys,upperBsequences,alpha=0.5,edgecolor='k', facecolor='red')
fig1.fill_between(mys,upperBsequences,2*np.ones(100),alpha=0.5,edgecolor='k', facecolor='blue')
fig1.text(0.4,1.5,r'PA',fontsize = label_size + 6)
fig1.text(0.7,0.6,r'SA',fontsize = label_size + 6)
fig1.text(0.1,0.4,r'dSA',fontsize = label_size + 6)

fig1.set_xlim([0,1.0])
fig1.set_ylim([0,2.])
fig1.set_yticks([1,2])
fig1.set_xticks([0,0.5,1])
fig1.set_xlabel(r'Feed-forward (s)',fontsize=label_size)
fig1.set_ylabel(r'Recurrent (w)',fontsize=label_size)



fig2 = figure.add_subplot(222)
phi_pw_true,timepw_true =  dynamics_bif[2][0]
phi_pw_linear,timepw_linear = dynamics_bif[2][1]
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
fig2.plot(timepw_true,phi_pw_true[:,0:n],lw=2)
fig2.plot(timepw_linear,phi_pw_linear[:,0:n],color = 'gray',ls = '--',lw=lw_alp,alpha=alp)
fig2.text(260,0.85 * 3.,r'PA',fontsize = label_size + 6)
fig2.set_xlim([0,350])
fig2.set_xticks([100,200,300])
fig2.set_yticks([0,1,2,3])
#plt.tick_params(labelsize=35)
plt.ylim([0,3])
fig2.set_xlabel('Time (ms)',fontsize = label_size)
fig2.set_ylabel('Rate',fontsize = label_size)




fig3 = figure.add_subplot(223)

phi_pw_true,timepw_true =  dynamics_bif[1][0]
phi_pw_linear,timepw_linear = dynamics_bif[1][1]



plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
fig3.plot(timepw_true,phi_pw_true[:,0:n],lw=2)
fig3.plot(timepw_linear,phi_pw_linear[:,0:n],color = 'gray',ls = '--',lw=lw_alp,alpha=alp)
fig3.set_xlim([0,350])
fig3.set_xticks([100,200,300])
fig3.set_ylim([0,1.2])
fig3.set_yticks([0,1])
fig3.text(260,1.2 * 0.85,r'dSA',fontsize = label_size + 6)
#plt.tick_params(labelsize=35)
fig3.set_xlabel('Time (ms)',fontsize = label_size)
fig3.set_ylabel('Rate',fontsize = label_size)





fig4 = figure.add_subplot(224)
phi_pw_true,timepw_true =  dynamics_bif[0][0]
phi_pw_linear,timepw_linear = dynamics_bif[0][1]
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9,n)])
plt.rc('xtick',labelsize=28)
fig4.plot(timepw_true,phi_pw_true[:,0:n],lw=2)
fig4.plot(timepw_linear,phi_pw_linear[:,0:n],color= 'gray',ls='--',lw=lw_alp,alpha=alp)
fig4.text(260,3 * 0.85,r'SA',fontsize = label_size + 6)
fig4.set_xlim([0,350])
fig4.set_ylim([0,3.])
fig4.set_xticks([100,200,300])
fig4.set_yticks([0,1,2,3])
#plt.tick_params(labelsize=35)
fig4.set_xlabel('Time (ms)',fontsize = label_size)
fig4.set_ylabel('Rate',fontsize = label_size)
figure.savefig('bifurcation_diagram.pdf', bbox_inches='tight',transparent = True)
plt.close()
#plt.show()

# transfer function



plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
figure=plt.figure(figsize=(5,5))
figure.subplots_adjust(hspace = .3,wspace=.3) # vertical space bw figures

nu=2.
tau=10.
nu_brunel=0.4*nu
theta=-0.0
theta_brunel=-0.1
uc=1/nu
uc_brunel=uc
#print(1./(nu+sdel))
a1=6.
b1=-0.25

fig1 = figure.add_subplot(111)
myu=np.linspace(-.5,1.5,200)
fig1.plot(myu,phi(myu,nu,theta,uc),'k',lw=2.5,label='PL',ls = '-')
fig1.plot(myu,phi_tanh(myu,a1,b1),'k',lw=2.5,label='S',ls = '--')
fig1.plot(myu,phi_brunel(myu,nu_brunel,theta_brunel,uc_brunel),'k',lw=2.5,label='PNL',ls = ':')
fig1.set_xlim([-.2,0.8])
fig1.set_ylim([0,1.8])
fig1.set_xticks([-.2,0.3,0.8])
fig1.set_yticks([0.5,1,1.5])
#fig1.tick_params(labelsize=35)
fig1.set_xlabel('Current',fontsize = 20)
fig1.set_ylabel('Rate',fontsize = 20)
fig1.legend(loc='upper left',fontsize = 20)
figure.savefig('transferFunctions.pdf', bbox_inches='tight')
plt.close()
#plt.show()


