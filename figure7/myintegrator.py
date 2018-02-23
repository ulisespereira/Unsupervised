import  numpy as np

""" This is an adhoc integrator made for DDE"""

class myintegrator:
	def __init__(self,D,dt,N,tmax):
		self.npts=np.floor(D/dt)+1         # points delay
		self.dt=dt
		self.N=N
		self.tmax=tmax
		self.D=D
		self.fast=True

	
	def DDE_Inh(self,f,x0,uI0,W0,WEI0,stim):
		'''In this method we use the 'brute force' aditive normalization
		and include adaptation'''
		# this method the delay is substracting, 
		memory=list(x0)# i.e. use the info from the past
		myu=list(x0)
		myuI=list(uI0)
		myW=list(W0)
		myWEI=list(WEI0)
		n=int(np.floor((self.tmax-self.D)/self.dt)+1)
		t=self.D	
		time=list(np.linspace(0,self.D,self.npts))
		
		#initial conditions
		un=memory[-1]
		uIn=uI0[-1]
		Wn=W0[-1]
		WEIn=WEI0[-1]

		memory=np.array(memory)
		for i in range(0,n):
			un=un+self.dt*f(t,memory,uIn,Wn,WEIn,stim)[0]
			uIn=uIn+self.dt*f(t,memory,uIn,Wn,WEIn,stim)[1]
			Wn=Wn+self.dt*f(t,memory,uIn,Wn,WEIn,stim)[2]
			WEIn=WEIn+self.dt*f(t,memory,uIn,Wn,WEIn,stim)[3]

			myu.append(un)	
			myuI.append(uIn)
			myWEI.append(WEIn)
			t=t+self.dt
			if self.fast==False:
				myW.append(Wn)
			time.append(t)
			memory=np.delete(memory,(0),axis=0)
			memory=np.vstack((memory,un))
			print "Porcentage of the simulation done:",round(100.*float(i)/float(n),2)		
		myW[-1]=Wn # the  last entry is the last connectivity matrix
		return  np.array(myu),np.array(myuI),np.array(myW),np.array(myWEI),np.array(time)
