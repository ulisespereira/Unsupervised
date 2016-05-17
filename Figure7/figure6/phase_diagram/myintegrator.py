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

	def DDE(self,f,x0,W0):
		''' This is a method to solve delayed diferential equations'''
		
		# this method the delay is substracting, 
		memory=list(x0)# i.e. use the info from the past
		dyn=list(x0)
		myW_diag=[np.diag(w) for w in W0]
		myW0=list(W0)
		myW_off_diag=[np.diag(w,k=-1) for w in W0]
		n=int(np.floor((self.tmax-self.D)/self.dt)+1)
		t=self.D	
		time=list(np.linspace(0,self.D,self.npts))
		xn=memory[-1]
		Wn=W0[-1]
		memory=np.array(memory)
		for i in range(0,n):
			xn=xn+self.dt*f(t,memory,Wn)[0]
			Wn=Wn+self.dt*f(t,memory,Wn)[1]
			t=t+self.dt
			if self.fast==False:
				dyn.append(xn)	
				myW_diag.append(np.diag(Wn))
				myW_off_diag.append(np.diag(Wn,k=-1))
				myW0.append(Wn)
			time.append(t)
			memory=np.delete(memory,(0),axis=0)
			memory=np.vstack((memory,xn))
			print "Porcentage of the simulation done:",round(100.*float(i)/float(n),2)
		return  np.array(dyn),np.array(myW_diag),np.array(myW_off_diag),np.array(myW0),Wn,np.array(time)
	
	def DDE_Norm_additive(self,f,x0,W0):
		
		'''In this method we use the 'brute force' additive normalization  alone'''
		# this method the delay is substracting, 
		memory=list(x0)# i.e. use the info from the past
		dyn=list(x0)
		myW_diag=[np.diag(w) for w in W0]
		myW0=list(W0)
		myW_off_diag=[np.diag(w,k=-1) for w in W0]
		n=int(np.floor((self.tmax-self.D)/self.dt)+1)
		t=self.D	
		time=list(np.linspace(0,self.D,self.npts))
		xn=memory[-1]
		Wn=W0[-1]
		memory=np.array(memory)
		myN=len(xn)
		rowsum=[np.sum(w,axis=1) for w in W0]
		for i in range(0,n):
			xn=xn+self.dt*f(t,memory,Wn)[0]
			increaseW=list(self.dt*f(t,memory,Wn)[1])
			tot_increase=np.sum(increaseW,axis=1)
			N_increase=np.sum(Wn>0,axis=1)
			decreaseW=-np.outer(tot_increase/N_increase,np.ones(myN))	
			decreaseW[Wn<=0]=0.
			Wn=Wn+increaseW+decreaseW
			rowsum.append(np.sum(Wn,axis=1))
			t=t+self.dt
			if self.fast==False:
				dyn.append(xn)	
				myW_diag.append(np.diag(Wn))
				myW_off_diag.append(np.diag(Wn,k=-1))
				myW0.append(Wn)
			time.append(t)
			memory=np.delete(memory,(0),axis=0)
			memory=np.vstack((memory,xn))
			print "Porcentage of the simulation done:",round(100.*float(i)/float(n),2)		
		return  np.array(rowsum),np.array(dyn),np.array(myW_diag),np.array(myW_off_diag),np.array(myW0),Wn,np.array(time)
	
	def DDE_Norm_additive_adapt(self,f,a0,x0,W0):
		'''In this method we use the 'brute force' aditive normalization
		and include adaptation'''
		# this method the delay is substracting, 
		memory=list(x0)# i.e. use the info from the past
		dyn=list(x0)
		adapt=list(a0)
		myW_diag=[np.diag(w) for w in W0]
		myW0=list(W0)
		myW_off_diag=[np.diag(w,k=-1) for w in W0]
		n=int(np.floor((self.tmax-self.D)/self.dt)+1)
		t=self.D	
		time=list(np.linspace(0,self.D,self.npts))
		xn=memory[-1]
		Wn=W0[-1]
		an=a0[-1]
		memory=np.array(memory)
		myN=len(xn)
		rowsum=[np.sum(w,axis=1) for w in W0]
		for i in range(0,n):
			an=xn+self.dt*f(t,an,memory,Wn)[0]
			xn=xn+self.dt*f(t,an,memory,Wn)[1]
			#normalization part
			increaseW=list(self.dt*f(t,an,memory,Wn)[2])
			tot_increase=np.sum(increaseW,axis=1)
			N_increase=np.sum(Wn>0,axis=1)
			decreaseW=-np.outer(tot_increase/N_increase,np.ones(myN))	
			decreaseW[Wn<=0]=0.
			Wn=Wn+increaseW+decreaseW
			rowsum.append(np.sum(Wn,axis=1))
			dyn.append(xn)	
			adapt.append(an)
			t=t+self.dt
			if self.fast==False:
				myW_diag.append(np.diag(Wn))
				myW_off_diag.append(np.diag(Wn,k=-1))
				myW0.append(Wn)
			time.append(t)
			memory=np.delete(memory,(0),axis=0)
			memory=np.vstack((memory,xn))
			print "Porcentage of the simulation done:",round(100.*float(i)/float(n),2)		
		return  np.array(adapt),np.array(rowsum),np.array(dyn),np.array(myW_diag),np.array(myW_off_diag),np.array(myW0),Wn,np.array(time)
	
	def DDEInh(self,f,x0,W0,winh0):
		''' This is a method to solve delayed diferential equations'''
		
		# this method the delay is substracting, 
		memory=list(x0)# i.e. use the info from the past
		dyn=list(x0)
		myW_diag=[np.diag(w) for w in W0]
		myW0=list(W0)
		mywinh=list(winh0)
		myW_off_diag=[np.diag(w,k=-1) for w in W0]
		n=int(np.floor((self.tmax-self.D)/self.dt)+1)
		t=self.D	
		time=list(np.linspace(0,self.D,self.npts))
		xn=memory[-1]
		Wn=W0[-1]
		winhn=winh0[-1]
		memory=np.array(memory)
		for i in range(0,n):
			xn=xn+self.dt*f(t,memory,Wn,winhn)[0]
			Wn=Wn+self.dt*f(t,memory,Wn,winhn)[1]
			winhn=winhn+self.dt*f(t,memory,Wn,winhn)[2]
			t=t+self.dt
			if self.fast==False:
				dyn.append(xn)	
				mywinh.append(winhn)
				myW_diag.append(np.diag(Wn))
				myW_off_diag.append(np.diag(Wn,k=-1))
				myW0.append(Wn)
			time.append(t)
			memory=np.delete(memory,(0),axis=0)
			memory=np.vstack((memory,xn))
			print "Porcentage of the simulation done:",round(100.*float(i)/float(n),2)		
		return  np.array(dyn),np.array(myW_diag),np.array(myW_off_diag),np.array(myW0),Wn,np.array(mywinh),np.array(time)
	
	def DDE_Norm_additive_adapt_inh(self,f,a0,x0,W0,winh0):
		'''In this method we use the 'brute force' aditive normalization
		and include adaptation'''
		# this method the delay is substracting, 
		memory=list(x0)# i.e. use the info from the past
		dyn=list(x0)
		adapt=list(a0)
		myW_diag=[np.diag(w) for w in W0]
		myW0=list(W0)
		mywinh=list(winh0)
		myW_off_diag=[np.diag(w,k=-1) for w in W0]
		n=int(np.floor((self.tmax-self.D)/self.dt)+1)
		t=self.D	
		time=list(np.linspace(0,self.D,self.npts))
		xn=memory[-1]
		Wn=W0[-1]
		an=a0[-1]
		winhn=winh0[-1]
		memory=np.array(memory)
		myN=len(xn)
		rowsum=[np.sum(w,axis=1) for w in W0]
		for i in range(0,n):
			an=xn+self.dt*f(t,an,memory,Wn,winhn)[0]
			xn=xn+self.dt*f(t,an,memory,Wn,winhn)[1]
			winhn=winhn+self.dt*f(t,an,memory,Wn,winhn)[3]
			#normalization part
			increaseW=list(self.dt*f(t,an,memory,Wn,winhn)[2])
			tot_increase=np.sum(increaseW,axis=1)
			N_increase=np.sum(Wn>0,axis=1)
			decreaseW=-np.outer(tot_increase/N_increase,np.ones(myN))	
			decreaseW[Wn<=0]=0.
			Wn=Wn+increaseW+decreaseW
			rowsum.append(np.sum(Wn,axis=1))
			dyn.append(xn)	
			adapt.append(an)
			mywinh.append(winhn)
			t=t+self.dt
			if self.fast==False:
				myW_diag.append(np.diag(Wn))
				myW_off_diag.append(np.diag(Wn,k=-1))
				myW0.append(Wn)
			time.append(t)
			memory=np.delete(memory,(0),axis=0)
			memory=np.vstack((memory,xn))
			print "Porcentage of the simulation done:",round(100.*float(i)/float(n),2)		
		return  np.array(adapt),np.array(rowsum),np.array(dyn),np.array(myW_diag),np.array(myW_off_diag),np.array(myW0),Wn,np.array(mywinh),np.array(time)
	
	def DDE_Norm_Miller(self,f,a0,x0,W0,H0):
		'''In this method we use the 'brute force' aditive normalization
		and include adaptation'''
		# this method the delay is substracting, 
		memory=list(x0)# i.e. use the info from the past
		dyn=list(x0)
		adapt=list(a0)
		myW0=list(W0)
		myH=list(H0)
		n=int(np.floor((self.tmax-self.D)/self.dt)+1)
		t=self.D	
		time=list(np.linspace(0,self.D,self.npts))
		xn=memory[-1]
		Wn=W0[-1]
		an=a0[-1]
		Hn=H0[-1]
		memory=np.array(memory)
		myN=len(xn)
		for i in range(0,n):
			an=xn+self.dt*f(t,an,memory,Wn,Hn)[0]
			xn=xn+self.dt*f(t,an,memory,Wn,Hn)[1]
			Wn=Wn+self.dt*f(t,an,memory,Wn,Hn)[2]
			Hn=Hn+self.dt*f(t,an,memory,Wn,Hn)[3]

			dyn.append(xn)	
			adapt.append(an)
			myH.append(Hn)
			t=t+self.dt
			if self.fast==False:
				myW0.append(Wn)
			time.append(t)
			memory=np.delete(memory,(0),axis=0)
			memory=np.vstack((memory,xn))
#			print "Porcentage of the simulation done:",round(100.*float(i)/float(n),2)		
		return  np.array(adapt),np.array(dyn),np.array(myW0),Wn,np.array(myH),np.array(time)
	
