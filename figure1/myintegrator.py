import numpy as np




def rk4(f,y0,a0,I0,dt,T):
	mysolu=[]
	mysola=[]
	mysolI=[]
	mytime=[]
	t=0
	un=y0
	an=a0
	In=I0
	mytime.append(t)
	mysolu.append(un)
	mysola.append(an)
	mysolI.append(In)
	while t<=T:
		k1_e,k1_a,k1_I=f(un,an,In,t)
		k2_e,k2_a,k2_I=f(un+(dt/2)*k1_e,an+(dt/2)*k1_a,In+(dt/2)*k1_I,t+dt/2)
		k3_e,k3_a,k3_I=f(un+(dt/2)*k2_e,an+(dt/2)*k2_a,In+(dt/2)*k2_I,t+dt/2)
		k4_e,k4_a,k4_I=f(un+dt*k3_e,an+dt*k3_a,In+dt*k3_I,t+dt)
		un=un+(dt/6)*(k1_e+2*k2_e+2*k3_e+k4_e)
		an=an+(dt/6)*(k1_a+2*k2_a+2*k3_a+k4_a)
		In=In+(dt/6)*(k1_I+2*k2_I+2*k3_I+k4_I)
		t=t+dt
		mysolu.append(un)
		mysola.append(an)
		mysolI.append(In)
		mytime.append(t)
		print(t)
	
	return np.array(mysolu),np.array(mysola),np.array(mysolI),np.array(mytime)


