import numpy as np


class stimulus:
	def __init__(self,patterns,lag,delta,period,times):
		self.patterns=patterns
		self.tstim=period # duration of stimulation
		self.inten=0.1
		self.lag=lag # duration inter chunks of stimulations
		self.delta=delta # duration intra each pop stimulation
		self.delay_begin=50
		self.times=times #number of times of stimulation
		self.n=len(self.patterns)
		self.T_one_rep=(self.tstim+self.delta)*self.n+self.lag #time one stimulation
		self.T_one_stim=self.tstim+self.delta
		self.T_total=self.T_one_rep*self.times+self.delay_begin
	def stim(self,t):
		"""In this estimulation we deliver the patterns 1 by 1 self.times times
		you want to create first a  list with all the times where you are delivdring the patterns
		and then in this times you want to deliver the actual patterns,
		recall that that the commentarized line is bc I prefer not include 2 delays between the 
		number of times that I delivered the patterns
		"""

		t_actual=t-self.delay_begin
		if t<self.delay_begin :
			return np.zeros(self.n)
		elif t>self.T_total:
			return np.zeros(self.n)
		else:
			#times
			t_rs_rep=t_actual-self.T_one_rep*np.floor(t_actual/self.T_one_rep) # how many time after the last repetition
			t_rs_stim=t_rs_rep-self.T_one_stim*np.floor(t_rs_rep/self.T_one_stim) #how many time after last stimulation
			if t_rs_rep>(self.n*self.T_one_stim) and t_rs_rep<self.T_one_rep:
				return np.zeros(self.n)
			else:
				if t_rs_stim>self.tstim and t_rs_stim<=self.T_one_stim:
					return np.zeros(self.n)
				else:
					stim_index=int(np.floor(t_rs_rep/self.T_one_stim)) # stimulation index
					return self.inten*self.patterns[stim_index]
		
