#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:42:18 2022

@author: wenzhengdong
L=2  Dyson 1~3

calculate the qubit observable .

In this script, the "i" is control index which starts from 1 !!!!!!
				the "n" is  window index which starts from 1 !!!!!!

"""

import numpy as np
from qutip import *
from joblib import Parallel, delayed
from datetime import datetime
from operator import add
import sys

cos = np.cos
sin = np.sin

sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1=  np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])


read_theta=np.load('result_D1_qns_theta.npy')
read_n = np.load('result_D1_qns_n.npy')
read_rho = np.load('result_D1_qns_rho.npy')
read_matrix = np.load('result_D1_qns_Matrix.npy')


TIME_NUMBER = 100 # partition of [0,T] 
GAMMA_FLIP = 0.02 # flipping rate gamma
G_COUPLING =0.01 # coupling
ENSEMBLE_SIZE = 6 # size of noise realization
T = 2
t_vec = np.linspace(0,T,TIME_NUMBER)



# C_qns controls // From QNS_files
theta_list = read_theta # QNS_theta From QNS_files
n_list = read_n # QNS_n 
init_list = read_rho # QNS_rho
O_to_S_matrix = read_matrix # from observabel to S matrix

ddd = np.shape(O_to_S_matrix)[1] # the number of exp we should compare


# qubbit dynamics solver
theta = lambda i,n : theta_list[i-1][n-1] # i,n starts with 1
nx = lambda i,n : n_list[i-1][n-1][0] # i,n starts with 1
ny = lambda i,n : n_list[i-1][n-1][1] # i,n starts with 1
nz = lambda i,n : n_list[i-1][n-1][2] # i,n starts with 1

def RTN_generator():
	"""
	1. produce a zero mean noise
	2. Ff we know that state is s at time t: z_s(t), then at t+dt, the flip to s' has probablity
	P_flip(t, t+dt) = e^(-gamma dt)
	"""
	dt = T/TIME_NUMBER
	trajectory_table = np.zeros((ENSEMBLE_SIZE,TIME_NUMBER))
	# make a constant noise
	for i in range(ENSEMBLE_SIZE):
		trajectory_table[i][0] = 1 # +1 !
		j=1
		while j<TIME_NUMBER:
			trajectory_table[i][j] =  trajectory_table[i][j-1] # constant noise not change in time
			j+=1
	"""
	for i in range(ENSEMBLE_SIZE):
		trajectory_table[i][0] = 1 if (np.random.uniform(0, 1)>0.5) else -1 # +1 or -1 even chance
		j=1
		while j<TIME_NUMBER:
			trajectory_table[i][j] = -1* trajectory_table[i][j-1] if (np.e**(-GAMMA_FLIP*dt) < np.random.uniform(0, 1)) \
									else trajectory_table[i][j-1]
			j+=1
	"""
	return trajectory_table
	
RTN_trajectories= RTN_generator() 	

def RTN(j, t):
	"""
	j is realization index, t is time. 
	"""
	return (RTN_trajectories[j, int(TIME_NUMBER*t/T)] if (t<T) else RTN_trajectories[j,-1])


def U_ctrl(i,t):
	#clacualte the propagator of H_ctrl Hamiltonian, the bath is left sigma[0].
	"""
	The propagator calculator based on rotation axis is piecewise constant
	"""
	my_window= 2 if t==T else int(t/(T/2))+1
	if my_window==1:
		U = cos(theta(i,1)*(t))*identity(2)-1j* sin(theta(i,1)*(t))*(nx(i,1)*sigmax()+ny(i,1)*sigmay()+nz(i,1)*sigmaz())
	else:
		U = (cos(theta(i,2)*(t-T/2))*identity(2)-1j* sin(theta(i,2)*(t-T/2))*(nx(i,2)*sigmax()+ny(i,2)*sigmay()+nz(i,2)*sigmaz()))\
			* (cos(theta(i,1)*(T/2))*identity(2)-1j* sin(theta(i,1)*(T/2))*(nx(i,1)*sigmax()+ny(i,1)*sigmay()+nz(i,1)*sigmaz()))	
	return tensor(U,identity(2))

def U_ctrl_dag(i,t):
	return U_ctrl(i,t).dag()


def U_sb(i,j,tt):
	"""
	i is C_ctrl index, j is j is realization index;
	t is time
	"""
	H_sb_tm=lambda t,args : G_COUPLING * (U_ctrl(i,t).dag() * tensor(sigmaz(), RTN(j,t)*identity(2)) * U_ctrl(i,t))
	#print('bbbb',H_sb_tm(1,1))
	f_sb_x = lambda t,args: np.trace(H_sb_tm(t,args)*tensor(sigmax(), identity(2)))
	f_sb_y = lambda t,args: np.trace(H_sb_tm(t,args)*tensor(sigmay(), identity(2)))
	f_sb_z = lambda t,args: np.trace(H_sb_tm(t,args)*tensor(sigmaz(), identity(2)))
	return propagator( [[tensor(sigmax(),identity(2)), f_sb_x],\
						[tensor(sigmay(),identity(2)), f_sb_y],\
						[tensor(sigmaz(),identity(2)), f_sb_z]],\
						tt, c_op_list=[], args={} )

def U_sb_dag(i,j,tt):
	return U_sb(i,j,tt).dag()

def obs_x_avg(i):
	"""
	<Ox>|i where i tells which ctrl and which init_state
    in exp_f: 1/2 since trace env produce extra 2 
	"""
	f_map = lambda s: identity(2) if (s==0) else (sigmax() if s==1 else ( sigmay() if s==2 else sigmaz()) )
	exp_f =lambda j: 1/2*np.trace(U_sb(i,j,T) * tensor(f_map(init_list[i-1]),identity(2))\
								* U_sb_dag(i,j,T)*tensor(sigmax(), identity(2)) )
	# init_list[i-1] since i starts from 1 
	obs_x_each = Parallel(n_jobs=-1, verbose=0)(delayed(exp_f)(j) for j in range (ENSEMBLE_SIZE) )
	return np.average(obs_x_each) # averge over all realization

###########################################################################
## Integrating trajectory and get <Ox>
###########################################################################

#obs_avg_simu = Parallel(n_jobs=-1, verbose=100)(delayed(obs_x_avg)(i) for i in range (6) )
"""
due to the need to do ensemble avarage in "obs_x_avg", where 'Parallel' is used, 
then iterating on C_qns should NOT use Parallel again, so make it simpel loop
"""

print("Start numerics: ",datetime.now().strftime("%H:%M:%S"))
obs_avg_simu = []# [obs_x_avg(i) for i in range (6)]
for i in range(1,ddd+1): # since i starts from 1.  
 	obs_avg_simu.append(obs_x_avg(i)) 
 	print('finish C_i, i = ',i)
obs_avg_simu=np.array(obs_avg_simu)     
print("end numerics: ",datetime.now().strftime("%H:%M:%S"))

np.save('experiment_D1_qns_observables', obs_avg_simu)

S_qns_results = np.array(O_to_S_matrix @ (np.matrix(obs_avg_simu).T) ) # from QND_file py

"""
since (M).(S) = O, then we know S = (M^-1). O 
"""







