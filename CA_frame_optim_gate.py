#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:42:18 2022

@author: wenzhengdong
L=2  Dyson 1~3

optimization of qubit control parameters 

"""
import numpy as np
from joblib import Parallel, delayed
from datetime import datetime
from operator import add
from qutip import * 
import pandas as pd 
from matplotlib import pyplot as plt
import scipy.optimize as opt

from CA_frame_functions import Cfx, Cfy, Cfz, D1_00, D1_01, D1_10,D1_11, D2_00, D2_01, D2_10,D2_11,\
								D3_00, D3_01, D3_10,D3_11,  D4_00, D4_01, D4_10,D4_11, \
								control_dynamics, sigma_x_T, sigma_y_T,sigma_z_T



sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1 = np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])



############################################################
#    functions used in gate optimization      
############################################################

"""
we know that the fidleity=1/8(2+ sum^3_{s=1} <O_s>_|rho_s)

"""
def fidelity(ctrl_params):
	"""
	Q-D Xu; make these variables global to avoid my unassigned issue
	"""
	#  pulse  parameters plugin
	theta_1, theta_2 = ctrl_params[0], ctrl_params[1]
	nx_1, ny_1, nz_1 = ctrl_params[2], ctrl_params[3], ctrl_params[4]
	nx_2, ny_2, nz_2 = ctrl_params[5], ctrl_params[6], ctrl_params[7]
	#I n "CA_frame_functions.py" the arg contains 'i'- which control [essential in QNS search], 
	#it is null in optmi gate !!!! SO i will be NONE
	theta =  lambda i,n : theta_1 if n==1 else (theta_1 if n==2 else print("bound error"))
	nx = lambda i,n : nx_1 if n==1 else (nx_1 if n==2 else print("bound error"))
	ny = lambda i,n : ny_1 if n==1 else (ny_1 if n==2 else print("bound error"))
	nz = lambda i,n : nz_1 if n==1 else (nz_1 if n==2 else print("bound error"))
	return -1/8* (2+sigma_x_T(control_dynamics(None),sigma_1))\
			+100*(abs(nx_1**2+ny_1**2+nz_1**1-1)+abs(nx_2**2+ny_2**2+nz_2**1-1)) 
	# return negative and max the -F.
	# Lagragians to make the vector normalization




############################################################
#    Input : knowledge of spectra      
############################################################

#load results: D2 qns 
read_D2_experiment = np.load('experiment_D2_qns_observables.py')
O_to_S_D2_matrix = np.load('result_D2_qns_Matrix.npy')
S_D2_results = np.array(O_to_S_D2_matrix @ (np.matrix(read_D2_experiment).T) ) #

#load results: D4 qns 
read_D4_experiment = np.load('experiment_D4_qns_observables.py')
O_to_S_D4_matrix = np.load('result_D4_qns_Matrix.npy')
S_D4_results = np.array(O_to_S_D4_matrix @ (np.matrix(read_D4_experiment).T) ) #



############################################################
#    k=4 non-G optimization 
############################################################

qns_data_s1 =  # 
qns_data_s2 =  # 
qns_data_s3 =  # 
qns_data_s4 =  # 

# get the PSD
S1_qns = lambda n,m : qns_data_s1[n-1][m-1]  # i,n starts with 1
S2_qns = lambda n1,n2,m1,m2: qns_data_s2[n1-1][n2-1][m1-1][m2-1]   # i,n starts with 1
S3_qns = lambda n1,n2,n3,m1,m2,m3 : qns_data_s3[n1-1][n2-1][n3-1][m1-1][m2-1][m3-1] # i,n starts with 1
S4_qns = lambda n1,n2,n3,n4,m1,m2,m3,m4: : qns_data_s4[n1-1][n2-1][n3-1][n4-1][m1-1][m2-1][m3-1][m4-1]

# replace the Sn_vec_func() to S_qns results/knowledge
S1_vec_func = lambda n,m : S1_qns(n,m)
S2_vec_func = lambda n1,n2,m1,m2: S2_qns(n1,n2,m1,m2)
S3_vec_func = lambda n1,n2,n3,m1,m2,m3: S3_qns(n1,n2,n3,m1,m2,m3)
S4_vec_func = lambda n1,n2,n3,n4,m1,m2,m3,m4: S4_qns(n1,n2,n3,n4,m1,m2,m3,m4: )


# optimization k4
initial_guess = [0.5, 0.5, 1,0,0, 1,0,0]
print("Start k4 optimization: ",datetime.now().strftime("%H:%M:%S"))
optimize_k4_sol = opt.minimize(fun = fidelity(),x0= initial_guess,args = (), method ='Nelder-Mead')
print("End k4 optimization: ",datetime.now().strftime("%H:%M:%S"))
optimal_k4_theta_1 = optimize_k4_sol.x[0]
optimal_k4_theta_2 = optimize_k4_sol.x[1]
optimal_k4_nx_1 = optimize_k4_sol.x[2]
optimal_k4_ny_1 = optimize_k4_sol.x[3]
optimal_k4_nz_1 = optimize_k4_sol.x[4]
optimal_k4_nx_2 = optimize_k4_sol.x[5]
optimal_k4_ny_2 = optimize_k4_sol.x[6]
optimal_k4_nz_2 = optimize_k4_sol.x[7]


############################################################
#    k=2 Gau optimization 
############################################################

qns_data_s1 =  # 
qns_data_s2 =  # 

# get the PSD
S1_qns = lambda n,m : qns_data_s1[n-1][m-1]  # i,n starts with 1
S2_qns = lambda n1,n2,m1,m2: qns_data_s2[n1-1][n2-1][m1-1][m2-1]   # i,n starts with 1
"""
to save the code redundancy, we make the Gaus-opt also stops at k=4, while 
making S_3==0 && S_4 ==0.
"""
S3_qns = lambda n1,n2,n3,m1,m2,m3 : 0  # k>2 ->0
S4_qns = lambda n1,n2,n3,n4,m1,m2,m3,m4:  0 # k>2 ->0

# replace the Sn_vec_func() to S_qns results/knowledge
S1_vec_func = lambda n,m : S1_qns(n,m)
S2_vec_func = lambda n1,n2,m1,m2: S2_qns(n1,n2,m1,m2)
S3_vec_func = lambda n1,n2,n3,m1,m2,m3: S3_qns(n1,n2,n3,m1,m2,m3)
S4_vec_func = lambda n1,n2,n3,n4,m1,m2,m3,m4: S4_qns(n1,n2,n3,n4,m1,m2,m3,m4: )



# optimization k2
initial_guess = [0.5, 0.5, 1,0,0, 1,0,0]
print("Start k2 optimization: ",datetime.now().strftime("%H:%M:%S"))
optimize_k2_sol = opt.minimize(fun = fidelity(),x0= initial_guess,args = (), method ='Nelder-Mead')
print("End k2 optimization: ",datetime.now().strftime("%H:%M:%S"))
optimal_k2_theta_1 = optimize_k2_sol.x[0]
optimal_k2_theta_2 = optimize_k2_sol.x[1]
optimal_k2_nx_1 = optimize_k2_sol.x[2]
optimal_k2_ny_1 = optimize_k2_sol.x[3]
optimal_k2_nz_1 = optimize_k2_sol.x[4]
optimal_k2_nx_2 = optimize_k4_sol.x[5]
optimal_k2_ny_2 = optimize_k2_sol.x[6]
optimal_k2_nz_2 = optimize_k2_sol.x[7]
