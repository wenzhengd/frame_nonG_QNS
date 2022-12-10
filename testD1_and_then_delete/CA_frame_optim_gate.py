#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:42:18 2022

@author: wenzhengdong
L=2  Dyson 1~3

optimization of qubit control parameters 

The analytical expression of function 'control_dyanmcics' takes  _13_ min to finish
The analytical expression of function 'sigma_x_T ' takes _-1ç_ min to finish

"""
import numpy as np
from joblib import Parallel, delayed
from datetime import datetime
from operator import add
from qutip import * 
import pandas as pd 
from matplotlib import pyplot as plt
import scipy.optimize as opt
import sympy as sp


#from CA_frame_functions import Cfx, Cfy, Cfz, theta, nx, ny, nz,  \
#								D1_00, D1_01, D1_10,D1_11, D2_00, D2_01, D2_10,D2_11,\
#								D3_00, D3_01, D3_10,D3_11,  D4_00, D4_01, D4_10,D4_11, \
#								control_dynamics, sigma_x_T, sigma_y_T,sigma_z_T

sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1 = np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])


cos = np.cos
sin = np.sin



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
	#it is null in optmi gate !!!! SO 'i' will be NONE
	theta =  lambda i,n : theta_1 if n==1 else (theta_1 if n==2 else print("bound error"))
	nx = lambda i,n : nx_1 if n==1 else (nx_2 if n==2 else print("bound error"))
	ny = lambda i,n : ny_1 if n==1 else (ny_2 if n==2 else print("bound error"))
	nz = lambda i,n : nz_1 if n==1 else (nz_2 if n==2 else print("bound error"))
	# replace the Sn_vec_func() to S_qns results/knowledge
	S1_vec_func = lambda n,m : S1_qns(n,m)
	S2_vec_func = lambda n1,n2,m1,m2: S2_qns(n1,n2,m1,m2)
	S3_vec_func = lambda n1,n2,n3,m1,m2,m3: S3_qns(n1,n2,n3,m1,m2,m3)
	S4_vec_func = lambda n1,n2,n3,n4,m1,m2,m3,m4: S4_qns(n1,n2,n3,n4,m1,m2,m3,m4)	
	#####################################################
	
	#####################################################
	return -1/8* (2+sigma_x_T(control_dynamics(None),sigma_1) + 2 + 2 )\
			+100*(abs(nx_1**2+ny_1**2+nz_1**1-1)+abs(nx_2**2+ny_2**2+nz_2**1-1)) 
	# return negative and max the -F.
	# Lagragians to make the vector normalization




############################################################
#    Input : knowledge of spectra      
############################################################

#load results: D2 qns 
read_D2_experiment = np.load('experiment_D2_qns_observables.npy')
O_to_S_D2_matrix = np.load('result_D2_qns_Matrix.npy')
S_D2_results = np.real(np.array(O_to_S_D2_matrix @ (np.matrix(read_D2_experiment).T) )) #


#load results: D4 qns 
#read_D4_experiment = np.load('experiment_D4_qns_observables.py')
#O_to_S_D4_matrix = np.load('result_D4_qns_Matrix.npy')
#S_D4_results = np.array(O_to_S_D4_matrix @ (np.matrix(read_D4_experiment).T) ) #



############################################################
#    k=4 non-G optimization 
############################################################

"""
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
"""

############################################################
#    k=2 Gau optimization 
############################################################

qns_data_s1 = np.resize(np.array(S_D2_results[0:10]),(2,5))   # slice the results to S1 
qns_data_s2 = np.resize(np.array(S_D2_results[10:-1]),(2,2,5,5))  # slice the results to S2 

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
S4_vec_func = lambda n1,n2,n3,n4,m1,m2,m3,m4: S4_qns(n1,n2,n3,n4,m1,m2,m3,m4)


# Define some symbols
theta_1 = sp.Symbol('theta_1')
theta_2 = sp.Symbol('theta_2')
nx_1 = sp.Symbol('nx_1')
ny_1 = sp.Symbol('ny_1')
nz_1 = sp.Symbol('nz_1')
nx_2 = sp.Symbol('nx_2')
ny_2 = sp.Symbol('ny_2')
nz_2 = sp.Symbol('nz_2')

theta =  lambda i,n : theta_1 if n==1 else (theta_2 if n==2 else print("bound error"))
nx = lambda i,n : nx_1 if n==1 else (nx_2 if n==2 else print("bound error"))
ny = lambda i,n : ny_1 if n==1 else (ny_2 if n==2 else print("bound error"))
nz = lambda i,n : nz_1 if n==1 else (nz_2 if n==2 else print("bound error"))

def Cfx(i,n,m):
	#n is window, m is frame, i is which control
    if(n==1 and m==1):
        return 1.*nx(i,1)*nz(i,1) - nx(i,1)*nz(i,1)*((44.06035162512098*sp.sin(2*theta(i,1)))/theta(i,1) + (-1060.793564231141 + 107.4808595280735*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(71.65390635204905*theta(i,1) - 29.04023441674732*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) - ny(i,1)*((88.12070325024196*sp.sin(theta(i,1))**2)/theta(i,1) + (sp.sin(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + theta(i,1)*(71.65390635204898 - 29.040234416747314*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-71.65390635204905 + 29.04023441674732*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==2):
        return  -((nx(i,1)*nz(i,1)*(-23.541617541212254 + 9.541058216523389*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(35.31242631181839*theta(i,1) - 3.577896831196271*theta(i,1)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) - (ny(i,1)*(sp.sin(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + theta(i,1)*(-35.31242631181839 + 3.5778968311962718*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-35.31242631181839 + 3.577896831196271*theta(i,1)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)
    elif (n==1 and m==3):
        return -(nx(i,1)*nz(i,1)*((-68.42444032663414*sp.sin(2*theta(i,1)))/theta(i,1) + (1666.290634181943 - 168.83053934745612*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-112.55369289830418*theta(i,1) + 45.61629355108943*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))) - ny(i,1)*((-136.84888065326828*sp.sin(theta(i,1))**2)/theta(i,1) + (sp.sin(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + theta(i,1)*(-112.55369289830406 + 45.61629355108942*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(112.55369289830418 - 45.61629355108943*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==4):
        return -(nx(i,1)*nz(i,1)*((-29.040234416747317*sp.sin(2*theta(i,1)))/theta(i,1) + (707.1957094874274 - 71.65390635204899*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-50.236672001638375*theta(i,1) + 20.360156277831546*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))) - ny(i,1)*((-58.080468833494635*sp.sin(theta(i,1))**2)/theta(i,1) + (sp.sin(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + theta(i,1)*(-50.23667200163832 + 20.360156277831535*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(50.236672001638375 - 20.360156277831546*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==5):
        return -((nx(i,1)*nz(i,1)*(27.73431477040989 - 11.240294400188406*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-29.974118400502412*theta(i,1) + 3.0370131549744803*theta(i,1)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) - (ny(i,1)*(sp.sin(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + theta(i,1)*(29.97411840050242 - 3.037013154974481*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(29.974118400502412 - 3.0370131549744803*theta(i,1)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)
    elif (n==2 and m==1):
        return 1.*((nx(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. + (nx(i,1)**2*nx(i,2)*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) - sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) - (nx(i,2)*ny(i,1)**2*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. + nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) - sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) - (nx(i,2)*nz(i,1)**2*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. - nx(i,2)*ny(i,1)*nz(i,2)*sp.sin(2*theta(i,1)) + nx(i,2)*nz(i,1)*nz(i,2)*sp.sin(2*theta(i,1))) + (-0.5*(nx(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sp.sin(2*theta(i,1)))*((44.06035162512098*sp.sin(2*theta(i,2)))/theta(i,2) + (-1060.793564231141 + 107.4808595280735*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(71.65390635204905*theta(i,2) - 29.04023441674732*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*ny(i,2) - (sp.cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sp.sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*((88.12070325024196*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + theta(i,2)*(71.65390635204898 - 29.040234416747314*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-71.65390635204905 + 29.04023441674732*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==2):
        return ((-0.5*(nx(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sp.sin(2*theta(i,1)))*(-23.541617541212254 + 9.541058216523389*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(35.31242631181839*theta(i,2) - 3.577896831196271*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((-0.5*ny(i,2) - (sp.cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sp.sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*(sp.sin(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + theta(i,2)*(-35.31242631181839 + 3.5778968311962718*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-35.31242631181839 + 3.577896831196271*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    elif (n==2 and m==3):
        return (-0.5*(nx(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sp.sin(2*theta(i,1)))*((-68.42444032663414*sp.sin(2*theta(i,2)))/theta(i,2) + (1666.290634181943 - 168.83053934745612*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-112.55369289830418*theta(i,2) + 45.61629355108943*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*ny(i,2) - (sp.cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sp.sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*((-136.84888065326828*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + theta(i,2)*(-112.55369289830406 + 45.61629355108942*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(112.55369289830418 - 45.61629355108943*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==3):
        return (-0.5*(nx(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sp.sin(2*theta(i,1)))*((-29.040234416747317*sp.sin(2*theta(i,2)))/theta(i,2) + (707.1957094874274 - 71.65390635204899*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-50.236672001638375*theta(i,2) + 20.360156277831546*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*ny(i,2) - (sp.cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sp.sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*((-58.080468833494635*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + theta(i,2)*(-50.23667200163832 + 20.360156277831535*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(50.236672001638375 - 20.360156277831546*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    else :
        return ((-0.5*(nx(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sp.sin(2*theta(i,1)))*(27.73431477040989 - 11.240294400188406*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-29.974118400502412*theta(i,2) + 3.0370131549744803*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((-0.5*ny(i,2) - (sp.cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sp.sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*(sp.sin(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + theta(i,2)*(29.97411840050242 - 3.037013154974481*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(29.974118400502412 - 3.0370131549744803*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)

def Cfz(i,n,m):
    if (n==1 and m==1):
        return 0.5*(1 - nx(i,1)**2 - ny(i,1)**2 + nz(i,1)**2) + ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*((44.06035162512098*sp.sin(2*theta(i,1)))/theta(i,1) + (-1060.793564231141 + 107.4808595280735*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(71.65390635204905*theta(i,1) - 29.04023441674732*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)))/2.
    elif (n==1 and m==2):
        return ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*(-23.541617541212254 + 9.541058216523389*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(35.31242631181839*theta(i,1) - 3.577896831196271*theta(i,1)**3)))/(2.*(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==3):
        return ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*((-68.42444032663414*sp.sin(2*theta(i,1)))/theta(i,1) + (1666.290634181943 - 168.83053934745612*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-112.55369289830418*theta(i,1) + 45.61629355108943*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)))/2.
    elif (n==1 and m==4):
        return ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*((-29.040234416747317*sp.sin(2*theta(i,1)))/theta(i,1) + (707.1957094874274 - 71.65390635204899*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-50.236672001638375*theta(i,1) + 20.360156277831546*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)))/2.
    elif (n==1 and m==5):
        return ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*(27.73431477040989 - 11.240294400188406*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-29.974118400502412*theta(i,1) + 3.0370131549744803*theta(i,1)**3)))/(2.*(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==2 and m==1):
        return 1.*(0.25 + sp.cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2)/4. - nx(i,2)**2/4. - (sp.cos(2*theta(i,1))*nx(i,2)**2)/4. + (nx(i,1)**2*nx(i,2)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2)/4. + (nx(i,2)**2*ny(i,1)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. - ny(i,2)**2/4. - (sp.cos(2*theta(i,1))*ny(i,2)**2)/4. + (nx(i,1)**2*ny(i,2)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. + (ny(i,1)**2*ny(i,2)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. - (nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. - (nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (ny(i,1)*ny(i,2)**2*nz(i,1))/2. + (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (sp.cos(2*theta(i,1))*nz(i,1)**2)/4. - (nx(i,2)**2*nz(i,1)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. - (ny(i,2)**2*nz(i,1)**2)/4. + (sp.cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. + nz(i,2)**2/4. + (sp.cos(2*theta(i,1))*nz(i,2)**2)/4. - (nx(i,1)**2*nz(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. - (ny(i,1)**2*nz(i,2)**2)/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. + (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (nz(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*sp.sin(2*theta(i,1)))/2. + (nx(i,1)*nx(i,2)**2*sp.sin(2*theta(i,1)))/2. + (ny(i,1)*sp.sin(2*theta(i,1)))/2. - (nx(i,2)**2*ny(i,1)*sp.sin(2*theta(i,1)))/2. + (nx(i,1)*ny(i,2)**2*sp.sin(2*theta(i,1)))/2. - (ny(i,1)*ny(i,2)**2*sp.sin(2*theta(i,1)))/2. - (nx(i,1)*nz(i,2)**2*sp.sin(2*theta(i,1)))/2. + (ny(i,1)*nz(i,2)**2*sp.sin(2*theta(i,1)))/2.) + (0.25 + sp.cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (sp.cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (sp.cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (sp.cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*((44.06035162512098*sp.sin(2*theta(i,2)))/theta(i,2) + (-1060.793564231141 + 107.4808595280735*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(71.65390635204905*theta(i,2) - 29.04023441674732*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*(sp.cos(2*theta(i,1))*nx(i,1)) - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1))/2. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*((88.12070325024196*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + theta(i,2)*(71.65390635204898 - 29.040234416747314*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-71.65390635204905 + 29.04023441674732*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==2):
        return ((0.25 + sp.cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (sp.cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (sp.cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (sp.cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*(-23.541617541212254 + 9.541058216523389*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(35.31242631181839*theta(i,2) - 3.577896831196271*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((-0.5*(sp.cos(2*theta(i,1))*nx(i,1)) - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1))/2. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*(sp.sin(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + theta(i,2)*(-35.31242631181839 + 3.5778968311962718*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-35.31242631181839 + 3.577896831196271*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    elif (n==2 and m==3):
        return (0.25 + sp.cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (sp.cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (sp.cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (sp.cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*((-68.42444032663414*sp.sin(2*theta(i,2)))/theta(i,2) + (1666.290634181943 - 168.83053934745612*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-112.55369289830418*theta(i,2) + 45.61629355108943*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*(sp.cos(2*theta(i,1))*nx(i,1)) - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1))/2. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*((-136.84888065326828*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + theta(i,2)*(-112.55369289830406 + 45.61629355108942*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(112.55369289830418 - 45.61629355108943*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==3):
        return (0.25 + sp.cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (sp.cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (sp.cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (sp.cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*((-29.040234416747317*sp.sin(2*theta(i,2)))/theta(i,2) + (707.1957094874274 - 71.65390635204899*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-50.236672001638375*theta(i,2) + 20.360156277831546*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*(sp.cos(2*theta(i,1))*nx(i,1)) - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1))/2. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*((-58.080468833494635*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + theta(i,2)*(-50.23667200163832 + 20.360156277831535*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(50.236672001638375 - 20.360156277831546*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    else :
        return ((0.25 + sp.cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (sp.cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (sp.cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (sp.cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (sp.cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (sp.cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*(27.73431477040989 - 11.240294400188406*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-29.974118400502412*theta(i,2) + 3.0370131549744803*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((-0.5*(sp.cos(2*theta(i,1))*nx(i,1)) - (sp.cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1))/2. + (sp.cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (sp.cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (sp.cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*(sp.sin(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + theta(i,2)*(29.97411840050242 - 3.037013154974481*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(29.974118400502412 - 3.0370131549744803*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)

def Cfy(i,n,m):
    if (n==1 and m==1):
        return 1.*ny(i,1)*nz(i,1) - ny(i,1)*nz(i,1)*((44.06035162512098*sp.sin(2*theta(i,1)))/theta(i,1)+ (-1060.793564231141 + 107.4808595280735*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(71.65390635204905*theta(i,1) - 29.04023441674732*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) + nx(i,1)*((88.12070325024196*sp.sin(theta(i,1))**2)/theta(i,1) + (sp.sin(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + theta(i,1)*(71.65390635204898 - 29.040234416747314*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-71.65390635204905 + 29.04023441674732*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==2):
        return -((ny(i,1)*nz(i,1)*(-23.541617541212254 + 9.541058216523389*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(35.31242631181839*theta(i,1) - 3.577896831196271*theta(i,1)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) + (nx(i,1)*(sp.sin(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + theta(i,1)*(-35.31242631181839 + 3.5778968311962718*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-35.31242631181839 + 3.577896831196271*theta(i,1)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)
    elif (n==1 and m==3):
        return -(ny(i,1)*nz(i,1)*((-68.42444032663414*sp.sin(2*theta(i,1)))/theta(i,1) + (1666.290634181943 - 168.83053934745612*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-112.55369289830418*theta(i,1) + 45.61629355108943*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))) + nx(i,1)*((-136.84888065326828*sp.sin(theta(i,1))**2)/theta(i,1) + (sp.sin(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + theta(i,1)*(-112.55369289830406 + 45.61629355108942*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(112.55369289830418 - 45.61629355108943*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==4):
        return -(ny(i,1)*nz(i,1)*((-29.040234416747317*sp.sin(2*theta(i,1)))/theta(i,1) + (707.1957094874274 - 71.65390635204899*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-50.236672001638375*theta(i,1) + 20.360156277831546*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))) + nx(i,1)*((-58.080468833494635*sp.sin(theta(i,1))**2)/theta(i,1) + (sp.sin(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + theta(i,1)*(-50.23667200163832 + 20.360156277831535*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(50.236672001638375 - 20.360156277831546*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==5):
        return -((ny(i,1)*nz(i,1)*(27.73431477040989 - 11.240294400188406*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + sp.sin(2.*theta(i,1))*(-29.974118400502412*theta(i,1) + 3.0370131549744803*theta(i,1)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) + (nx(i,1)*(sp.sin(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + theta(i,1)*(29.97411840050242 - 3.037013154974481*theta(i,1)**2 + sp.cos(2.*theta(i,1))*(29.974118400502412 - 3.0370131549744803*theta(i,1)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)
    elif (n==2 and m==1):
        return 1.*((ny(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. - (nx(i,1)**2*ny(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. + nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + (ny(i,1)**2*ny(i,2)*nz(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. + ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) - sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) - (ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,1)*ny(i,2)*nz(i,2)*sp.sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*nz(i,2)*sp.sin(2*theta(i,1))) + (-0.5*(ny(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sp.sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sp.sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sp.sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sp.sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sp.sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sp.sin(2*theta(i,1)))/2.)*((44.06035162512098*sp.sin(2*theta(i,2)))/theta(i,2) + (-1060.793564231141 + 107.4808595280735*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(71.65390635204905*theta(i,2) - 29.04023441674732*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*((88.12070325024196*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + theta(i,2)*(71.65390635204898 - 29.040234416747314*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-71.65390635204905 + 29.04023441674732*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==2):
        return ((-0.5*(ny(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sp.sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sp.sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sp.sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sp.sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sp.sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sp.sin(2*theta(i,1)))/2.)*(-23.541617541212254 + 9.541058216523389*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(35.31242631181839*theta(i,2) - 3.577896831196271*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*(sp.sin(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + theta(i,2)*(-35.31242631181839 + 3.5778968311962718*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-35.31242631181839 + 3.577896831196271*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    elif (n==2 and m==3):
        return (-0.5*(ny(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sp.sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sp.sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sp.sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sp.sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sp.sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sp.sin(2*theta(i,1)))/2.)*((-68.42444032663414*sp.sin(2*theta(i,2)))/theta(i,2) + (1666.290634181943 - 168.83053934745612*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-112.55369289830418*theta(i,2) + 45.61629355108943*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*((-136.84888065326828*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + theta(i,2)*(-112.55369289830406 + 45.61629355108942*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(112.55369289830418 - 45.61629355108943*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==3):
        return (-0.5*(ny(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sp.sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sp.sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sp.sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sp.sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sp.sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sp.sin(2*theta(i,1)))/2.)*((-29.040234416747317*sp.sin(2*theta(i,2)))/theta(i,2) + (707.1957094874274 - 71.65390635204899*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-50.236672001638375*theta(i,2) + 20.360156277831546*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*((-58.080468833494635*sp.sin(theta(i,2))**2)/theta(i,2) + (sp.sin(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + theta(i,2)*(-50.23667200163832 + 20.360156277831535*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(50.236672001638375 - 20.360156277831546*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    else :
        return ((-0.5*(ny(i,2)*nz(i,2)) - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (sp.cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (sp.cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sp.sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sp.sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sp.sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sp.sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sp.sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sp.sin(2*theta(i,1)))/2.)*(27.73431477040989 - 11.240294400188406*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + sp.sin(2.*theta(i,2))*(-29.974118400502412*theta(i,2) + 3.0370131549744803*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - sp.cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + sp.cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sp.sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sp.sin(2*theta(i,1)))*(sp.sin(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + theta(i,2)*(29.97411840050242 - 3.037013154974481*theta(i,2)**2 + sp.cos(2.*theta(i,2))*(29.974118400502412 - 3.0370131549744803*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)

def D1_00(i):
    tmp =-2 *1j* Cfz(i,1, 1) * S1_vec_func(1, 1) - 2 *1j* Cfz(i,1, 2) * S1_vec_func(1, 2)-2 *1j* Cfz(i,1, 3) * S1_vec_func(1, 3) - 2 *1j* Cfz(i,1, 4) * S1_vec_func(1, 4) -\
        2 *1j* Cfz(i,1, 5) * S1_vec_func(1, 5) - 2 *1j* Cfz(i,2, 1) * S1_vec_func(2, 1) - \
            2 *1j* Cfz(i,2, 2) * S1_vec_func(2, 2) - 2 *1j* Cfz(i,2, 3) * S1_vec_func(2, 3)-\
                2 *1j* Cfz(i,2, 4) * S1_vec_func(2, 4) - 2 *1j* Cfz(i,2, 5) * S1_vec_func(2, 5)
    return tmp

def D1_01(i):
	tmp = -2 * Cfy(i,1, 1) * S1_vec_func(1, 1) - 2 * Cfy(i,1, 2) * S1_vec_func(1, 2) \
	- 2 * Cfy(i,1, 3) * S1_vec_func(1, 3) - 2 * Cfy(i,1, 4) * S1_vec_func(1, 4) -\
	 2 * Cfy(i,1, 5) * S1_vec_func(1, 5) - 2 * Cfy(i,2, 1) * S1_vec_func(2, 1) - \
	 2 * Cfy(i,2, 2) * S1_vec_func(2, 2) - 2 * Cfy(i,2, 3) * S1_vec_func(2, 3) - \
	 2 * Cfy(i,2, 4) * S1_vec_func(2, 4) - 2 * Cfy(i,2, 5) * S1_vec_func(2, 5)
	return tmp

def D1_10(i):
    tmp =2 * Cfy(i,1, 1) * S1_vec_func(1, 1) + \
        2 * Cfy(i,1, 2) * S1_vec_func(1, 2) + 2 * Cfy(i,1, 3) * S1_vec_func(1, 3) + \
        2 * Cfy(i,1, 4) * S1_vec_func(1, 4) + 2 * Cfy(i,1, 5) * S1_vec_func(1, 5) + \
        2 * Cfy(i,2, 1) * S1_vec_func(2, 1) + 2 * Cfy(i,2, 2) * S1_vec_func(2, 2) + \
        2 * Cfy(i,2, 3) * S1_vec_func(2, 3) + 2 * Cfy(i,2, 4) * S1_vec_func(2, 4) + \
        2 * Cfy(i,2, 5) * S1_vec_func(2, 5)
    return tmp

def D1_11(i):
    return 2 *1j* Cfz(i,1, 1) * S1_vec_func(1, 1) + 2 *1j* Cfz(i,1, 2) * S1_vec_func(1, 2) + \
        2 *1j* Cfz(i,1, 3) * S1_vec_func(1, 3) + 2 *1j* Cfz(i,1, 4) * S1_vec_func(1, 4) + \
        2 *1j* Cfz(i,1, 5) * S1_vec_func(1, 5) + 2 *1j* Cfz(i,2, 1) * S1_vec_func(2, 1) + \
        2 *1j* Cfz(i,2, 2) * S1_vec_func(2, 2) + 2 *1j* Cfz(i,2, 3) * S1_vec_func(2, 3) + \
        2 *1j* Cfz(i,2, 4) * S1_vec_func(2, 4) + \
        2 *1j* Cfz(i,2, 5) * S1_vec_func(2, 5)

def D2_00(i):
    """
    i is which control
    """
    tmp = np.sum([4 * (1j * Cfx(i,1, m2) * Cfy(i,1, m1) - Cfy(i,1, m1) * Cfy(i,1, m2) - Cfz(i,1, m1) * Cfz(i,1, m2)) * S2_vec_func(1, 1 , m1, m2) + 
   4 * (1j * Cfx(i,1, m2) * Cfy(i,2, m1) - Cfy(i,1, m2) * Cfy(i,2, m1) - Cfz(i,1, m2) * Cfz(i,2, m1)) * S2_vec_func(2, 1 , m1, m2) + 
   4 * (1j * Cfx(i,2, m2) * Cfy(i,2, m1) - Cfy(i,2, m1) * Cfy(i,2, m2) - Cfz(i,2, m1) * Cfz(i,2, m2)) * S2_vec_func(2, 2 , m1, m2) for m1 in range(1,6) for m2 in range(1,6)],axis =0)
    return tmp

def D2_01(i):
    tmp =np.sum([-4 * Cfx(i,1,  m2) * Cfz(i,1, m1) * S2_vec_func(1, 1 , m1, m2) - 4 * Cfx(i,1, m2) * Cfz(i,2, m1) * S2_vec_func(2, 1 , m1, m2) - 
      4 * Cfx(i,2, m2) * Cfz(i,2, m1) * S2_vec_func(2, 2 , m1, m2) for m1 in range(1,6) for m2 in range(1,6)],axis =0) 
    return tmp 

def D2_10(i):
    tmp =np.sum([4 * Cfx(i,1,  m2) * Cfz(i,1, m1) * S2_vec_func(1, 1 , m1, m2) +4 * Cfx(i,1, m2) * Cfz(i,2, m1) * S2_vec_func(2, 1 , m1, m2) + 
         4 * Cfx(i,2, m2) * Cfz(i,2, m1) * S2_vec_func(2, 2 , m1, m2) for  m1 in range(1,6) for m2 in range(1,6)],axis =0)
    return tmp

def D2_11(i):
    tmp = np.sum([4 * (-1j * Cfx(i,1, m2) * Cfy(i,1, m1) - Cfy(i,1, m1) * Cfy(i,1, m2) - Cfz(i,1, m1) * Cfz(i,1, m2)) * S2_vec_func(1, 1 , m1, m2) + 
   4 * (-1j * Cfx(i,1, m2) * Cfy(i,2, m1) - Cfy(i,1, m2) * Cfy(i,2, m1) - Cfz(i,1, m2) * Cfz(i,2, m1)) * S2_vec_func(2, 1 , m1, m2) + 
   4 * (-1j * Cfx(i,2, m2) * Cfy(i,2, m1) - Cfy(i,2, m1) * Cfy(i,2, m2) - Cfz(i,2, m1) * Cfz(i,2, m2)) * S2_vec_func(2, 2 , m1, m2) for  m1 in range(1,6) for m2 in range(1,6)],axis =0)
    return tmp

def D3_00(i):
    """
    i is which control
    """
    return np.sum([8 * 1j * ((Cfy(i,1, m1) * Cfy(i,1, m2) + Cfz(i,1, m1) * Cfz(i,1, m2)) * Cfz(i,1, m3) + 
      Cfx(i,1, m2) * (Cfx(i,1, m3) * Cfz(i,1, m1) +  1j * (Cfy(i,1, m3) * Cfz(i,1, m1) - 
            Cfy(i,1, m1) * Cfz(i,1, m3))))* S3_vec_func(1, 1, 1, m1, m2, m3) + 8 * (Cfx(i,1, m2) * (Cfy(i,2, m1) * Cfz(i,1, m3) + 
         1j * (Cfx(i,1, m3) + 1j * Cfy(i,1, m3)) * Cfz(i,2, m1)) + 1j * Cfz(i,1, m3) * (Cfy(i,1, m2) * Cfy(i,2, m1) + 
         Cfz(i,1, m2) * Cfz(i,2, m1)))* S3_vec_func(2, 1, 1, m1, m2, m3) +  8 * (Cfx(i,2, m2) * (Cfy(i,2, m1) * Cfz(i,1, m3) + 
         1j * (Cfx(i,1, m3) + 1j * Cfy(i,1, m3)) * Cfz(i,2, m1)) + 1j * Cfz(i,1, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + 
         Cfz(i,2, m1) * Cfz(i,2, m2)))* S3_vec_func(2, 2, 1, m1, m2, m3) +  8 * 1j * ((Cfy(i,2, m1) * Cfy(i,2, m2) + 
         Cfz(i,2, m1) * Cfz(i,2, m2)) * Cfz(i,2, m3) +  Cfx(i,2, m2) * (Cfx(i,2, m3) * Cfz(i,2, m1) + 
         1j * (Cfy(i,2, m3) * Cfz(i,2, m1) -  Cfy(i,2, m1) * Cfz(i,2, m3))))* S3_vec_func(2, 2, 2,m1, m2, m3) 
         for m1 in range(1,6) for m2 in range(1,6) for m3 in range(1,6)],axis =0)

def D3_01(i):
    return np.sum([8 * (Cfx(i,1, m2) * Cfx(i,1, m3) * Cfy(i,1, m1) +  Cfy(i,1, m3) * (Cfy(i,1, m1) * Cfy(i,1, m2) + 
         Cfz(i,1, m1) * Cfz(i,1, m2)))* S3_vec_func(1, 1, 1, m1, m2, m3) +  8 * (Cfx(i,1, m2) * Cfx(i,1, m3) * Cfy(i,2, m1) + 
         Cfy(i,1, m3) * (Cfy(i,1, m2) * Cfy(i,2, m1) +  Cfz(i,1, m2) * Cfz(i,2, m1)))* S3_vec_func(2, 1, 1, m1, m2, m3) + 
        8 * (Cfx(i,1, m3) * Cfx(i,2, m2) * Cfy(i,2, m1) + Cfy(i,1, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + 
         Cfz(i,2, m1) * Cfz(i,2, m2)))* S3_vec_func(2, 2, 1, m1, m2, m3) + 8 * (Cfx(i,2, m2) * Cfx(i,2, m3) * Cfy(i,2, m1) + 
         Cfy(i,2, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + Cfz(i,2, m1) * Cfz(i,2, m2)))* S3_vec_func(2, 2, 2,m1, m2, m3) 
         for m1 in range(1,6) for m2 in range(1,6) for m3 in range(1,6) ],axis =0) 

def D3_10(i):
    return np.sum([-8 * (Cfx(i,1, m2) * Cfx(i,1, m3) * Cfy(i,1, m1) +  Cfy(i,1, m3) * (Cfy(i,1, m1) * Cfy(i,1, m2) + 
              Cfz(i,1, m1) * Cfz(i,1, m2)))* S3_vec_func(1, 1, 1, m1, m2, m3) -  8 * (Cfx(i,1, m2) * Cfx(i,1, m3) * Cfy(i,2, m1) + 
               Cfy(i,1, m3) * (Cfy(i,1, m2) * Cfy(i,2, m1) + Cfz(i,1, m2) * Cfz(i,2, m1)))* S3_vec_func(2, 1, 1, m1, m2, m3) - 
               8 * (Cfx(i,1, m3) * Cfx(i,2, m2) * Cfy(i,2, m1) +  Cfy(i,1, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + 
                Cfz(i,2, m1) * Cfz(i,2, m2)))* S3_vec_func(2, 2, 1, m1, m2, m3) -  8 * (Cfx(i,2, m2) * Cfx(i,2, m3) * Cfy(i,2, m1) + 
                  Cfy(i,2, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) +  Cfz(i,2, m1) * Cfz(i,2, m2)))* S3_vec_func(2, 2, 2,m1, m2, m3) 
                for m1 in range(1,6) for m2 in range(1,6) for m3 in range(1,6)],axis =0)

def D3_11(i):
    return np.sum([-8 * 1j * ((Cfy(i,1, m1) * Cfy(i,1, m2) +  Cfz(i,1, m1) * Cfz(i,1, m2)) * Cfz(i,1, m3) + 
          Cfx(i,1, m2) * (Cfx(i,1, m3) * Cfz(i,1, m1) - 1j * Cfy(i,1, m3) * Cfz(i,1, m1) + 
             1j * Cfy(i,1, m1) * Cfz(i,1, m3)))* S3_vec_func(1, 1, 1, m1, m2, m3) + 8 * (Cfx(i,1, m2) * (Cfy(i,2, m1) * Cfz(i,1, m3) + (-1j * Cfx(i,1, m3) - 
                Cfy(i,1, m3)) * Cfz(i,2, m1)) -  1j * Cfz(i,1, m3) * (Cfy(i,1, m2) * Cfy(i,2, m1) + 
             Cfz(i,1, m2) * Cfz(i,2, m1)))* S3_vec_func(2, 1, 1, m1, m2, m3) +  8 * (Cfx(i,2, m2) * (Cfy(i,2, m1) * Cfz(i,1, m3) + (-1j * Cfx(i,1, m3) - 
                Cfy(i,1, m3)) * Cfz(i,2, m1)) - 1j * Cfz(i,1, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + 
             Cfz(i,2, m1) * Cfz(i,2, m2)))* S3_vec_func(2, 2, 1, m1, m2, m3) - 8 * 1j * ((Cfy(i,2, m1) * Cfy(i,2, m2) + 
             Cfz(i,2, m1) * Cfz(i,2, m2)) * Cfz(i,2, m3) + Cfx(i,2, m2) * (Cfx(i,2, m3) * Cfz(i,2, m1) - 
             1j * Cfy(i,2, m3) * Cfz(i,2, m1) + 1j * Cfy(i,2, m1) * Cfz(i,2, m3)))* S3_vec_func(2, 2, 2,m1, m2, m3) 
             for m1 in range(1,6) for m2 in range(1,6) for m3 in range(1,6)],axis =0)

def D4_00(i):
    return np.sum([16* ((Cfy(i,1, m1) * Cfy(i,1, m2) + Cfz(i,1, m1) * Cfz(i,1, m2)) * (-1j* Cfx(i,1, m4) * Cfy(i,1, m3) + 
         Cfy(i,1, m3) * Cfy(i,1, m4) + Cfz(i,1, m3) * Cfz(i,1, m4)) + 
      Cfx(i,1, m2) * (1j* (Cfy(i,1, m3) * Cfz(i,1, m1) - Cfy(i,1, m1) * Cfz(i,1, m3)) * Cfz(i,1, m4) + 
         Cfx(i,1, m3) * (-1j* Cfx(i,1, m4) * Cfy(i,1, m1) + Cfy(i,1, m1) * Cfy(i,1, m4) + 
            Cfz(i,1, m1) * Cfz(i,1, m4)))) * S4_vec_func(1, 1, 1, 1, m1, m2, m3, m4) + 
   16* ((-1j* Cfx(i,1, m4) * Cfy(i,1, m3) + Cfy(i,1, m3) * Cfy(i,1, m4) + Cfz(i,1, m3) * Cfz(i,1, m4)) * (Cfy(i,1, m2) * Cfy(i,2, m1) 
   +Cfz(i,1, m2) * Cfz(i,2, m1)) +  Cfx(i,1,m2) * (-1j* Cfz(i,1, m4) * (Cfy(i,2, m1) * Cfz(i,1, m3) - 
            Cfy(i,1, m3) * Cfz(i,2, m1)) + Cfx(i,1, m3) * (-1j* Cfx(i,1, m4) * Cfy(i,2, m1) + 
            Cfy(i,1, m4) * Cfy(i,2, m1) + Cfz(i,1, m4) * Cfz(i,2, m1)))) * S4_vec_func(2, 1, 1, 1, m1, m2, m3, m4) + 
   16* (Cfy(i,1, m3) * Cfy(i,1, m4) * Cfy(i,2, m1) * Cfy(i,2, m2) - 1j* Cfx(i,2, m2) * Cfy(i,2, m1) * Cfz(i,1, m3) * Cfz(i,1, m4) + 
      Cfy(i,2, m1) * Cfy(i,2, m2) * Cfz(i,1, m3) * Cfz(i,1, m4) +  1j* Cfx(i,2, m2) * Cfy(i,1, m3) * Cfz(i,1, m4) * Cfz(i,2, m1) + 
      Cfx(i,1, m3) * Cfx(i,2, m2) * (-1j* Cfx(i,1, m4) * Cfy(i,2, m1) + Cfy(i,1, m4) * Cfy(i,2, m1) + Cfz(i,1, m4) * Cfz(i,2, m1)) + 
      Cfy(i,1, m3) * Cfy(i,1, m4) * Cfz(i,2, m1) * Cfz(i,2, m2) +  Cfz(i,1, m3) * Cfz(i,1, m4) * Cfz(i,2, m1) * Cfz(i,2, m2) - 
      1j* Cfx(i,1, m4) * Cfy(i,1, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + Cfz(i,2, m1) * Cfz(i,2, m2))) * S4_vec_func(2, 2, 1, 1, m1, m2, m3, m4) + 
   16* (-1j* Cfx(i,1, m4) * (Cfx(i,2, m2) * Cfx(i,2, m3) * Cfy(i,2, m1) + Cfy(i,2, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + 
            Cfz(i,2, m1) * Cfz(i,2, m2))) + (Cfy(i,2, m1) * Cfy(i,2, m2) + Cfz(i,2, m1) * Cfz(i,2, m2)) * (Cfy(i,1, m4) * Cfy(i,2, m3) + 
         Cfz(i,1, m4) * Cfz(i,2, m3)) +  Cfx(i,2, m2) * (Cfx(i,2, m3) * (Cfy(i,1, m4) * Cfy(i,2, m1) + 
            Cfz(i,1, m4) * Cfz(i,2, m1)) +  1j* Cfz(i,1, m4) * (Cfy(i,2, m3) * Cfz(i,2, m1) - 
            Cfy(i,2, m1) * Cfz(i,2, m3)))) * S4_vec_func(2, 2, 2, 1, m1, m2, m3, m4) + 
   16* ((Cfy(i,2, m1) * Cfy(i,2, m2) +  Cfz(i,2, m1) * Cfz(i,2, m2)) * (-1j* Cfx(i,2, m4) * Cfy(i,2, m3) + 
         Cfy(i,2, m3) * Cfy(i,2, m4) + Cfz(i,2, m3) * Cfz(i,2, m4)) + Cfx(i,2,  m2) * (1j* (Cfy(i,2, m3) * Cfz(i,2, m1) - 
            Cfy(i,2, m1) * Cfz(i,2, m3)) * Cfz(i,2, m4) + Cfx(i,2, m3) * (-1j* Cfx(i,2, m4) * Cfy(i,2, m1) + 
            Cfy(i,2, m1) * Cfy(i,2, m4) + Cfz(i,2, m1) * Cfz(i,2, m4)))) * S4_vec_func(2, 2, 2, 2, m1, m2, m3, m4) 
   for m1 in range(1,6) for m2 in range(1,6) for m3 in range(1,6) for m4 in range(1,6)],axis=0)

def D4_01(i):
    return np.sum([16* (Cfx(i,1, m4) * (Cfy(i,1, m1) * Cfy(i,1, m2) + Cfz(i,1, m1) * Cfz(i,1, m2)) * Cfz(i,1, m3) + 
          Cfx(i,1, m2) * (Cfx(i,1, m3) * Cfx(i,1, m4) * Cfz(i,1, m1) +  Cfy(i,1, m4) * (Cfy(i,1, m3) * Cfz(i,1, m1) - 
                Cfy(i,1, m1) * Cfz(i,1, m3)))) * S4_vec_func(1, 1, 1, 1, m1, m2, m3, m4) + 
       16* (Cfx(i,1, m2) * Cfy(i,1, m4) * (-Cfy(i,2, m1) * Cfz(i,1, m3) +  Cfy(i,1, m3) * Cfz(i,2, m1)) + 
          Cfx(i,1, m4) * (Cfy(i,1, m2) * Cfy(i,2, m1) * Cfz(i,1, m3) + (Cfx(i,1, m2) * Cfx(i,1, m3) + 
                Cfz(i,1, m2) * Cfz(i,1, m3)) * Cfz(i,2, m1))) * S4_vec_func(2, 1, 1, 
          1, m1, m2, m3, m4) + 16* (Cfx(i,2, m2) * (Cfx(i,1, m3) * Cfx(i,1, m4) * Cfz(i,2, m1) + 
             Cfy(i,1, m4) * (-Cfy(i,2, m1) * Cfz(i,1, m3) + Cfy(i,1, m3) * Cfz(i,2, m1))) + 
          Cfx(i,1, m4) * Cfz(i,1, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) +  Cfz(i,2, m1) * Cfz(i,2, m2))) * S4_vec_func(2, 2, 1, 1, m1, m2, m3, m4) 
          + 16* (Cfx(i,2, m2) * Cfy(i,1, m4) * (Cfy(i,2, m3) * Cfz(i,2, m1) -  Cfy(i,2, m1) * Cfz(i,2, m3)) + 
          Cfx(i,1, m4) * (Cfx(i,2, m2) * Cfx(i,2, m3) * Cfz(i,2, m1) + (Cfy(i,2, m1) * Cfy(i,2, m2) + 
                Cfz(i,2, m1) * Cfz(i,2, m2)) * Cfz(i,2, m3))) * S4_vec_func(2, 2, 2, 1, m1, m2, m3, m4) + 16* (Cfx(i,2, m4) * (Cfy(i,2, m1) * Cfy(i,2, m2) + 
             Cfz(i,2, m1) * Cfz(i,2, m2)) * Cfz(i,2, m3) +  Cfx(i,2, m2) * (Cfx(i,2, m3) * Cfx(i,2, m4) * Cfz(i,2, m1) + 
             Cfy(i,2, m4) * (Cfy(i,2, m3) * Cfz(i,2, m1) -  Cfy(i,2, m1) * Cfz(i,2, m3)))) * S4_vec_func(2, 2, 2, 2, m1, m2, m3, m4)
               for m1 in range(1,6) for m2 in range(1,6) for m3 in range(1,6) for m4 in range(1,6)],axis=0)

def D4_10(i):
    return np.sum([-16* (Cfx(i,1, m4) * (Cfy(i,1, m1) * Cfy(i,1, m2) + Cfz(i,1, m1) * Cfz(i,1, m2)) * Cfz(i,1, m3) + 
            Cfx(i,1, m2) * (Cfx(i,1, m3) * Cfx(i,1, m4) * Cfz(i,1, m1) +  Cfy(i,1, m4) * (Cfy(i,1, m3) * Cfz(i,1, m1) - 
                  Cfy(i,1, m1) * Cfz(i,1, m3)))) * S4_vec_func(1, 1, 1, 1, m1, m2, m3, m4) - 
         16* (Cfx(i,1, m2) * Cfy(i,1, m4) * (-Cfy(i,2, m1) * Cfz(i,1, m3) +  Cfy(i,1, m3) * Cfz(i,2, m1)) + 
            Cfx(i,1, m4) * (Cfy(i,1, m2) * Cfy(i,2, m1) * Cfz(i,1, m3) + (Cfx(i,1, m2) * Cfx(i,1, m3) + 
                  Cfz(i,1, m2) * Cfz(i,1, m3)) * Cfz(i,2, m1))) * S4_vec_func(2, 1, 1,  1, m1, m2, m3, m4) + 
         16* (Cfx(i,2, m2) * (-Cfx(i,1, m3) * Cfx(i,1, m4) * Cfz(i,2, m1) +  Cfy(i,1, m4) * (Cfy(i,2, m1) * Cfz(i,1, m3) - 
                  Cfy(i,1, m3) * Cfz(i,2, m1))) -  Cfx(i,1, m4) * Cfz(i,1, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + 
               Cfz(i,2, m1) * Cfz(i,2, m2))) * S4_vec_func(2, 2, 1, 1, m1, m2, m3, m4) - 
         16* (Cfx(i,2, m2) * Cfy(i,1, m4) * (Cfy(i,2, m3) * Cfz(i,2, m1) -  Cfy(i,2, m1) * Cfz(i,2, m3)) + 
            Cfx(i,1, m4) * (Cfx(i,2, m2) * Cfx(i,2, m3) * Cfz(i,2, m1) + (Cfy(i,2, m1) * Cfy(i,2, m2) + 
                  Cfz(i,2, m1) * Cfz(i,2, m2)) * Cfz(i,2, m3))) * S4_vec_func(2, 2, 2, 1, m1, m2, m3, m4) - 
         16* (Cfx(i,2, m4) * (Cfy(i,2, m1) * Cfy(i,2, m2) + Cfz(i,2, m1) * Cfz(i,2, m2)) * Cfz(i,2, m3) + 
            Cfx(i,2, m2) * (Cfx(i,2, m3) * Cfx(i,2, m4) * Cfz(i,2, m1) + Cfy(i,2, m4) * (Cfy(i,2, m3) * Cfz(i,2, m1) - 
                  Cfy(i,2, m1) * Cfz(i,2, m3)))) * S4_vec_func(2, 2, 2, 2, m1, m2, m3, m4) 
                     for m1 in range(1,6) for m2 in range(1,6) for m3 in range(1,6) for m4 in range(1,6)],axis=0)

def D4_11(i):
    return np.sum([16* ((Cfy(i,1, m1) * Cfy(i,1, m2) +  Cfz(i,1, m1) * Cfz(i,1, m2)) * (1j* Cfx(i,1, m4) * Cfy(i,1, m3) + 
             Cfy(i,1, m3) * Cfy(i,1, m4) + Cfz(i,1, m3) * Cfz(i,1, m4)) + Cfx(i,1, 
             m2) * (-1j* (Cfy(i,1, m3) * Cfz(i,1, m1) - Cfy(i,1, m1) * Cfz(i,1, m3)) * Cfz(i,1, m4) + 
             Cfx(i,1, m3) * (1j* Cfx(i,1, m4) * Cfy(i,1, m1) + Cfy(i,1, m1) * Cfy(i,1, m4) + 
                Cfz(i,1, m1) * Cfz(i,1, m4)))) * S4_vec_func(1, 1, 1, 1, m1, m2, m3, m4) + 
       16* ((1j* Cfx(i,1, m4) * Cfy(i,1, m3) + Cfy(i,1, m3) * Cfy(i,1, m4) + Cfz(i,1, m3) * Cfz(i,1, m4)) * (Cfy(i,1, m2) * Cfy(i,2, m1) + 
             Cfz(i,1, m2) * Cfz(i,2, m1)) +  Cfx(i,1, m2) * (1j* Cfz(i,1, m4) * (Cfy(i,2, m1) * Cfz(i,1, m3) - 
                Cfy(i,1, m3) * Cfz(i,2, m1)) +  Cfx(i,1, m3) * (1j* Cfx(i,1, m4) * Cfy(i,2, m1) + 
                Cfy(i,1, m4) * Cfy(i,2, m1) +  Cfz(i,1, m4) * Cfz(i,2, m1)))) * S4_vec_func(2, 1, 1, 1, m1, m2, m3, m4) + 
       16* (Cfy(i,1, m3) * Cfy(i,1, m4) * Cfy(i,2, m1) * Cfy(i,2, m2) +  1j* Cfx(i,2, m2) * Cfy(i,2, m1) * Cfz(i,1, m3) * Cfz(i,1, m4) + 
          Cfy(i,2, m1) * Cfy(i,2, m2) * Cfz(i,1, m3) * Cfz(i,1, m4) - 1j* Cfx(i,2, m2) * Cfy(i,1, m3) * Cfz(i,1, m4) * Cfz(i,2, m1) + 
          Cfx(i,1, m3) * Cfx(i,2, m2) * (1j* Cfx(i,1, m4) * Cfy(i,2, m1) + Cfy(i,1, m4) * Cfy(i,2, m1) + Cfz(i,1, m4) * Cfz(i,2, m1)) + 
          Cfy(i,1, m3) * Cfy(i,1, m4) * Cfz(i,2, m1) * Cfz(i,2, m2) + Cfz(i,1, m3) * Cfz(i,1, m4) * Cfz(i,2, m1) * Cfz(i,2, m2) + 
          1j* Cfx(i,1, m4) * Cfy(i,1, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + Cfz(i,2, m1) * Cfz(i,2, m2))) * S4_vec_func(2, 2, 1, 1, m1, m2, m3,  m4) 
          + 16* (1j* Cfx(i,1, m4) * (Cfx(i,2, m2) * Cfx(i,2, m3) * Cfy(i,2, m1) +  Cfy(i,2, m3) * (Cfy(i,2, m1) * Cfy(i,2, m2) + 
                Cfz(i,2, m1) * Cfz(i,2, m2))) + (Cfy(i,2, m1) * Cfy(i,2, m2) + Cfz(i,2, m1) * Cfz(i,2, m2)) * (Cfy(i,1, m4) * Cfy(i,2, m3) + 
             Cfz(i,1, m4) * Cfz(i,2, m3)) + Cfx(i,2, m2) * (Cfx(i,2, m3) * (Cfy(i,1, m4) * Cfy(i,2, m1) + 
                Cfz(i,1, m4) * Cfz(i,2, m1)) -  1j* Cfz(i,1, m4) * (Cfy(i,2, m3) * Cfz(i,2, m1) - 
                Cfy(i,2, m1) * Cfz(i,2, m3)))) * S4_vec_func(2, 2, 2, 1, m1, m2,m3, m4) + 
       16* ((Cfy(i,2, m1) * Cfy(i,2, m2) +  Cfz(i,2, m1) * Cfz(i,2, m2)) * (1j* Cfx(i,2, m4) * Cfy(i,2, m3) + 
             Cfy(i,2, m3) * Cfy(i,2, m4) + Cfz(i,2, m3) * Cfz(i,2, m4)) + 
          Cfx(i,2, m2) * (-1j* (Cfy(i,2, m3) * Cfz(i,2, m1) - Cfy(i,2, m1) * Cfz(i,2, m3)) * Cfz(i,2, m4) + 
             Cfx(i,2, m3) * (1j* Cfx(i,2, m4) * Cfy(i,2, m1) + Cfy(i,2, m1) * Cfy(i,2, m4) + 
                Cfz(i,2, m1) * Cfz(i,2, m4)))) * S4_vec_func(2, 2, 2, 2, m1, m2, m3, m4) 
       for m1 in range(1,6) for m2 in range(1,6) for m3 in range(1,6) for m4 in range(1,6)],axis=0)    

def control_dynamics (i):
    """
    It is actually dyson expansion with control -"i"- specified
    specify the random control and simplify the Dyson into numerics
    return is the Dyson that filtering is numerical & S[n,m] is numerically vectors
    """
    tmp_00 = D1_00(i)+D2_00(i)#+D3_00(i)+D4_00(i) 
    tmp_01 = D1_01(i)+D2_01(i)#+D3_01(i)+D4_00(i) 
    tmp_10 = D1_10(i)+D2_10(i)#+D3_10(i)+D4_00(i) 
    tmp_11 = D1_11(i)+D2_11(i)#+D3_11(i)+D4_00(i) 
    return [[tmp_00, tmp_01] ,[tmp_10,tmp_11]]

def sigma_x_T (dyson, init):
    """
    make the expectation value of pauli matrix
    """
    expectation = np.array(dyson)[0,0]*(sigma_1 @ init)[0,0]\
    + np.array(dyson)[1,0]*(sigma_1 @ init)[0,1]\
    +np.array(dyson)[0,1]*(sigma_1 @ init)[1,0]\
    +np.array(dyson)[1,1]*(sigma_1 @ init)[1,1]  
    # = np.trace(np.array(dyson)@((sigma_1 @ init))) while dyson is vectorized 
    return expectation

# optimization k2

print('start making k2 Symbol expression',datetime.now().strftime("%H:%M:%S"))
"""
the expr_k2 is the negative fidelity (since need to minimize)
"""
expr_k2= -1/8* ( sigma_x_T(control_dynamics(None), sigma_1)+2 +2 )
print('end making k2 symbol expression',datetime.now().strftime("%H:%M:%S"))

def fidelity_k2(ctrl_params):
	"""
	Q-D Xu; make these variables global to avoid my unassigned issue
	"""
	#  pulse  parameters plugin// The params without _  are variables, cf symbols
	theta1, theta2 = ctrl_params[0], ctrl_params[1]
	nx1, ny1, nz1 = ctrl_params[2], ctrl_params[3], ctrl_params[4]
	nx2, ny2, nz2 = ctrl_params[5], ctrl_params[6], ctrl_params[7]
	return expr_k2.subs([(theta_1, theta1),(theta_2,theta2),\
					(nx_1, nx1),(nx_2, nx2),(ny_1,ny1),(ny_2,ny2),(nz_1,nz1),(nz_2,nz2)])

initial_guess = [0.2, 0.1, 1, 0, 0, 0, 0, 1]
cons =({'type': 'eq', 'fun': lambda x:  x[2]**2+ x[3]**2+ x[4]**2-1},
        {'type': 'eq', 'fun': lambda x:  x[5]**2+ x[6]**2+ x[7]**2-1})
bnds = ((0, 2*np.pi), (0, 2*np.pi), (-1,1),(-1,1),(-1,1), (-1,1),(-1,1),(-1,1))

Nfeval = 1

def callback_func(Xi):
    global Nfeval
    print('{0:4d} {1: 3.6f} {2: 3.6f}  {3: 3.6f}  {4: 3.6f}  {5: 3.6f}  {6: 3.6f}  {7: 3.6f} {8: 3.6f}  {9: 3.6f} '.format(Nfeval, Xi[0], Xi[1], Xi[2],Xi[3],Xi[4],Xi[5],Xi[6],Xi[7], fidelity_k2(Xi)))
    Nfeval += 1


print("Start k2 optimization: ",datetime.now().strftime("%H:%M:%S"))
print ('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s} {5:9s} {6:9s} {7:9s} {8:9s} {9:9s}'.format('Iter', ' theta_1', ' theta_2', ' nx_1', 'ny_1', 'nz_1','nx_2','ny_2', 'nz_2', 'F(C)'))
optimize_k2_sol = opt.minimize(fun = fidelity_k2, x0= initial_guess,args = (),method ='Nelder-Mead', 
                               constraints=cons, callback=callback_func)
print("End k2 optimization: ",datetime.now().strftime("%H:%M:%S"))
optimal_k2_theta_1 = optimize_k2_sol.x[0]
optimal_k2_theta_2 = optimize_k2_sol.x[1]
optimal_k2_nx_1 = optimize_k2_sol.x[2]
optimal_k2_ny_1 = optimize_k2_sol.x[3]
optimal_k2_nz_1 = optimize_k2_sol.x[4]
optimal_k2_nx_2 = optimize_k2_sol.x[5]
optimal_k2_ny_2 = optimize_k2_sol.x[6]
optimal_k2_nz_2 = optimize_k2_sol.x[7]


print('hello_test')
#print(theta(1,1))
#print(nx(1,2))
#print(Cfx(1,1,1))
#print(Cfx(1,2,2))
#print(Cfy(1,1,3))
#print(Cfy(1,1,4))
#print(Cfz(1,1,4))
#print(Cfz(1,1,5))
#print(D1_00(99))
#print(D2_01(99))
#print('control_dyanmcics:',datetime.now().strftime("%H:%M:%S"))
#control_dynamics(None)
#print('sigma_x_(T)',datetime.now().strftime("%H:%M:%S"))
#test=sigma_x_T(control_dynamics(None),sigma_1)
#print("end test ",datetime.now().strftime("%H:%M:%S"))


#print("Start k2 optimization: ",datetime.now().strftime("%H:%M:%S"))
#optimize_k2_sol = opt.minimize(fun = fidelity,x0= initial_guess,args = (), method ='Nelder-Mead')
#print("End k2 optimization: ",datetime.now().strftime("%H:%M:%S"))
#optimal_k2_theta_1 = optimize_k2_sol.x[0]
#optimal_k2_theta_2 = optimize_k2_sol.x[1]
#optimal_k2_nx_1 = optimize_k2_sol.x[2]
#optimal_k2_ny_1 = optimize_k2_sol.x[3]
#optimal_k2_nz_1 = optimize_k2_sol.x[4]
#optimal_k2_nx_2 = optimize_k4_sol.x[5]
#optimal_k2_ny_2 = optimize_k2_sol.x[6]
#optimal_k2_nz_2 = optimize_k2_sol.x[7]


