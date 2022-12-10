#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:42:18 2022

@author: wenzhengdong
L=2  Dyson 1~2

O_tilde = sigma_x

The Sf[n,m] is vectorized.
The Cfx Cfy Cfz are functions 

index 'i' starts from 0 !!!!!!!!!!!!!!!!
"""

import numpy as np
import scipy.integrate as integrate
from joblib import Parallel, delayed
from datetime import datetime
from operator import add
import pandas as pd
import sys
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
#sys.path.insert(1, '/Users/wenzhengdong/Dropbox (Dartmouth College)/Wenzheng/frame_QNS_python_simulation')
from subspace_basis import * 

cos = np.cos
sin = np.sin

ROUND_TOLERANCE = 5 # round small numbers: 10-digit decimal 

# Control & simplification
theta_list = [np.pi*1/6, np.pi*1/5,  np.pi*1/4, np.pi*2/3]
theta_list = [[i,j] for i in theta_list for j in theta_list]
n_list = [[0,0,1],[0,1,0],[0,0,1], [1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),0,1/np.sqrt(2)], [0,1/np.sqrt(2),1/np.sqrt(2)]]
F1_list = [[i,[j,k]] for i in theta_list for j in n_list for k in n_list]
print(len(F1_list))
F1_theta = [row[0] for row in F1_list]
F1_n = [row[1] for row in F1_list]

ddd= len(F1_list) 


sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1 = np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])


# functions that relevant to controls
# n is window, i is which control 
theta = lambda i,n : F1_list[i][0][0] if(n ==1 ) else ( F1_list[i][0][1] if (n==2) else print("bound error")) # n can ONLY take 1 or 2 ,two pulses
nx = lambda i,n : F1_list[i][1][0][0] if(n ==1 ) else ( F1_list[i][1][1][0] if (n==2) else print("bound error"))
ny = lambda i,n : F1_list[i][1][0][1] if(n ==1 ) else ( F1_list[i][1][1][1] if (n==2) else print("bound error"))
nz = lambda i,n : F1_list[i][1][0][2] if(n ==1 ) else ( F1_list[i][1][1][2] if (n==2) else print("bound error"))

def sigma_function(v):
    if (v==0):
        return sigma_0
    elif (v==1):
        return sigma_1
    elif (v==2):
        return sigma_2
    elif (v==3):
        return sigma_3    

def frame_dual(m,t):
    """
    dual frame: m is frame index // form see Mathematica
    """
    if (m==1):
        return 88.1207 - 58.0805* cos(6.28319 *t) - 136.849* sin(3.14159 *t)
    elif(m==2):
        return 7.15579 *cos(3.14159 *t) - 6.07403 *sin(6.28319* t)
    elif(m==3):
        return -136.849 + 91.2326* cos(6.28319 *t) + 214.962 *sin(3.14159 *t)
    elif(m==4):
        return -58.0805 + 40.7203 *cos(6.28319* t) + 91.2326 *sin(3.14159 *t)
    elif(m==5):
        return  7.15579 *(-0.848826 *cos(3.14159 *t) + sin(6.28319 *t))

def y(i,v,n,t):
    """
    filter function // v is pauli_dir, n is window // i is control 
    """
    if (n==1):
        # the previous full bin has time 1, WTLG
        return 1/2 *np.trace(\
            (cos(theta(i,1)*t)* sigma_0 +1j*sin(theta(i,1)*t)*(nx(i,1)*sigma_1+ny(i,1)*sigma_2+nz(i,1)*sigma_3)) @ \
            sigma_3 @ \
            (cos(theta(i,1)*t)* sigma_0 -1j*sin(theta(i,1)*t)*(nx(i,1)*sigma_1+ny(i,1)*sigma_2+nz(i,1)*sigma_3)) @ \
            sigma_function(v) )
    if (n==2):
        # the previous full bin has time 1, WTLG
        return  1/2* np.trace( \
            (cos(theta(i,1)*1)* sigma_0 +1j*sin(theta(i,1)*1)*(nx(i,1)*sigma_1+ny(i,1)*sigma_2+nz(i,1)*sigma_3)) @\
            (cos(theta(i,2)*t)* sigma_0 +1j*sin(theta(i,2)*t)*(nx(i,2)*sigma_1+ny(i,2)*sigma_2+nz(i,2)*sigma_3)) @ \
            sigma_3 @ \
            (cos(theta(i,2)*t)* sigma_0 -1j*sin(theta(i,2)*t)*(nx(i,2)*sigma_1+ny(i,2)*sigma_2+nz(i,2)*sigma_3)) @\
            (cos(theta(i,1)*1)* sigma_0 -1j*sin(theta(i,1)*1)*(nx(i,1)*sigma_1+ny(i,1)*sigma_2+nz(i,1)*sigma_3)) @ \
            sigma_function(v)  )   

def Cfx(i,n,m):
    """
    F1_v (n,m) = Integrate( y_v(t) * frame_dual(m) * W_n(t) )
    """
    integrand = lambda t: y(i,1,n,t)*frame_dual(m,t)
    return integrate.quad( integrand, 0,1)[0] # only return real since img is tooo small
def Cfy(i,n,m):
    integrand = lambda t: y(i,2,n,t)*frame_dual(m,t)
    return integrate.quad( integrand, 0,1)[0] # only return real since img is tooo small
def Cfz(i,n,m):
    integrand = lambda t: y(i,3,n,t)*frame_dual(m,t)
    return integrate.quad( integrand, 0,1)[0] # only return real since img is tooo small



# vecterized S[n,m]
size_basis = 10 # num_spcetra =85 D1+D2 
def position_s1_func(n,m):
    if (n==1):
        return m
    elif(n==2):
        return 1*5+m
S1_vec_func = lambda n,m: np.array([int(bool((i+1)==position_s1_func(n,m))) for i in range(size_basis)])

# Dyson vectorized
def D1_00(i):
    tmp =-2 *1j* Cfz(i,1, 1) * S1_vec_func(1, 1) - 2 *1j* Cfz(i,1, 2) * S1_vec_func(1, 2)\
        -2 *1j* Cfz(i,1, 3) * S1_vec_func(1, 3) - 2 *1j* Cfz(i,1, 4) * S1_vec_func(1, 4) \
        -2 *1j* Cfz(i,1, 5) * S1_vec_func(1, 5) - 2 *1j* Cfz(i,2, 1) * S1_vec_func(2, 1) \
        -2 *1j* Cfz(i,2, 2) * S1_vec_func(2, 2) - 2 *1j* Cfz(i,2, 3) * S1_vec_func(2, 3)\
        -2 *1j* Cfz(i,2, 4) * S1_vec_func(2, 4) - 2 *1j* Cfz(i,2, 5) * S1_vec_func(2, 5)
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


# obtian the control dynamics: sympy add all Dyson together
def control_dynamics (i):
    """
    It is actually dyson expansion with control -"i"- specified
    specify the random control and simplify the Dyson into numerics
    return is the Dyson that filtering is numerical & S[n,m] is numerically vectors
    """
    tmp_00 = D1_00(i) #+D2_00(i)   #+D3_00(i)
    tmp_01 = D1_01(i) #+D2_01(i)   #+D3_01(i)
    tmp_10 = D1_10(i) #+D2_10(i)   #+D3_10(i)
    tmp_11 = D1_11(i) #+D2_11(i)   #+D3_11(i)
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

def sigma_y_T (dyson, init):
    """
    make the expectation value of pauli matrix
    """
    expectation =np.array(dyson)[0,0]*(sigma_2 @ init)[0,0]\
    + np.array(dyson)[1,0]*(sigma_2 @ init)[0,1]\
    +np.array(dyson)[0,1]*(sigma_2 @ init)[1,0]\
    +np.array(dyson)[1,1]*(sigma_2 @ init)[1,1]    #np.trace(np.array(dyson)@((sigma_2 @ init)))
    return expectation

def sigma_z_T (dyson, init):
    """
    make the expectation value of pauli matrix
    """
    expectation =np.array(dyson)[0,0]*(sigma_3 @ init)[0,0]\
    + np.array(dyson)[1,0]*(sigma_3 @ init)[0,1]\
    +np.array(dyson)[0,1]*(sigma_3 @ init)[1,0]\
    +np.array(dyson)[1,1]*(sigma_3 @ init)[1,1]   #np.trace(np.array(dyson)@((sigma_3 @ init)))
    return expectation


############################################################
#    ONLY focus on <sigma_x(T)> where O-tilde = sigma_x      
############################################################

# list of dyson results
print("Start numerics: ",datetime.now().strftime("%H:%M:%S"))
results = Parallel(n_jobs=-1, verbose=10)(delayed(control_dynamics)(i) for i in range (ddd) )
print("End numerics: ",datetime.now().strftime("%H:%M:%S"))


# list of <Ox> for different rho_0
exp_x_0= np.array([ sigma_x_T(results[i],sigma_0) for i in range(ddd)])
exp_x_1= np.array([ sigma_x_T(results[i],sigma_1) for i in range(ddd)])
exp_x_2= np.array([ sigma_x_T(results[i],sigma_2) for i in range(ddd)])
exp_x_3= np.array([ sigma_x_T(results[i],sigma_3) for i in range(ddd)])


################################################################################################################################################
##### Inverse independent sub-matrix to find true C_qns !
################################################################################################################################################

# https://stackoverflow.com/questions/28816627/how-to-find-linearly-independent-rows-from-a-matrix
# https://math.stackexchange.com/questions/748500/how-to-find-linearly-independent-columns-in-a-matrix


matrix = np.vstack((exp_x_0,exp_x_1,exp_x_2,exp_x_3)) # rows are different ctrl & 4types of init, columns are S.

independent_rows = indep_rows(matrix)

# Find positions of independent_rows in matrix // for QNS_ctrl
QNS_index = [ matrix.tolist().index(ele.tolist()) for ele in independent_rows]
QNS_theta = [ F1_theta[np.mod(ele, ddd)]  for ele in QNS_index] # list of theta1 theta2 in QNS
QNS_n =  [ F1_n[np.mod(ele, ddd)]  for ele in QNS_index] # list of nx(y/z)1 and  nx(y/z)12 in QNS
QNS_rho = [ int( np.floor(ele/ddd) )  for ele in QNS_index]
QNS_matrix = [matrix[ele] for ele in QNS_index] # row= ctrl col = S
obs_to_S_matrix = np.linalg.pinv(np.array(QNS_matrix))

print('The theta1, theta2 in QNS are \n: ', QNS_theta)
print('The vec(n)1, vec(n)1 in QNS are \n: ', QNS_n)
print('The initial state/ Pauli in QNS are \n',QNS_rho)
print('The matrix from observable array to S arry is \n:'\
      ,obs_to_S_matrix)

#https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file
np.save('result_D1_qns_theta', QNS_theta)
np.save('result_D1_qns_n', QNS_n)
np.save('result_D1_qns_rho', QNS_rho)
np.save('result_D1_qns_Matrix', obs_to_S_matrix)


#print(np.linalg.matrix_rank(matrix))
#q,r = np.linalg.qr(matrix.T)
#q_round = np.real(q)
#r_round = np.round(np.real(r), ROUND_TOLERANCE)
#np.savetxt("r_round_test.csv", r_round, delimiter=",")



"""
simple_theta_list = 2*np.pi* np.array([ (i+1)/10  for i in range(10)])
F1_theta = 2*np.pi* np.array([[e1, e2]  for e1 in simple_theta_list for e2 in simple_theta_list])
theta_rnd = np.pi* np.array([[e1, e2]  for e1 in simple_theta_list for e2 in simple_theta_list])
phi_rand = 2*np.pi* np.array([[e1, e2]  for e1 in simple_theta_list for e2 in simple_theta_list])
F1_n = np.array([ [[cos(theta_rnd[i][j])*cos(phi_rand[i][j]), cos(theta_rnd[i][j])*sin(phi_rand[i][j]), \
       sin(theta_rnd[i][j])] for j in range(2)] for i in range(len(phi_rand))])
ddd = len(F1_theta)       
"""


