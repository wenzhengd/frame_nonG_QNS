#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:42:18 2022

@author: wenzhengdong
L=2  Dyson 1~2

O_tilde = sigma_x

The Sf[n,m] is vectorized.
The Cfx Cfy Cfz are functions 

"""

import numpy as np
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
ddd= 85 # # num_spcetra = 85
F1_theta =  2*np.pi* np.array([[np.random.random(), np.random.random()]  for i in range(ddd)])
theta_rnd = np.pi* np.array([[np.random.random(), np.random.random()]  for i in range(ddd)])
phi_rand = 2*np.pi* np.array([[np.random.random(), np.random.random()]  for i in range(ddd)])
F1_n = np.array([ [[cos(theta_rnd[i][j])*cos(phi_rand[i][j]), cos(theta_rnd[i][j])*sin(phi_rand[i][j]), \
      sin(theta_rnd[i][j])] for j in range(2)] for i in range(ddd)])


sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1 = np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])


# functions that relevant to controls
# n is window, i is which control 
theta = lambda i,n : F1_theta[i][0] if(n ==1 ) else ( F1_theta[i][1] if (n==2) else print("bound error")) # n can ONLY take 1 or 2 ,two pulses
nx = lambda i,n : F1_n[i][0][0] if(n ==1 ) else ( F1_n[i][1][0] if (n==2) else print("bound error"))
ny = lambda i,n : F1_n[i][0][1] if(n ==1 ) else ( F1_n[i][1][1] if (n==2) else print("bound error"))
nz = lambda i,n : F1_n[i][0][2] if(n ==1 ) else ( F1_n[i][1][2] if (n==2) else print("bound error"))

def Cfx(i,n,m):
    #n is window, m is frame, i is which control
    if(n==1 and m==1):
        return 1.*nx(i,1)*nz(i,1) - nx(i,1)*nz(i,1)*((44.06035162512098*sin(2*theta(i,1)))/theta(i,1) + (-1060.793564231141 + 107.4808595280735*theta(i,1)**2 + cos(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + sin(2.*theta(i,1))*(71.65390635204905*theta(i,1) - 29.04023441674732*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) - ny(i,1)*((88.12070325024196*sin(theta(i,1))**2)/theta(i,1) + (sin(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + theta(i,1)*(71.65390635204898 - 29.040234416747314*theta(i,1)**2 + cos(2.*theta(i,1))*(-71.65390635204905 + 29.04023441674732*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==2):
        return  -((nx(i,1)*nz(i,1)*(-23.541617541212254 + 9.541058216523389*theta(i,1)**2 + cos(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + sin(2.*theta(i,1))*(35.31242631181839*theta(i,1) - 3.577896831196271*theta(i,1)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) - (ny(i,1)*(sin(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + theta(i,1)*(-35.31242631181839 + 3.5778968311962718*theta(i,1)**2 + cos(2.*theta(i,1))*(-35.31242631181839 + 3.577896831196271*theta(i,1)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)
    elif (n==1 and m==3):
        return -(nx(i,1)*nz(i,1)*((-68.42444032663414*sin(2*theta(i,1)))/theta(i,1) + (1666.290634181943 - 168.83053934745612*theta(i,1)**2 + cos(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + sin(2.*theta(i,1))*(-112.55369289830418*theta(i,1) + 45.61629355108943*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))) - ny(i,1)*((-136.84888065326828*sin(theta(i,1))**2)/theta(i,1) + (sin(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + theta(i,1)*(-112.55369289830406 + 45.61629355108942*theta(i,1)**2 + cos(2.*theta(i,1))*(112.55369289830418 - 45.61629355108943*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==4):
        return -(nx(i,1)*nz(i,1)*((-29.040234416747317*sin(2*theta(i,1)))/theta(i,1) + (707.1957094874274 - 71.65390635204899*theta(i,1)**2 + cos(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + sin(2.*theta(i,1))*(-50.236672001638375*theta(i,1) + 20.360156277831546*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))) - ny(i,1)*((-58.080468833494635*sin(theta(i,1))**2)/theta(i,1) + (sin(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + theta(i,1)*(-50.23667200163832 + 20.360156277831535*theta(i,1)**2 + cos(2.*theta(i,1))*(50.236672001638375 - 20.360156277831546*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==5):
        return -((nx(i,1)*nz(i,1)*(27.73431477040989 - 11.240294400188406*theta(i,1)**2 + cos(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + sin(2.*theta(i,1))*(-29.974118400502412*theta(i,1) + 3.0370131549744803*theta(i,1)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) - (ny(i,1)*(sin(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + theta(i,1)*(29.97411840050242 - 3.037013154974481*theta(i,1)**2 + cos(2.*theta(i,1))*(29.974118400502412 - 3.0370131549744803*theta(i,1)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)
    elif (n==2 and m==1):
        return 1.*((nx(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. + (nx(i,1)**2*nx(i,2)*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) - cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) - (nx(i,2)*ny(i,1)**2*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. + nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) - cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) - (nx(i,2)*nz(i,1)**2*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. - nx(i,2)*ny(i,1)*nz(i,2)*sin(2*theta(i,1)) + nx(i,2)*nz(i,1)*nz(i,2)*sin(2*theta(i,1))) + (-0.5*(nx(i,2)*nz(i,2)) - (cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sin(2*theta(i,1)))*((44.06035162512098*sin(2*theta(i,2)))/theta(i,2) + (-1060.793564231141 + 107.4808595280735*theta(i,2)**2 + cos(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + sin(2.*theta(i,2))*(71.65390635204905*theta(i,2) - 29.04023441674732*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*ny(i,2) - (cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sin(2*theta(i,1)))*((88.12070325024196*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + theta(i,2)*(71.65390635204898 - 29.040234416747314*theta(i,2)**2 + cos(2.*theta(i,2))*(-71.65390635204905 + 29.04023441674732*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==2):
        return ((-0.5*(nx(i,2)*nz(i,2)) - (cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sin(2*theta(i,1)))*(-23.541617541212254 + 9.541058216523389*theta(i,2)**2 + cos(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + sin(2.*theta(i,2))*(35.31242631181839*theta(i,2) - 3.577896831196271*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((-0.5*ny(i,2) - (cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sin(2*theta(i,1)))*(sin(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + theta(i,2)*(-35.31242631181839 + 3.5778968311962718*theta(i,2)**2 + cos(2.*theta(i,2))*(-35.31242631181839 + 3.577896831196271*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    elif (n==2 and m==3):
        return (-0.5*(nx(i,2)*nz(i,2)) - (cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sin(2*theta(i,1)))*((-68.42444032663414*sin(2*theta(i,2)))/theta(i,2) + (1666.290634181943 - 168.83053934745612*theta(i,2)**2 + cos(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + sin(2.*theta(i,2))*(-112.55369289830418*theta(i,2) + 45.61629355108943*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*ny(i,2) - (cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sin(2*theta(i,1)))*((-136.84888065326828*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + theta(i,2)*(-112.55369289830406 + 45.61629355108942*theta(i,2)**2 + cos(2.*theta(i,2))*(112.55369289830418 - 45.61629355108943*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==4):
        return (-0.5*(nx(i,2)*nz(i,2)) - (cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sin(2*theta(i,1)))*((-29.040234416747317*sin(2*theta(i,2)))/theta(i,2) + (707.1957094874274 - 71.65390635204899*theta(i,2)**2 + cos(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + sin(2.*theta(i,2))*(-50.236672001638375*theta(i,2) + 20.360156277831546*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*ny(i,2) - (cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sin(2*theta(i,1)))*((-58.080468833494635*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + theta(i,2)*(-50.23667200163832 + 20.360156277831535*theta(i,2)**2 + cos(2.*theta(i,2))*(50.236672001638375 - 20.360156277831546*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==5):
        return ((-0.5*(nx(i,2)*nz(i,2)) - (cos(2*theta(i,1))*nx(i,2)*nz(i,2))/2. - (nx(i,1)**2*nx(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)*nz(i,2))/2. - nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*ny(i,1)*nz(i,2) + (nx(i,2)*ny(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*ny(i,1)**2*nz(i,2))/2. - nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*nx(i,2)*nz(i,1)*nz(i,2) + (nx(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,2)*ny(i,1)*nz(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*nz(i,2)*sin(2*theta(i,1)))*(27.73431477040989 - 11.240294400188406*theta(i,2)**2 + cos(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + sin(2.*theta(i,2))*(-29.974118400502412*theta(i,2) + 3.0370131549744803*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((-0.5*ny(i,2) - (cos(2*theta(i,1))*ny(i,2))/2. - (nx(i,1)**2*ny(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2) + (ny(i,1)**2*ny(i,2))/2. - (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2))/2. - nx(i,1)*ny(i,2)*nz(i,1) + cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,1) + (ny(i,2)*nz(i,1)**2)/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2)/2. + ny(i,1)*ny(i,2)*sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*sin(2*theta(i,1)))*(sin(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + theta(i,2)*(29.97411840050242 - 3.037013154974481*theta(i,2)**2 + cos(2.*theta(i,2))*(29.974118400502412 - 3.0370131549744803*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    else:
        print("frame bound error")    

def Cfz(i,n,m):
    if (n==1 and m==1):
        return 0.5*(1 - nx(i,1)**2 - ny(i,1)**2 + nz(i,1)**2) + ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*((44.06035162512098*sin(2*theta(i,1)))/theta(i,1) + (-1060.793564231141 + 107.4808595280735*theta(i,1)**2 + cos(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + sin(2.*theta(i,1))*(71.65390635204905*theta(i,1) - 29.04023441674732*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)))/2.
    elif (n==1 and m==2):
        return ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*(-23.541617541212254 + 9.541058216523389*theta(i,1)**2 + cos(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + sin(2.*theta(i,1))*(35.31242631181839*theta(i,1) - 3.577896831196271*theta(i,1)**3)))/(2.*(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==3):
        return ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*((-68.42444032663414*sin(2*theta(i,1)))/theta(i,1) + (1666.290634181943 - 168.83053934745612*theta(i,1)**2 + cos(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + sin(2.*theta(i,1))*(-112.55369289830418*theta(i,1) + 45.61629355108943*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)))/2.
    elif (n==1 and m==4):
        return ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*((-29.040234416747317*sin(2*theta(i,1)))/theta(i,1) + (707.1957094874274 - 71.65390635204899*theta(i,1)**2 + cos(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + sin(2.*theta(i,1))*(-50.236672001638375*theta(i,1) + 20.360156277831546*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)))/2.
    elif (n==1 and m==5):
        return ((1 + nx(i,1)**2 + ny(i,1)**2 - nz(i,1)**2)*(27.73431477040989 - 11.240294400188406*theta(i,1)**2 + cos(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + sin(2.*theta(i,1))*(-29.974118400502412*theta(i,1) + 3.0370131549744803*theta(i,1)**3)))/(2.*(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==2 and m==1):
        return 1.*(0.25 + cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (cos(2*theta(i,1))*nx(i,1)**2)/4. - nx(i,2)**2/4. - (cos(2*theta(i,1))*nx(i,2)**2)/4. + (nx(i,1)**2*nx(i,2)**2)/4. - (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (cos(2*theta(i,1))*ny(i,1)**2)/4. + (nx(i,2)**2*ny(i,1)**2)/4. - (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. - ny(i,2)**2/4. - (cos(2*theta(i,1))*ny(i,2)**2)/4. + (nx(i,1)**2*ny(i,2)**2)/4. - (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. + (ny(i,1)**2*ny(i,2)**2)/4. - (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. - (nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. - (nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (ny(i,1)*ny(i,2)**2*nz(i,1))/2. + (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (cos(2*theta(i,1))*nz(i,1)**2)/4. - (nx(i,2)**2*nz(i,1)**2)/4. + (cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. - (ny(i,2)**2*nz(i,1)**2)/4. + (cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. + nz(i,2)**2/4. + (cos(2*theta(i,1))*nz(i,2)**2)/4. - (nx(i,1)**2*nz(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. - (ny(i,1)**2*nz(i,2)**2)/4. + (cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. + (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (nz(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*sin(2*theta(i,1)))/2. + (nx(i,1)*nx(i,2)**2*sin(2*theta(i,1)))/2. + (ny(i,1)*sin(2*theta(i,1)))/2. - (nx(i,2)**2*ny(i,1)*sin(2*theta(i,1)))/2. + (nx(i,1)*ny(i,2)**2*sin(2*theta(i,1)))/2. - (ny(i,1)*ny(i,2)**2*sin(2*theta(i,1)))/2. - (nx(i,1)*nz(i,2)**2*sin(2*theta(i,1)))/2. + (ny(i,1)*nz(i,2)**2*sin(2*theta(i,1)))/2.) + (0.25 + cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*((44.06035162512098*sin(2*theta(i,2)))/theta(i,2) + (-1060.793564231141 + 107.4808595280735*theta(i,2)**2 + cos(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + sin(2.*theta(i,2))*(71.65390635204905*theta(i,2) - 29.04023441674732*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*(cos(2*theta(i,1))*nx(i,1)) - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1))/2. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*((88.12070325024196*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + theta(i,2)*(71.65390635204898 - 29.040234416747314*theta(i,2)**2 + cos(2.*theta(i,2))*(-71.65390635204905 + 29.04023441674732*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==2):
        return ((0.25 + cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*(-23.541617541212254 + 9.541058216523389*theta(i,2)**2 + cos(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + sin(2.*theta(i,2))*(35.31242631181839*theta(i,2) - 3.577896831196271*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((-0.5*(cos(2*theta(i,1))*nx(i,1)) - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1))/2. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*(sin(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + theta(i,2)*(-35.31242631181839 + 3.5778968311962718*theta(i,2)**2 + cos(2.*theta(i,2))*(-35.31242631181839 + 3.577896831196271*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    elif (n==2 and m==3):
        return (0.25 + cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*((-68.42444032663414*sin(2*theta(i,2)))/theta(i,2) + (1666.290634181943 - 168.83053934745612*theta(i,2)**2 + cos(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + sin(2.*theta(i,2))*(-112.55369289830418*theta(i,2) + 45.61629355108943*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*(cos(2*theta(i,1))*nx(i,1)) - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1))/2. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*((-136.84888065326828*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + theta(i,2)*(-112.55369289830406 + 45.61629355108942*theta(i,2)**2 + cos(2.*theta(i,2))*(112.55369289830418 - 45.61629355108943*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==4):
        return (0.25 + cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*((-29.040234416747317*sin(2*theta(i,2)))/theta(i,2) + (707.1957094874274 - 71.65390635204899*theta(i,2)**2 + cos(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + sin(2.*theta(i,2))*(-50.236672001638375*theta(i,2) + 20.360156277831546*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (-0.5*(cos(2*theta(i,1))*nx(i,1)) - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1))/2. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*((-58.080468833494635*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + theta(i,2)*(-50.23667200163832 + 20.360156277831535*theta(i,2)**2 + cos(2.*theta(i,2))*(50.236672001638375 - 20.360156277831546*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==5):
        return ((0.25 + cos(2*theta(i,1))/4. - nx(i,1)**2/4. + (cos(2*theta(i,1))*nx(i,1)**2)/4. + nx(i,2)**2/4. + (cos(2*theta(i,1))*nx(i,2)**2)/4. - (nx(i,1)**2*nx(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*nx(i,2)**2)/4. - ny(i,1)**2/4. + (cos(2*theta(i,1))*ny(i,1)**2)/4. - (nx(i,2)**2*ny(i,1)**2)/4. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)**2)/4. + ny(i,2)**2/4. + (cos(2*theta(i,1))*ny(i,2)**2)/4. - (nx(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)**2)/4. - (ny(i,1)**2*ny(i,2)**2)/4. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)**2)/4. + (nx(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nz(i,1))/2. + (nx(i,1)*nx(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2*nz(i,1))/2. + (ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,1))/2. + (nx(i,2)**2*ny(i,1)*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1)*nz(i,1))/2. + (nx(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2*nz(i,1))/2. + (ny(i,1)*ny(i,2)**2*nz(i,1))/2. - (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2*nz(i,1))/2. + nz(i,1)**2/4. - (cos(2*theta(i,1))*nz(i,1)**2)/4. + (nx(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*nx(i,2)**2*nz(i,1)**2)/4. + (ny(i,2)**2*nz(i,1)**2)/4. - (cos(2*theta(i,1))*ny(i,2)**2*nz(i,1)**2)/4. - nz(i,2)**2/4. - (cos(2*theta(i,1))*nz(i,2)**2)/4. + (nx(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*nx(i,1)**2*nz(i,2)**2)/4. + (ny(i,1)**2*nz(i,2)**2)/4. - (cos(2*theta(i,1))*ny(i,1)**2*nz(i,2)**2)/4. - (nx(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,1)*nz(i,2)**2)/2. - (ny(i,1)*nz(i,1)*nz(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*nz(i,1)*nz(i,2)**2)/2. - (nz(i,1)**2*nz(i,2)**2)/4. + (cos(2*theta(i,1))*nz(i,1)**2*nz(i,2)**2)/4.)*(27.73431477040989 - 11.240294400188406*theta(i,2)**2 + cos(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + sin(2.*theta(i,2))*(-29.974118400502412*theta(i,2) + 3.0370131549744803*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((-0.5*(cos(2*theta(i,1))*nx(i,1)) - (cos(2*theta(i,1))*nx(i,1)*nx(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1))/2. + (cos(2*theta(i,1))*nx(i,2)**2*ny(i,1))/2. - (cos(2*theta(i,1))*nx(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*ny(i,1)*ny(i,2)**2)/2. + (cos(2*theta(i,1))*nx(i,1)*nz(i,2)**2)/2. - (cos(2*theta(i,1))*ny(i,1)*nz(i,2)**2)/2.)*(sin(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + theta(i,2)*(29.97411840050242 - 3.037013154974481*theta(i,2)**2 + cos(2.*theta(i,2))*(29.974118400502412 - 3.0370131549744803*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    else:
        print("frame bound error")  

def Cfy(i,n,m):
    if (n==1 and m==1):
        return 1.*ny(i,1)*nz(i,1) - ny(i,1)*nz(i,1)*((44.06035162512098*sin(2*theta(i,1)))/theta(i,1)+ (-1060.793564231141 + 107.4808595280735*theta(i,1)**2 + cos(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + sin(2.*theta(i,1))*(71.65390635204905*theta(i,1) - 29.04023441674732*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) + nx(i,1)*((88.12070325024196*sin(theta(i,1))**2)/theta(i,1) + (sin(2.*theta(i,1))*(-1060.793564231141 + 107.48085952807347*theta(i,1)**2) + theta(i,1)*(71.65390635204898 - 29.040234416747314*theta(i,1)**2 + cos(2.*theta(i,1))*(-71.65390635204905 + 29.04023441674732*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==2):
        return -((ny(i,1)*nz(i,1)*(-23.541617541212254 + 9.541058216523389*theta(i,1)**2 + cos(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + sin(2.*theta(i,1))*(35.31242631181839*theta(i,1) - 3.577896831196271*theta(i,1)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) + (nx(i,1)*(sin(2.*theta(i,1))*(23.54161754121226 - 9.541058216523389*theta(i,1)**2) + theta(i,1)*(-35.31242631181839 + 3.5778968311962718*theta(i,1)**2 + cos(2.*theta(i,1))*(-35.31242631181839 + 3.577896831196271*theta(i,1)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)
    elif (n==1 and m==3):
        return -(ny(i,1)*nz(i,1)*((-68.42444032663414*sin(2*theta(i,1)))/theta(i,1) + (1666.290634181943 - 168.83053934745612*theta(i,1)**2 + cos(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + sin(2.*theta(i,1))*(-112.55369289830418*theta(i,1) + 45.61629355108943*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))) + nx(i,1)*((-136.84888065326828*sin(theta(i,1))**2)/theta(i,1) + (sin(2.*theta(i,1))*(1666.290634181943 - 168.83053934745607*theta(i,1)**2) + theta(i,1)*(-112.55369289830406 + 45.61629355108942*theta(i,1)**2 + cos(2.*theta(i,1))*(112.55369289830418 - 45.61629355108943*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==4):
        return -(ny(i,1)*nz(i,1)*((-29.040234416747317*sin(2*theta(i,1)))/theta(i,1) + (707.1957094874274 - 71.65390635204899*theta(i,1)**2 + cos(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + sin(2.*theta(i,1))*(-50.236672001638375*theta(i,1) + 20.360156277831546*theta(i,1)**3))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))) + nx(i,1)*((-58.080468833494635*sin(theta(i,1))**2)/theta(i,1) + (sin(2.*theta(i,1))*(707.1957094874274 - 71.65390635204898*theta(i,1)**2) + theta(i,1)*(-50.23667200163832 + 20.360156277831535*theta(i,1)**2 + cos(2.*theta(i,1))*(50.236672001638375 - 20.360156277831546*theta(i,1)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4))
    elif (n==1 and m==5):
        return -((ny(i,1)*nz(i,1)*(27.73431477040989 - 11.240294400188406*theta(i,1)**2 + cos(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + sin(2.*theta(i,1))*(-29.974118400502412*theta(i,1) + 3.0370131549744803*theta(i,1)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)) + (nx(i,1)*(sin(2.*theta(i,1))*(-27.734314770409892 + 11.240294400188406*theta(i,1)**2) + theta(i,1)*(29.97411840050242 - 3.037013154974481*theta(i,1)**2 + cos(2.*theta(i,1))*(29.974118400502412 - 3.0370131549744803*theta(i,1)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,1)**2 + 1.*theta(i,1)**4)
    elif (n==2 and m==1):
        return 1.*((ny(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. - (nx(i,1)**2*ny(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. + nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + (ny(i,1)**2*ny(i,2)*nz(i,2))/2. - (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. + ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) - cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) - (ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + nx(i,1)*ny(i,2)*nz(i,2)*sin(2*theta(i,1)) - ny(i,2)*nz(i,1)*nz(i,2)*sin(2*theta(i,1))) + (-0.5*(ny(i,2)*nz(i,2)) - (cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sin(2*theta(i,1)))/2.)*((44.06035162512098*sin(2*theta(i,2)))/theta(i,2) + (-1060.793564231141 + 107.4808595280735*theta(i,2)**2 + cos(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + sin(2.*theta(i,2))*(71.65390635204905*theta(i,2) - 29.04023441674732*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sin(2*theta(i,1)))*((88.12070325024196*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(-1060.793564231141 + 107.48085952807347*theta(i,2)**2) + theta(i,2)*(71.65390635204898 - 29.040234416747314*theta(i,2)**2 + cos(2.*theta(i,2))*(-71.65390635204905 + 29.04023441674732*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==2):
        return ((-0.5*(ny(i,2)*nz(i,2)) - (cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sin(2*theta(i,1)))/2.)*(-23.541617541212254 + 9.541058216523389*theta(i,2)**2 + cos(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + sin(2.*theta(i,2))*(35.31242631181839*theta(i,2) - 3.577896831196271*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sin(2*theta(i,1)))*(sin(2.*theta(i,2))*(23.54161754121226 - 9.541058216523389*theta(i,2)**2) + theta(i,2)*(-35.31242631181839 + 3.5778968311962718*theta(i,2)**2 + cos(2.*theta(i,2))*(-35.31242631181839 + 3.577896831196271*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    elif (n==2 and m==3):
        return (-0.5*(ny(i,2)*nz(i,2)) - (cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sin(2*theta(i,1)))/2.)*((-68.42444032663414*sin(2*theta(i,2)))/theta(i,2) + (1666.290634181943 - 168.83053934745612*theta(i,2)**2 + cos(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + sin(2.*theta(i,2))*(-112.55369289830418*theta(i,2) + 45.61629355108943*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sin(2*theta(i,1)))*((-136.84888065326828*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(1666.290634181943 - 168.83053934745607*theta(i,2)**2) + theta(i,2)*(-112.55369289830406 + 45.61629355108942*theta(i,2)**2 + cos(2.*theta(i,2))*(112.55369289830418 - 45.61629355108943*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==4):
        return (-0.5*(ny(i,2)*nz(i,2)) - (cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sin(2*theta(i,1)))/2.)*((-29.040234416747317*sin(2*theta(i,2)))/theta(i,2) + (707.1957094874274 - 71.65390635204899*theta(i,2)**2 + cos(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + sin(2.*theta(i,2))*(-50.236672001638375*theta(i,2) + 20.360156277831546*theta(i,2)**3))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)) + (nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sin(2*theta(i,1)))*((-58.080468833494635*sin(theta(i,2))**2)/theta(i,2) + (sin(2.*theta(i,2))*(707.1957094874274 - 71.65390635204898*theta(i,2)**2) + theta(i,2)*(-50.23667200163832 + 20.360156277831535*theta(i,2)**2 + cos(2.*theta(i,2))*(50.236672001638375 - 20.360156277831546*theta(i,2)**2)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4))
    elif (n==2 and m==5): 
        return ((-0.5*(ny(i,2)*nz(i,2)) - (cos(2*theta(i,1))*ny(i,2)*nz(i,2))/2. + (nx(i,1)**2*ny(i,2)*nz(i,2))/2. - (cos(2*theta(i,1))*nx(i,1)**2*ny(i,2)*nz(i,2))/2. - nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*nx(i,1)*ny(i,1)*ny(i,2)*nz(i,2) - (ny(i,1)**2*ny(i,2)*nz(i,2))/2. + (cos(2*theta(i,1))*ny(i,1)**2*ny(i,2)*nz(i,2))/2. - ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + cos(2*theta(i,1))*ny(i,1)*ny(i,2)*nz(i,1)*nz(i,2) + (ny(i,2)*nz(i,1)**2*nz(i,2))/2. - (cos(2*theta(i,1))*ny(i,2)*nz(i,1)**2*nz(i,2))/2. + (nx(i,2)*sin(2*theta(i,1)))/2. + (nx(i,1)**2*nx(i,2)*sin(2*theta(i,1)))/2. - nx(i,1)*nx(i,2)*ny(i,1)*sin(2*theta(i,1)) - (nx(i,2)*ny(i,1)**2*sin(2*theta(i,1)))/2. - nx(i,2)*ny(i,1)*nz(i,1)*sin(2*theta(i,1)) + (nx(i,2)*nz(i,1)**2*sin(2*theta(i,1)))/2.)*(27.73431477040989 - 11.240294400188406*theta(i,2)**2 + cos(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + sin(2.*theta(i,2))*(-29.974118400502412*theta(i,2) + 3.0370131549744803*theta(i,2)**3)))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4) + ((nx(i,2)/2. - (nx(i,1)**2*nx(i,2))/2. + nx(i,1)*nx(i,2)*ny(i,1) + (nx(i,2)*ny(i,1)**2)/2. + nx(i,2)*ny(i,1)*nz(i,1) - (nx(i,2)*nz(i,1)**2)/2. - cos(2*theta(i,1))*nx(i,1)*ny(i,2)*nz(i,2) + cos(2*theta(i,1))*ny(i,2)*nz(i,1)*nz(i,2) + nx(i,1)*nx(i,2)*sin(2*theta(i,1)) - nx(i,2)*nz(i,1)*sin(2*theta(i,1)))*(sin(2.*theta(i,2))*(-27.734314770409892 + 11.240294400188406*theta(i,2)**2) + theta(i,2)*(29.97411840050242 - 3.037013154974481*theta(i,2)**2 + cos(2.*theta(i,2))*(29.974118400502412 - 3.0370131549744803*theta(i,2)**2))))/(24.352272758500607 - 12.337005501361698*theta(i,2)**2 + 1.*theta(i,2)**4)
    else:
        print("frame boudn error")

# vecterized S[n,m]
size_basis = 85 # num_spcetra =85 D1+D2 
def position_s1_func(n,m):
    if (n==1):
        return m
    elif(n==2):
        return 1*5+m
S1_vec_func = lambda n,m: np.array([int(bool((i+1)==position_s1_func(n,m))) for i in range(size_basis)])

def position_s2_func(n1,n2,m1,m2):
    if (n1==1)&(n2==1):
        return 0*25 + (m1-1)*5+m2 +10
    elif (n1==2)&(n2==1):
        return 1*25 + (m1-1)*5+m2 +10
    elif (n1==2)&(n2==2):
        return 2*25 + (m1-1)*5+m2 +10
    else:
        print("error bound")
S2_vec_func = lambda n1,n2,m1,m2: np.array([int(bool((i+1)==position_s2_func(n1,n2,m1,m2))) for i in range(size_basis)])

#def position_s3_func(n1,n2,n3,m1,m2,m3):
#    return (n1 + n2 + n3 - 3)*5**3 + (m3) + (m2 - 1)*5**1 + (m1 - 1)*5**2 +10+75;     
#S3_vec_func = lambda n1,n2,n3,m1,m2,m3: np.array([int(bool((i+1)==position_s3_func(n1,n2,n3,m1,m2,m3) )) for i in range(size_basis)])    

# Dyson vectorized
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


# obtian the control dynamics: sympy add all Dyson together
def control_dynamics (i):
    """
    It is actually dyson expansion with control -"i"- specified
    specify the random control and simplify the Dyson into numerics
    return is the Dyson that filtering is numerical & S[n,m] is numerically vectors
    """
    tmp_00 = D1_00(i)+D2_00(i)   #+D3_00(i)
    tmp_01 = D1_01(i)+D2_01(i)   #+D3_01(i)
    tmp_10 = D1_10(i)+D2_10(i)   #+D3_10(i)
    tmp_11 = D1_11(i)+D2_11(i)   #+D3_11(i)
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
results = Parallel(n_jobs=-1, verbose=100)(delayed(control_dynamics)(i) for i in range (ddd) )
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


matrix = np.resize(np.array([exp_x_0,exp_x_1,exp_x_2,exp_x_3]),(4*ddd,ddd)) # rows are different ctrl & 4types of init, columns are S.

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
np.save('result_D2_qns_theta', QNS_theta)
np.save('result_D2_qns_n', QNS_n)
np.save('result_D2_qns_rho', QNS_rho)
np.save('result_D2_qns_Matrix', obs_to_S_matrix)


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





















































