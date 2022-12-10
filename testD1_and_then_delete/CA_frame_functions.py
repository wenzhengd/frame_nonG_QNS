#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:42:18 2022

@author: wenzhengdong
L=2  Dyson 1, 2, 3,4 

core functions 

"""

import numpy as np
from operator import add
import sympy as sp


sigma_0 = np.matrix([[1,0],[0,1]])
sigma_1= np.matrix([[0,1],[1,0]])
sigma_2 = np.matrix([[0,-1j],[1j,0]])
sigma_3 = np.matrix([[1,0],[0,-1]])

# Define some symbols
theta_1 = sp.Symbol('theta_1')
theta_2 = sp.Symbol('theta_2')
nx_1 = sp.Symbol('nx_1')
ny_1 = sp.Symbol('ny_1')
nz_1 = sp.Symbol('nz_1')
nx_2 = sp.Symbol('nx_2')
ny_2 = sp.Symbol('ny_2')
nz_2 = sp.Symbol('nz_2')

# functions that relevant to controls
# n is window, i is which control 

"""
theta = lambda i,n : F1_theta[i][0] if(n ==1 ) else ( F1_theta[i][1] if (n==2) else print("bound error")) # n can ONLY take 1 or 2 ,two pulses
nx = lambda i,n : F1_n[i][0][0] if(n ==1 ) else ( F1_n[i][1][0] if (n==2) else print("bound error"))
ny = lambda i,n : F1_n[i][0][1] if(n ==1 ) else ( F1_n[i][1][1] if (n==2) else print("bound error"))
nz = lambda i,n : F1_n[i][0][2] if(n ==1 ) else ( F1_n[i][1][2] if (n==2) else print("bound error"))
"""
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

"""
 THESE PART ARE COMMENTED FOF OPTIMIZED_GATE CODE

# vecterized S[n,m]
size_basis = 3710 # num_spcetra =3710 
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

def position_s3_func(n1,n2,n3,m1,m2,m3):
    return (n1 + n2 + n3 - 3)*5**3 + (m3) + (m2 - 1)*5**1 + (m1 - 1)*5**2 +10+75;     
S3_vec_func = lambda n1,n2,n3,m1,m2,m3: np.array([int(bool((i+1)==position_s3_func(n1,n2,n3,m1,m2,m3) )) for i in range(size_basis)])    

def position_s4_func(n1,n2,n3,n4,m1,m2,m3,m4):
   return (n1 + n2 + n3 +n4 - 4)*5**4 + (m4) + (m2 - 1)*5**1 + (m2 - 1)*5**2  +(m1-1)*5**3  +10+75+500
S4_vec_func = lambda n1,n2,n3,n4,m1,m2,m3,m4: np.array([int(bool((i+1)==position_s4_func(n1,n2,n3,n4,m1,m2,m3,m4))) for i in range(size_basis)])
"""

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


# obtian the control dynamics: sympy add all Dyson together
def control_dynamics (i):
    """
    It is actually dyson expansion with control -"i"- specified
    specify the random control and simplify the Dyson into numerics
    return is the Dyson that filtering is numerical & S[n,m] is numerically vectors
    """
    tmp_00 = D1_00(i)+D2_00(i)+D3_00(i)+D4_00(i) 
    tmp_01 = D1_01(i)+D2_01(i)+D3_01(i)+D4_00(i) 
    tmp_10 = D1_10(i)+D2_10(i)+D3_10(i)+D4_00(i) 
    tmp_11 = D1_11(i)+D2_11(i)+D3_11(i)+D4_00(i) 
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




###