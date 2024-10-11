#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:40:52 2022

@author: wakamototamaki
"""

#2-d
#graph
# %%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import gc

# %%
N = 20000 #number of time step
T = 200 #TIME
s = T/N #dt
Iy = 100 #number of cells (y-axis)
Ix = 100 #number of cells (x-axis)

# %%
#Biochemicals
Nm1 = np.zeros((Iy+2, Ix+2)) #Notch1 in membrane
Nm2 = np.zeros((Iy+2, Ix+2)) #Notch2 in membrane
Dm1 = np.zeros((Iy+2, Ix+2)) #DLL1 in membrane
Dm2 = np.zeros((Iy+2, Ix+2)) #DLL2 in membrane
Nc1 = np.zeros((Iy, Ix)) #Notch1 in cytosol
Nc2 = np.zeros((Iy, Ix)) #Notch2 in cytosol
Dc1 = np.zeros((Iy, Ix)) #DLL1 in cytosol
Dc2 = np.zeros((Iy, Ix)) #DLL2 in cytosol
I1  = np.zeros((Iy, Ix)) #NICD1
I2  = np.zeros((Iy, Ix)) #NICD2
H1  = np.zeros((Iy, Ix)) #Hes1

# %%
#initial condition
e = 0.01
y_init = e*np.random.rand(11, Iy, Ix)

# %%
#parameter
#binding rate
alpha1 = 6.0
alpha2 = 10.0
alpha3 = 8.0
alpha4 = 6.0
#basal decay
mu0 = 0.5
mu1 = 1.0
mu2 = 1.0
mu3s = np.linspace(0.02, 0.22, 11)
mu4 = 1.0
mu5 = 0.5
#move to membrane
gammaN = 2.0
gammaD = 1.0
#parameter of function
beta1 = 0.1
beta21 = 8.0
beta22 = 5.0
beta3 = 2.0
beta41 = 8.0
beta42 = 8.0
#parameter of Hes-1
nu0 = 0.5
nu1 = 5.0
nu2 = 25.0
nu3 = 5.0

change = mu3s
p = 4
'''
alpha16s = np.linspace(0.0, 15.0, 16)
mu125s = np.linspace(0.2, 2.2, 11)[4]
mu4s = np.linspace(0.02, 0.22, 11)[4]
mu06s = np.linspace(0.1, 1.1, 11)[4]
gammaNs = np.linspace(0.0, 4.0, 11)
gammaDs = np.linspace(0.0, 2.0, 11)
beta1s = np.linspace(0.0, 0.2, 11)
beta2_s = np.linspace(0.0, 10.0, 11)
beta3s = np.linspace(0.4, 4.4, 11)[4]
beta4_s = np.linspace(0.0, 16.0, 11)
nu2s = np.linspace(0.0, 50.0, 11)
nu0s = np.linspace(0.0, 1.0, 11)
nu1s = np.linspace(0.0, 10.0, 11)
nu3s = np.linspace(0.0, 10.0, 11)
'''
# %%
m = change.size

lccell5 = np.zeros(m)
lccell75 = np.zeros(m)
btcell5 = np.zeros(m)
btcell75 = np.zeros(m)

# %%
#activation and inhibition of Hes-1 by NICD
def hes(x1, x2):
    return nu0 + (nu2*(x1**2)/(nu1 + x1**2))*(1 - ((x2**2)/(nu3 + x2**2)))

#ODEs
def Notchm1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return -alpha1*d_ave1*nm1 - alpha2*d_ave2*nm1 - mu1*nm1 + gammaN*nc1
def Notchm2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return -alpha3*d_ave1*nm2 - alpha4*d_ave2*nm2 - mu1*nm2 + gammaN*nc2
    
def Deltam1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return -alpha1*n_ave1*dm1 -alpha3*n_ave2*dm1 - mu2*dm1 + gammaD*dc1    
def Deltam2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return -alpha2*n_ave1*dm2 - alpha4*n_ave2*dm2 - mu2*dm2 + gammaD*dc2
    
def NICD1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return alpha1*d_ave1*nm1 + alpha2*d_ave2*nm1 - mu4*i1
def NICD2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return alpha3*d_ave1*nm2 + alpha4*d_ave2*nm2  - mu3*i1
    
def Notchc1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return beta21*(i1**2)/(beta1 + i1**2) - (mu4 + gammaN)*nc1    
def Notchc2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return beta22*(i2**2)/(beta1 + i2**2) - (mu4 + gammaN)*nc2
    
def Deltac1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return beta41/(beta3 + h**2) - (mu5 + gammaD)*dc1    
def Deltac2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return beta42/(beta3 + h**2) - (mu5 + gammaD)*dc2

def BTHes1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return hes(i2, i1)-mu0*h
def LCHes1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return hes(i1, i2)-mu0*h

# %%
#Runge-Kutta method
def RungeKutta(Hes1, nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave):
    nm11 = Notchm1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    nm21 = Notchm2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    dm11 = Deltam1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    dm21 = Deltam2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    i11    = NICD1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    i21    = NICD2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    nc11 = Notchc1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    nc21 = Notchc2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    dc11 = Deltac1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    dc21 = Deltac2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    h11   = Hes1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave)
    
    nm12 = Notchm1(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    nm22 = Notchm2(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    dm12 = Deltam1(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    dm22 = Deltam2(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    i12    = NICD1(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    i22    = NICD2(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    nc12 = Notchc1(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    nc22 = Notchc2(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    dc12 = Deltac1(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    dc22 = Deltac2(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
    h12   =   Hes1(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)

    nm13 = Notchm2(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    nm23 = Notchm2(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    dm13 = Deltam1(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    dm23 = Deltam2(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    i13    = NICD1(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    i23    = NICD2(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    nc13 = Notchc1(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    nc23 = Notchc2(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    dc13 = Deltac1(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    dc23 = Deltac2(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
    h13   =   Hes1(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)

    nm14 = Notchm2(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    nm24 = Notchm2(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    dm14 = Deltam1(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    dm24 = Deltam2(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    i14    = NICD1(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    i24    = NICD2(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    nc14 = Notchc1(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    nc24 = Notchc2(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    dc14 = Deltac1(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    dc24 = Deltac2(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    h14   =   Hes1(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    
    nm1 = nm1 + (s/6)*(nm11 + 2*nm12 + 2*nm13 + nm14)
    nm2 = nm2 + (s/6)*(nm21 + 2*nm22 + 2*nm23 + nm24)
    dm1 = dm1 + (s/6)*(dm11 + 2*dm12 + 2*dm13 + dm14)
    dm2 = dm2 + (s/6)*(dm21 + 2*dm22 + 2*dm23 + dm24)
    i1 = i1 + (s/6)*(i11 + 2*i12 + 2*i13 + i14)
    i2 = i2 + (s/6)*(i21 + 2*i22 + 2*i23 + i24)
    nc1 = nc1 + (s/6)*(nc11 + 2*nc12 + 2*nc13 + nc14)
    nc2 = nc2 + (s/6)*(nc21 + 2*nc22 + 2*nc23 + nc24)
    dc1 = dc1 + (s/6)*(dc11 + 2*dc12 + 2*dc13 + dc14)
    dc2 = dc2 + (s/6)*(dc21 + 2*dc22 + 2*dc23 + dc24)
    h = h + (s/6)*(h11 + 2*h12 + 2*h13 + h14)
    return nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h

# %%
'''
NSCLC TYPE
'''
Nm1[1:-1, 1:-1] = y_init[0, :, :]
Nm2[1:-1, 1:-1] = y_init[1, :, :]
Dm1[1:-1, 1:-1] = y_init[2, :, :]
Dm2[1:-1, 1:-1] = y_init[3, :, :]
Nc1 = y_init[4, :, :]
Nc2 = y_init[5, :, :]
Dc1 = y_init[6, :, :]
Dc2 = y_init[7, :, :]
I1  = y_init[8, :, :]
I2  = y_init[9, :, :]
H1  = y_init[10, :, :]

mu3 = change[p]
for l in trange(N-1):
    n1_ave = (Nm1[0:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/4
    n2_ave = (Nm2[0:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/4
    d1_ave = (Dm1[0:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/4
    d2_ave = (Dm2[0:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/4
    
    Nm1[1:-1, 1:-1] = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[0]
    Nm2[1:-1, 1:-1] = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[1]
    Dm1[1:-1, 1:-1] = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[2]
    Dm2[1:-1, 1:-1] = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[3]
    I1              = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[4]
    I2              = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[5]
    Nc1             = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[6]
    Nc2             = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[7]
    Dc1             = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[8]
    Dc2             = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[9]
    H1              = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[10]

hmax = np.max(H1-y_init[-1, :, :])

# %%
for q in trange(m):
    Nm1[1:-1, 1:-1] = y_init[0, :, :]
    Nm2[1:-1, 1:-1] = y_init[1, :, :]
    Dm1[1:-1, 1:-1] = y_init[2, :, :]
    Dm2[1:-1, 1:-1] = y_init[3, :, :]
    Nc1 = y_init[4, :, :]
    Nc2 = y_init[5, :, :]
    Dc1 = y_init[6, :, :]
    Dc2 = y_init[7, :, :]
    I1  = y_init[8, :, :]
    I2  = y_init[9, :, :]
    H1  = y_init[10, :, :]
    mu3 = change[q]
    for l in range(N-1):
        n1_ave = (Nm1[0:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, 0:-2] + Nm1[1:-1, 2:])/4
        n2_ave = (Nm2[0:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, 0:-2] + Nm2[1:-1, 2:])/4
        d1_ave = (Dm1[0:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, 0:-2] + Dm1[1:-1, 2:])/4
        d2_ave = (Dm2[0:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, 0:-2] + Dm2[1:-1, 2:])/4
        Nm1[1:-1, 1:-1] = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[0]
        Nm2[1:-1, 1:-1] = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[1]
        Dm1[1:-1, 1:-1] = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[2]
        Dm2[1:-1, 1:-1] = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[3]
        I1              = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[4]
        I2              = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[5]
        Nc1             = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[6]
        Nc2             = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[7]
        Dc1             = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[8]
        Dc2             = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[9]
        H1              = RungeKutta(LCHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[10]
    h = (H1-y_init[-1, :, :])/hmax
    lccell5[q]  = 100*(np.count_nonzero(h>0.5))/(Iy*Ix)
    lccell75[q] = 100*(np.count_nonzero(h>0.75))/(Iy*Ix)

# %%
lccell5  = lccell5/lccell5[p]
lccell75 = lccell75/lccell75[p]
print("lung", lccell5)
print(lccell75)
print(gc.collect())

# %%
'''
EBT TYPE
'''

beta21 = 5.0
beta22 = 8.0
mu3 = change[p]

# %%
for l in range(N-1):
    n1_ave = (Nm1[:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, :-2] + Nm1[1:-1, 2:])/4
    n2_ave = (Nm2[:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, :-2] + Nm2[1:-1, 2:])/4
    d1_ave = (Dm1[:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, :-2] + Dm1[1:-1, 2:])/4
    d2_ave = (Dm2[:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, :-2] + Dm2[1:-1, 2:])/4
    
    Nm1[1:-1, 1:-1] = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[0]
    Nm2[1:-1, 1:-1] = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[1]
    Dm1[1:-1, 1:-1] = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[2]
    Dm2[1:-1, 1:-1] = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[3]
    I1              = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[4]
    I2              = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[5]
    Nc1             = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[6]
    Nc2             = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[7]
    Dc1             = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[8]
    Dc2             = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[9]
    H1              = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[10]
hmax = np.max(H1-y_init[-1, :, :])

# %%
for q in trange(m):
    Nm1[1:-1, 1:-1] = y_init[0, :, :]
    Nm2[1:-1, 1:-1] = y_init[1, :, :]
    Dm1[1:-1, 1:-1] = y_init[2, :, :]
    Dm2[1:-1, 1:-1] = y_init[3, :, :]
    Nc1 = y_init[4, :, :]
    Nc2 = y_init[5, :, :]
    Dc1 = y_init[6, :, :]
    Dc2 = y_init[7, :, :]
    I1  = y_init[8, :, :]
    I2  = y_init[9, :, :]
    H1  = y_init[10, :, :]
    mu3 = change[q]
    for l in range(N-1):
        n1_ave = (Nm1[:-2, 1:-1] + Nm1[2:,1:-1] + Nm1[1:-1, :-2] + Nm1[1:-1, 2:])/4
        n2_ave = (Nm2[:-2, 1:-1] + Nm2[2:,1:-1] + Nm2[1:-1, :-2] + Nm2[1:-1, 2:])/4
        d1_ave = (Dm1[:-2, 1:-1] + Dm1[2:,1:-1] + Dm1[1:-1, :-2] + Dm1[1:-1, 2:])/4
        d2_ave = (Dm2[:-2, 1:-1] + Dm2[2:,1:-1] + Dm2[1:-1, :-2] + Dm2[1:-1, 2:])/4
        Nm1[1:-1, 1:-1] = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[0]
        Nm2[1:-1, 1:-1] = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[1]
        Dm1[1:-1, 1:-1] = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[2]
        Dm2[1:-1, 1:-1] = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[3]
        I1              = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[4]
        I2              = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[5]
        Nc1             = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[6]
        Nc2             = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[7]
        Dc1             = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[8]
        Dc2             = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[9]
        H1              = RungeKutta(BTHes1, Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[10]

    h = (H1-y_init[-1, :, :])/hmax
    btcell5[q]  = 100*(np.count_nonzero(h>0.5))/(Iy*Ix)
    btcell75[q] = 100*(np.count_nonzero(h>0.75))/(Iy*Ix)
print(btcell5)
print(btcell75)

# %%
btcell5  = btcell5/btcell5[p]
btcell75 = btcell75/btcell75[p]
print("brain", btcell5)
print(btcell75)
print(gc.collect())

# %%
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(111)
ax1.set_xlabel(r"$\mu_{3}$", fontsize=15)
#ax1.set_ylabel("% of cells with high expression of Hes-1")
ax1.set_xlim([change[0], change[m-1]])
ax1.set_ylim([0, 2.0])
ax1.plot(change, lccell5, marker='o', markersize=10, color="g", label=">0.5 Model-LC")
ax1.plot(change, lccell75, marker='o', mfc="white", markersize=10, color="g", label=">0.75 Model-LC")
ax1.plot(change, btcell5, marker='o', color="r", markersize=10, label=">0.5 Model-BT")
ax1.plot(change, btcell75, marker='o', mfc="white", color="r", markersize=10, label=">0.75 Model-BT")
ax1.legend(loc="best", fontsize=10)
fig1.savefig("count_2d_mu3.png")

plt.show()
# %%
