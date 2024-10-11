#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:05:17 2022

@author: wakamototamaki
"""

#Notch1-DLL1-NICD1_Notchc1_DLLc1_Notch2-DLL2-NICD2_Notchc2_DLLc2_hes1 model
#2cells
#graph
# %%
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

# %%
N = 50000 #時間刻み数
T = 500 #最終時刻
s = T/N #時間の刻み幅
Iy = 1 #細胞の数（たて）
Ix = 2 #細胞の数（よこ）

# %%
#sequence
t = np.linspace(0, T, N+1)
Nm1 = np.zeros((Iy, Ix+2)) #Notch1 in membrane
Nm2 = np.zeros((Iy, Ix+2)) #Notch2 in membrane
Dm1 = np.zeros((Iy, Ix+2)) #DLL1 in membrane
Dm2 = np.zeros((Iy, Ix+2)) #DLL2 in membrane
Nc1 = np.zeros((Iy, Ix)) #Notch1 in cytosol
Nc2 = np.zeros((Iy, Ix)) #Notch2 in cytosol
Dc1 = np.zeros((Iy, Ix)) #DLL1 in cytosol
Dc2 = np.zeros((Iy, Ix)) #DLL2 in cytosol
I1  = np.zeros((Iy, Ix)) #NICD1
I2  = np.zeros((Iy, Ix)) #NICD2
H1  = np.zeros((Iy, Ix)) #Hes1

# %%
#parameter
#binding rate
alpha1 = 6.0
alpha2s = np.linspace(0.0, 15.0, 16)
alpha3 = 8.0
alpha4 = 6.0
#basal decay
mu0 = 0.5
mu1 = 1.0
mu2 = 1.0
mu3 = 0.1
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

change = alpha2s
'''
alpha1s = np.linspace(0.0, 15.0, 11)
mu0s = np.linspace(0.1, 1.1, 11)[4]
mu124s = np.linspace(0.2, 2.2, 11)[4]
mu3s = np.linspace(0.02, 0.22, 11)[4]
mu5s = np.linspace(0.1, 1.1, 11)[4]
gammaNs = np.linspace(0.0, 4.0, 11)
gammaDs = np.linspace(0.0, 2.0, 11)
beta1s = np.linspace(0.0, 2.0, 11)
beta21s = np.linspace(0.0, 10.0, 11)
beta3s = np.linspace(0.4, 4.4, 11)[4]
beta41s = np.linspace(0.0, 16.0, 11)
nu0s = np.linspace(0.0, 1.0, 11)
nu1s = np.linspace(0.0, 10.0, 11)
nu2s = np.linspace(0.0, 50.0, 11)
'''

# %%
#initial condition
e = 0.01
y_init = e*np.random.rand(11, Iy, Ix)
m = change.size

lc1 = np.zeros(m) #Model-LC_cell1
lc2 = np.zeros(m) #Model-LC_cell2
bt1 = np.zeros(m) #Model-BT_cell1
bt2 = np.zeros(m) #Model-BT_cell2

# %%
#activation and inhibition of Hes-1 by NICD
def hes(x1, x2): #x1:activation x2:inhibition
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
#Model-LC
for q in trange(m):
    alpha2 = change[q]
    Nm1[:, 1:-1] = y_init[0, :, :]
    Nm2[:, 1:-1] = y_init[1, :, :]
    Dm1[:, 1:-1] = y_init[2, :, :]
    Dm2[:, 1:-1] = y_init[3, :, :]
    Nc1 = y_init[4, :, :]
    Nc2 = y_init[5, :, :]
    Dc1 = y_init[6, :, :]
    Dc2 = y_init[7, :, :]
    I1  = y_init[8, :, :]
    I2  = y_init[9, :, :]
    H1  = y_init[10, :, :]
    for l in range(N-1):
        n1_ave = Nm1[1, 0:2] + Nm1[1, 2:]
        n2_ave = Nm2[1, 0:2] + Nm2[1, 2:]
        d1_ave = Dm1[1, 0:2] + Dm1[1, 2:]
        d2_ave = Dm2[1, 0:2] + Dm2[1, 2:]
        nm11 = Notchm1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        nm21 = Notchm2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        dm11 = Deltam1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        dm21 = Deltam2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        i11    = NICD1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        i21    = NICD2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        nc11 = Notchc1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        nc21 = Notchc2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        dc11 = Deltac1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        dc21 = Deltac2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        h11   = LCHes1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)

        nm12 = Notchm1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        nm22 = Notchm2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        dm12 = Deltam1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        dm22 = Deltam2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        i12    = NICD1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        i22    = NICD2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        nc12 = Notchc1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        nc22 = Notchc2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        dc12 = Deltac1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        dc22 = Deltac2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        h12   = LCHes1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)        
        
        nm13 = Notchm2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        nm23 = Notchm2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        dm13 = Deltam1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        dm23 = Deltam2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        i13    = NICD1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        i23    = NICD2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        nc13 = Notchc1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        nc23 = Notchc2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        dc13 = Deltac1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        dc23 = Deltac2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        h13   = LCHes1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)        
        
        nm14 = Notchm2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        nm24 = Notchm2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        dm14 = Deltam1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        dm24 = Deltam2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        i14    = NICD1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        i24    = NICD2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        nc14 = Notchc1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        nc24 = Notchc2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        dc14 = Deltac1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        dc24 = Deltac2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        h14   = LCHes1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        
        Nm1[:, 1:-1] = Nm1[:, 1:-1] + (s/6)*(nm11 + 2*nm12 + 2*nm13 + nm14)
        Nm2[:, 1:-1] = Nm2[:, 1:-1] + (s/6)*(nm21 + 2*nm22 + 2*nm23 + nm24)
        Dm1[:, 1:-1] = Dm1[:, 1:-1] + (s/6)*(dm11 + 2*dm12 + 2*dm13 + dm14)
        Dm2[:, 1:-1] = Dm2[:, 1:-1] + (s/6)*(dm21 + 2*dm22 + 2*dm23 + dm24)
        I1 = I1 + (s/6)*(i11 + 2*i12 + 2*i13 + i14)
        I2 = I2 + (s/6)*(i21 + 2*i22 + 2*i23 + i24)
        Nc1 = Nc1 + (s/6)*(nc11 + 2*nc12 + 2*nc13 + nc14)
        Nc2 = Nc2 + (s/6)*(nc21 + 2*nc22 + 2*nc23 + nc24)
        Dc1 = Dc1 + (s/6)*(dc11 + 2*dc12 + 2*dc13 + dc14)
        Dc2 = Dc2 + (s/6)*(dc21 + 2*dc22 + 2*dc23 + dc24)
        H1 = H1 + (s/6)*(h11 + 2*h12 + 2*h13 + h14)
    lc1[q] = H1[1, 1]
    lc2[q] = H1[1, 2]

# %%
#Set the result of the calculation with the parameters of the standard to 1
lc1 = lc1/lc1[10]
lc2 = lc2/lc2[10]

# %%
#Model-BT
beta2_1 = 5.0
beta2_2 = 8.0
for q in trange(m):
    alpha2 = change[q]
    Nm1[:, 1:-1] = y_init[0, :, :]
    Nm2[:, 1:-1] = y_init[1, :, :]
    Dm1[:, 1:-1] = y_init[2, :, :]
    Dm2[:, 1:-1] = y_init[3, :, :]
    Nc1 = y_init[4, :, :]
    Nc2 = y_init[5, :, :]
    Dc1 = y_init[6, :, :]
    Dc2 = y_init[7, :, :]
    I1  = y_init[8, :, :]
    I2  = y_init[9, :, :]
    H1  = y_init[10, :, :]
    for l in range(N-1):
        n1_ave = Nm1[1, 0:2] + Nm1[1, 2:]
        n2_ave = Nm2[1, 0:2] + Nm2[1, 2:]
        d1_ave = Dm1[1, 0:2] + Dm1[1, 2:]
        d2_ave = Dm2[1, 0:2] + Dm2[1, 2:]
        nm11 = Notchm1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        nm21 = Notchm2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        dm11 = Deltam1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        dm21 = Deltam2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        i11    = NICD1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        i21    = NICD2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        nc11 = Notchc1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        nc21 = Notchc2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        dc11 = Deltac1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        dc21 = Deltac2(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)
        h11   = BTHes1(Nm1[:, 1:-1], Nm2[:, 1:-1], Dm1[:, 1:-1], Dm2[:, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)

        nm12 = Notchm1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        nm22 = Notchm2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        dm12 = Deltam1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        dm22 = Deltam2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        i12    = NICD1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        i22    = NICD2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        nc12 = Notchc1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        nc22 = Notchc2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        dc12 = Deltac1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        dc22 = Deltac2(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)
        h12   = BTHes1(Nm1[:, 1:-1]+(s/2)*nm11, Nm2[:, 1:-1]+(s/2)*nm21, Dm1[:, 1:-1]+(s/2)*dm11, Dm2[:, 1:-1]+(s/2)*dm21, I1+(s/2)*i11, I2+(s/2)*i21, Nc1+(s/2)*nc11, Nc2+(s/2)*nc21, Dc1+(s/2)*dc11, Dc2+(s/2)*dc21, H1+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)        
        
        nm13 = Notchm2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        nm23 = Notchm2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        dm13 = Deltam1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        dm23 = Deltam2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        i13    = NICD1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        i23    = NICD2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        nc13 = Notchc1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        nc23 = Notchc2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        dc13 = Deltac1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        dc23 = Deltac2(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)
        h13   = BTHes1(Nm1[:, 1:-1]+(s/2)*nm12, Nm2[:, 1:-1]+(s/2)*nm22, Dm1[:, 1:-1]+(s/2)*dm12, Dm2[:, 1:-1]+(s/2)*dm22, I1+(s/2)*i12, I2+(s/2)*i22, Nc1+(s/2)*nc12, Nc2+(s/2)*nc22, Dc1+(s/2)*dc12, Dc2+(s/2)*dc22, H1+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)        
        
        nm14 = Notchm2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        nm24 = Notchm2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        dm14 = Deltam1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        dm24 = Deltam2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        i14    = NICD1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        i24    = NICD2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        nc14 = Notchc1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        nc24 = Notchc2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        dc14 = Deltac1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        dc24 = Deltac2(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        h14   = BTHes1(Nm1[:, 1:-1]+s*nm13, Nm2[:, 1:-1]+s*nm23, Dm1[:, 1:-1]+s*dm13, Dm2[:, 1:-1]+s*dm23, I1+s*i13, I2+s*i23, Nc1+s*nc13, Nc2+s*nc23, Dc1+s*dc13, Dc2+s*dc23, H1+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
        
        Nm1[:, 1:-1] = Nm1[:, 1:-1] + (s/6)*(nm11 + 2*nm12 + 2*nm13 + nm14)
        Nm2[:, 1:-1] = Nm2[:, 1:-1] + (s/6)*(nm21 + 2*nm22 + 2*nm23 + nm24)
        Dm1[:, 1:-1] = Dm1[:, 1:-1] + (s/6)*(dm11 + 2*dm12 + 2*dm13 + dm14)
        Dm2[:, 1:-1] = Dm2[:, 1:-1] + (s/6)*(dm21 + 2*dm22 + 2*dm23 + dm24)
        I1 = I1 + (s/6)*(i11 + 2*i12 + 2*i13 + i14)
        I2 = I2 + (s/6)*(i21 + 2*i22 + 2*i23 + i24)
        Nc1 = Nc1 + (s/6)*(nc11 + 2*nc12 + 2*nc13 + nc14)
        Nc2 = Nc2 + (s/6)*(nc21 + 2*nc22 + 2*nc23 + nc24)
        Dc1 = Dc1 + (s/6)*(dc11 + 2*dc12 + 2*dc13 + dc14)
        Dc2 = Dc2 + (s/6)*(dc21 + 2*dc22 + 2*dc23 + dc24)
        H1 = H1 + (s/6)*(h11 + 2*h12 + 2*h13 + h14)
    bt1[q] = H1[1, 1]
    bt2[q] = H1[1, 2]

# %%
#Set the result of the calculation with the parameters of the standard to 1
bt1 = bt1/bt1[10]
bt2 = bt2/bt2[10]

# %%
#making figure    
fig1 = plt.figure(figsize=(8,5))
ax1 = fig1.add_subplot(111)
ax1.plot(change, lc1, marker='o', markersize=10, color="g", label="Model-LC cell1")
ax1.plot(change, lc2, marker='o', mfc="white", markersize=10, color="g", label="Model-LC cell2")
ax1.plot(change, bt1, marker='o', color="r", markersize=10, label="Model-BT cell1")
ax1.plot(change, bt2, marker='o', mfc="white", color="r", markersize=10, label="Model-BT cell2")
ax1.set_xlabel(r"$\alpha_{2}$", fontsize=15)
ax1.set_ylabel(r"HES1$(t^*)$", fontsize=15)
ax1.legend(loc="best", fontsize=10)
ax1.set_xlim([change[0], change[m-1]])
ax1.set_ylim([0, 4.0])
fig1.savefig("2cell_hes_a2.png")
plt.show()
# %%
