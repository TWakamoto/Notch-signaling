#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:33:08 2023

@author: wakamototamaki
"""

#Notch1-DLL1-NICD1_Notchc1_DLLc1_Notch2-DLL2-NICD2_Notchc2_DLLc2_hes1 model
#graph(bar)
#NSCLC type
# %%============================
import matplotlib.pyplot as plt
import numpy as np


# %%=========================
N = 60000 #時間刻み数
T = 600 #最終時刻
s = T/N #時間の刻み幅
Iy = 100 #細胞の数（たて）
Ix = 100 #細胞の数（よこ）
ITVL = int(N/10)
m = 10 #the number of patients

count1 = np.zeros((m, 7))
count2 = np.zeros((m, 7))

# %%==============================================
#Biochemicals
e = 0.01
Nm1 = np.zeros((Iy+2, Ix+2)) #Notch1 in membrane
Nm2 = np.zeros((Iy+2, Ix+2)) #Notch2 in membrane
Dm1 = np.zeros((Iy+2, Ix+2)) #DLL1 in membrane
Dm2 = np.zeros((Iy+2, Ix+2)) #DLL2 in membrane
Nc1 = e*np.random.rand(Iy, Ix) #Notch1 in cytosol
Nc2 = e*np.random.rand(Iy, Ix) #Notch2 in cytosol
Dc1 = e*np.random.rand(Iy, Ix) #DLL1 in cytosol
Dc2 = e*np.random.rand(Iy, Ix) #DLL2 in cytosol
I1  = e*np.random.rand(Iy, Ix) #NICD1
I2  = e*np.random.rand(Iy, Ix) #NICD2
H1  = e*np.random.rand(Iy, Ix) #Hes1

Nm1[1:-1, 1:-1] = e*np.random.rand(Iy, Ix)
Nm2[1:-1, 1:-1] = e*np.random.rand(Iy, Ix)
Dm1[1:-1, 1:-1] = e*np.random.rand(Iy, Ix)
Dm2[1:-1, 1:-1] = e*np.random.rand(Iy, Ix)


# %%=========================
H_0 = np.copy(H1)

# %%==========
#parameter
#binding rate
alpha1 = 6.0
alpha2 = 10.0
alpha3 = 8.0
alpha4 = 6.0
#basal decay
mu0 = 0.5 ###
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

p_1 = 0.0
p_23 = 0.0
p_b = 0.0

# %%======================================================================
#function
#activation and inhibition of Hes-1 by NICD
def hes(x1, x2):
    return nu0 + (nu2*(x1**2)/(nu1 + x1**2))*(1 - ((x2**2)/(nu3 + x2**2)))

#ODEs
def Notchm1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return -alpha1*d_ave1*nm1 - alpha2*d_ave2*nm1 - mu1*nm1 + (1-p_1)*gammaN*nc1
def Notchm2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return -alpha3*d_ave1*nm2 - alpha4*d_ave2*nm2 - mu1*nm2 + (1-p_1)*gammaN*nc2
    
def Deltam1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return -alpha1*n_ave1*dm1 - alpha3*n_ave2*dm1 - mu2*dm1 + gammaD*dc1    
def Deltam2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return -alpha2*n_ave1*dm2 - alpha4*n_ave2*dm2 - mu2*dm2 + gammaD*dc2
    
def NICD1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return (1-p_23)*(alpha1*d_ave1*nm1 + alpha2*d_ave2*nm1) - mu4*i1
def NICD2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return (1-p_23)*(alpha3*d_ave1*nm2 + alpha4*d_ave2*nm2) - mu3*i1
    
def Notchc1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return (1-p_b)*beta21*(i1**2)/(beta1 + i1**2) - (mu4 + (1-p_1)*gammaN)*nc1
def Notchc2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return (1-p_b)*beta22*(i2**2)/(beta1 + i2**2) - (mu4 + (1-p_1)*gammaN)*nc2
    
def Deltac1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return beta41/(beta3 + h**2) - (mu5 + gammaD)*dc1    
def Deltac2(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return beta42/(beta3 + h**2) - (mu5 + gammaD)*dc2

def LCHes1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return hes(i1, i2)-mu0*h

# %%==================================================================
#runge-kutta
def ud_m(func, m, u1, u2, c): #function of Notch and Delta in membrane
    r11 = func(m,  u1, u2, c)
    r12 = func(m+(s/2)*r11, u1, u2, c)
    r13 = func(m+(s/2)*r12, u1, u2, c)
    r14 = func(m+s*r13, u1, u2, c)
    return m + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
def ud_i(func, i, d1, d2, nm): #function of NICD
    r11 = func(i,  d1, d2, nm)
    r12 = func(i+(s/2)*r11, d1, d2, nm)
    r13 = func(i+(s/2)*r12, d1, d2, nm)
    r14 = func(i+s*r13, d1, d2, nm)
    return i + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
def ud_c(func, c, ih): #function of Notch and Delta in cytosol
    r11 = func(c,  ih)
    r12 = func(c+(s/2)*r11, ih)
    r13 = func(c+(s/2)*r12, ih)
    r14 = func(c+s*r13, ih)
    return c + (s/6)*(r11 + 2*r12 + 2*r13 + r14)
def ud_h(func, h, i1, i2): #function of HES-1
    r11 = func(h, i1, i2)
    r12 = func(h+(s/2)*r11, i1, i2)
    r13 = func(h+(s/2)*r12, i1, i2)
    r14 = func(h+s*r13, i1, i2)
    return h + (s/6)*(r11 + 2*r12 + 2*r13 + r14)

# %%==================================================================
#runge-kutta
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

# %%================================================================================
'''
calculation (untreated)
'''
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

# %%=====================================================================
hmax = np.max(H1-H_0)
ut1 = 100*(np.count_nonzero((H1-H_0)/hmax>0.5))/(Iy*Ix)
ut2 = 100*(np.count_nonzero((H1-H_0)/hmax>0.75))/(Iy*Ix)

print(hmax, ut1, ut2)

# %%====================================================
'''
calculation (treated)
'''
#sequence
Nm1 = np.zeros((m, Iy+2, Ix+2, 7)) #Notch1 in membrane
Nm2 = np.zeros((m, Iy+2, Ix+2, 7)) #Notch2 in membrane
Dm1 = np.zeros((m, Iy+2, Ix+2, 7)) #DLL1 in membrane
Dm2 = np.zeros((m, Iy+2, Ix+2, 7)) #DLL2 in membrane
Nc1 = np.zeros((m, Iy, Ix, 7)) #Notch1 in cytosol
Nc2 = np.zeros((m, Iy, Ix, 7)) #Notch2 in cytosol
Dc1 = np.zeros((m, Iy, Ix, 7)) #DLL1 in cytosol
Dc2 = np.zeros((m, Iy, Ix, 7)) #DLL2 in cytosol
I1  = np.zeros((m, Iy, Ix, 7)) #NICD1
I2  = np.zeros((m, Iy, Ix, 7)) #NICD2
H1  = np.zeros((m, Iy, Ix, 7)) #Hes1

# %%===================================================
#initial condition
e = 0.01
Nm1[:, 1:-1, 1:-1,:] = e*np.random.rand(m, Iy, Ix, 7)
Nm2[:, 1:-1, 1:-1,:] = e*np.random.rand(m, Iy, Ix, 7)
Dm1[:, 1:-1, 1:-1,:] = e*np.random.rand(m, Iy, Ix, 7)
Dm2[:, 1:-1, 1:-1,:] = e*np.random.rand(m, Iy, Ix, 7)
Nc1 = e*np.random.rand(m, Iy, Ix, 7)
Nc2 = e*np.random.rand(m, Iy, Ix, 7)
Dc1 = e*np.random.rand(m, Iy, Ix, 7)
Dc2 = e*np.random.rand(m, Iy, Ix, 7)
I1  = e*np.random.rand(m, Iy, Ix, 7)
I2  = e*np.random.rand(m, Iy, Ix, 7)
H1  = e*np.random.rand(m, Iy, Ix, 7)

# %%==============================
H_0 = np.copy(H1)

# %%=========================================
p_1  = np.zeros((m, Iy, Ix, 7))
p_23 = np.zeros((m, Iy, Ix, 7))
p_b  = np.zeros((m, Iy, Ix, 7))
r_1  = np.random.rand(m, Iy, Ix)
r_23 = np.random.rand(m, Iy, Ix)
r_b  = np.random.rand(m, Iy, Ix)
for i in range(7):
    if i == 0 or i == 3 or i == 4 or i == 6:
        p_1[:,:,:,i] = r_1
    if i == 1 or i == 3 or i == 5 or i == 6:
        p_23[:,:,:,i] = r_23
    if i == 2 or i == 4 or i == 5 or i == 6:
        p_b[:,:,:,i] = r_b

# %%=====================================================================================================
for l in range(N-1):    
    n1_ave = (Nm1[:, :-2, 1:-1, :] + Nm1[:, 2:,1:-1, :] + Nm1[:, 1:-1, :-2, :] + Nm1[:, 1:-1, 2:, :])/4
    n2_ave = (Nm2[:, :-2, 1:-1, :] + Nm2[:, 2:,1:-1, :] + Nm2[:, 1:-1, :-2, :] + Nm2[:, 1:-1, 2:, :])/4
    d1_ave = (Dm1[:, :-2, 1:-1, :] + Dm1[:, 2:,1:-1, :] + Dm1[:, 1:-1, :-2, :] + Dm1[:, 1:-1, 2:, :])/4
    d2_ave = (Dm2[:, :-2, 1:-1, :] + Dm2[:, 2:,1:-1, :] + Dm2[:, 1:-1, :-2, :] + Dm2[:, 1:-1, 2:, :])/4
    Nm1[:, 1:-1, 1:-1:, ] = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[0]
    Nm2[:, 1:-1, 1:-1:, ] = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[1]
    Dm1[:, 1:-1, 1:-1:, ] = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[2]
    Dm2[:, 1:-1, 1:-1:, ] = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[3]
    I1                    = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[4]
    I2                    = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[5]
    Nc1                   = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[6]
    Nc2                   = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[7]
    Dc1                   = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[8]
    Dc2                   = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[9]
    H1                    = RungeKutta(LCHes1, Nm1[:, 1:-1, 1:-1, :], Nm2[:, 1:-1, 1:-1, :], Dm1[:, 1:-1, 1:-1, :], Dm2[:, 1:-1, 1:-1, :], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[10]

# %%==========================================================================
H1 = (H1-H_0)/hmax
for i in range(m):
    for j in range(7):
        count1[i, j] = 100*(np.count_nonzero(H1[i, :, :, j]>0.5))/(Iy*Ix)
        count2[i, j] = 100*(np.count_nonzero(H1[i, :, :, j]>0.75))/(Iy*Ix)
count1 = count1/ut1
count2 = count2/ut2

# %%======================================
ave05 = np.ones(8)
SD05 = np.zeros(8)
ave075 = np.ones(8)
SD075 = np.zeros(8)

for j in range(1, 8):
    ave05[j]  = np.mean(count1[:, j-1])
    ave075[j] = np.mean(count2[:, j-1])
    SD05[j]   = np.std(count1[:, j-1])
    SD075[j]  = np.std(count2[:, j-1])

# %%====================================================================================================================================================
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(111)
label = ['untreated', r'$p_1$', r'$p_{23}$', r'$p_{\beta}$', r'$p_1+p_{23}$', r'$p_1+p_{\beta}$', r'$p_{23}+p_{\beta}$', r'$p_1+p_{23}+p_{\beta}$']
left = np.arange(8)
ax1.bar(left, ave075, yerr=SD075, capsize=5, tick_label=label)
ax1.axhline(y=1.0, linestyle='dashed', color='black')
fig1.savefig("patient_clv_LC_p.png")

# %%=================================================================================
fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(111)
ax2.bar(label, ave05, width=-0.4, yerr=SD05, capsize=5, align='edge', label=">0.5")
ax2.bar(label, ave075, width=0.4, yerr=SD075, capsize=5, align='edge', label=">0.75")
ax2.axhline(y=1.0, linestyle='dashed', color='black')
ax2.legend(loc="best", fontsize=10)
fig2.savefig("patient_clv_LC.png")

plt.show()
# %%
