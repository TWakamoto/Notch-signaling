#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:16:07 2022

@author: wakamototamaki
"""

#Animation
#Non-small cell lung cancer

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import trange

N = 200000 #number of time step
T = 1000 #TIME
s = T/N #
Iy = 100 #number of cell (y-axis)
Ix = 100 #number of cell (x-axis)
ITVL = int(N/100)

# %%
e = 0.01
t = np.linspace(0, T, N+1)

#initial cndition
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

# %%
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
beta21 = 8.0 #notch1
beta22 = 5.0 #notch2
beta3 = 2.0
beta41 = 8.0 #delta1
beta42 = 8.0 #delta2
#parameter of Hes-1
nu0 = 0.5
nu1 = 5.0
nu2 = 25.0
nu3 = 5.0

# %%
#plot of initial condition
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Notch1 in membrane (Model-NSCLC)')
ims1 = []
im11 = plt.imshow(Nm1[1:-1, 1:-1], interpolation='nearest', animated=True, vmin=0, vmax=3, cmap='jet')
im12 = fig1.colorbar(im11, ax=ax1)
txt11 = ax1.text(0.1, -0.03, f't={t[0]:.2f}', transform=ax1.transAxes)
ims1.append([im11]+[txt11])

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
    return alpha1*d_ave1*nm1 + alpha2*d_ave2*nm1 - mu3*i1    
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

def Hes1(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d_ave1, d_ave2, n_ave1, n_ave2):
    return hes(i1, i2)-mu0*h

def RungeKutta(nm1, nm2, dm1, dm2, i1, i2, nc1, nc2, dc1, dc2, h, d1_ave, d2_ave, n1_ave, n2_ave):
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
    h12     = Hes1(nm1+(s/2)*nm11, nm2+(s/2)*nm21, dm1+(s/2)*dm11, dm2+(s/2)*dm21, i1+(s/2)*i11, i2+(s/2)*i21, nc1+(s/2)*nc11, nc2+(s/2)*nc21, dc1+(s/2)*dc11, dc2+(s/2)*dc21, h+(s/2)*h11, d1_ave, d2_ave, n1_ave, n2_ave)

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
    h13     = Hes1(nm1+(s/2)*nm12, nm2+(s/2)*nm22, dm1+(s/2)*dm12, dm2+(s/2)*dm22, i1+(s/2)*i12, i2+(s/2)*i22, nc1+(s/2)*nc12, nc2+(s/2)*nc22, dc1+(s/2)*dc12, dc2+(s/2)*dc22, h+(s/2)*h12, d1_ave, d2_ave, n1_ave, n2_ave)

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
    h14     = Hes1(nm1+s*nm13, nm2+s*nm23, dm1+s*dm13, dm2+s*dm23, i1+s*i13, i2+s*i23, nc1+s*nc13, nc2+s*nc23, dc1+s*dc13, dc2+s*dc23, h+s*h13, d1_ave, d2_ave, n1_ave, n2_ave)
    
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


# %% Runge-Kutta method

for l in trange(N-1):
    n1_ave = (Nm1[:-2, 1:-1] + Nm1[2:, 1:-1] + Nm1[1:-1, :-2] + Nm1[1:-1, 2:])/4
    n2_ave = (Nm2[:-2, 1:-1] + Nm2[2:, 1:-1] + Nm2[1:-1, :-2] + Nm2[1:-1, 2:])/4
    d1_ave = (Dm1[:-2, 1:-1] + Dm1[2:, 1:-1] + Dm1[1:-1, :-2] + Dm1[1:-1, 2:])/4
    d2_ave = (Dm2[:-2, 1:-1] + Dm2[2:, 1:-1] + Dm2[1:-1, :-2] + Dm2[1:-1, 2:])/4

    Nm1[1:-1, 1:-1] = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[0]
    Nm2[1:-1, 1:-1] = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[1]
    Dm1[1:-1, 1:-1] = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[2]
    Dm2[1:-1, 1:-1] = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[3]
    I1              = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[4]
    I2              = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[5]
    Nc1             = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[6]
    Nc2             = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[7]
    Dc1             = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[8]
    Dc2             = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[9]
    H1              = RungeKutta(Nm1[1:-1, 1:-1], Nm2[1:-1, 1:-1], Dm1[1:-1, 1:-1], Dm2[1:-1, 1:-1], I1, I2, Nc1, Nc2, Dc1, Dc2, H1, d1_ave, d2_ave, n1_ave, n2_ave)[10]
    if (l+1) % ITVL == 0:
    #plot
        im11 = plt.imshow(Nm1[1:-1, 1:-1], interpolation='nearest', vmin=0, vmax=3, animated=True, cmap='jet')
        txt11 = ax1.text(0.1, -0.03, f't={t[l+1]:.2f}', transform=ax1.transAxes)
        ims1.append([im11]+[txt11])

# %%
anim1 = animation.ArtistAnimation(fig1, ims1, interval=150)
anim1.save('NSCLC_100-100_Nm1.gif', writer="pillow")

plt.show()