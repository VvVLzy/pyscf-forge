import ctypes
import math
import numpy as np
from . import CGmatrix

Lmax = CGmatrix.Lmax

Complex_Real = np.zeros(((2*Lmax)**2, (2*Lmax)**2), dtype=complex)
Real_Complex = np.zeros(((2*Lmax)**2, (2*Lmax)**2), dtype=complex)
RealCG = np.zeros((Lmax**2, Lmax**2, (2*Lmax)**2), dtype=complex)
CG = CGmatrix.CG
YYTrans = 0*CG
idx = lambda l, m : l*l + m + l
for j1 in range(Lmax):
    for m1 in range(-j1, j1+1):

        for j2 in range(Lmax):
            for m2 in range(-j2, j2+1):

                for j in range(2*Lmax):
                    for m in range(-j, j+1):
                        YYTrans[idx(j1,m1), idx(j2,m2), idx(j,m)] = CG[idx(j1,m1), idx(j2,m2), idx(j,m)] * \
                                                                    CG[idx(j1,0), idx(j2,0), idx(j,0)] * \
                                                                    ((2*j1+1)*(2*j2+1)/4/np.pi/(2*j+1))**0.5  

for j in range(2*Lmax):
    for m in range(-j,j+1):
        if (m < 0):
            Complex_Real[ idx(j,m), idx(j,abs(m))] = 1./2.**0.5 
            Complex_Real[ idx(j,m), idx(j,-abs(m))] = -1.j/2.**0.5 
        elif (m > 0):
            Complex_Real[ idx(j,m), idx(j,abs(m))] = (-1)**m/2.**0.5 
            Complex_Real[ idx(j,m), idx(j,-abs(m))] = (-1)**m*1.j/2.**0.5
        elif (m == 0):
             Complex_Real[ idx(j,m), idx(j,abs(m))] = 1. 

for j in range(2*Lmax):
    for m in range(-j,j+1):
        if (m < 0):
            Real_Complex[ idx(j,m), idx(j,m)] = 1.j/2.**0.5 
            Real_Complex[ idx(j,m), idx(j,-m)] = -1.j*(-1)**m/2.**0.5 
        elif (m > 0):
            Real_Complex[ idx(j,m), idx(j,-m)] = 1./2.**0.5 
            Real_Complex[ idx(j,m), idx(j,m)] = (-1)**m/2.**0.5
        elif (m == 0):
             Real_Complex[ idx(j,m), idx(j,abs(m))] = 1. 


CRsmall = Complex_Real[:Lmax**2,:Lmax**2]
RealCG = np.einsum('cr,ds,et,cde->rst', CRsmall.conj(), CRsmall.conj(), Complex_Real, YYTrans, optimize='optimal').real
