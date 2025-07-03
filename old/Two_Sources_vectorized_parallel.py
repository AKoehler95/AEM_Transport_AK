#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 09:48:56 2022

@author: anton
"""
import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')                #dont print warnings
np.set_printoptions(precision=9)
import matplotlib as mpl
import matplotlib.pyplot as plt
import mathieu_functions_OG as mf
import timeit
from datetime import timedelta
import multiprocessing as mp

#Parameter
alpha_l = 10
alpha_t = 1
beta = 1/(2*alpha_l)
C0 = 10
C1 = 10                #-8 for acceptor source
Ca = 8
gamma = 3.5
r = 1

d = np.sqrt((r*np.sqrt(alpha_l/alpha_t))**2-r**2)
q = (d**2*beta**2)/4

n = 7            #Number of terms in mathieu series
M = 100           #Number of Control Points, 5x overspecification

#Mathieu Functions
m = mf.Mathieu(q)

#Real Mathieu Functions
def Se(order, psi):                    #even angular first kind
    return m.ce(order, psi).real
def So(order, psi):                    #odd angular first kind
    return m.se(order, psi).real
def Ye(order, eta):                    #even radial second Kind
    return m.Ke(order, eta).real
def Yo(order, eta):                    #odd radial second Kind
    return m.Ko(order, eta).real

#wrapper xy to eta psi
def uv(x, y):
    Y = np.sqrt(alpha_l/alpha_t)*y
    B = x**2+Y**2-d**2
    p = (-B+np.sqrt(B**2+4*d**2*x**2))/(2*d**2)
    q = (-B-np.sqrt(B**2+4*d**2*x**2))/(2*d**2)

    psi = np.nan
    psi_0 = np.arcsin(np.sqrt(p))
    if Y >= 0 and x >= 0:
        psi = psi_0
    if Y < 0 and x >= 0:
        psi = np.pi-psi_0
    if Y <= 0 and x < 0:
        psi = np.pi+psi_0
    if Y > 0 and x < 0:
        psi = 2*np.pi-psi_0
    eta = 0.5*np.log(1-2*q+2*np.sqrt(q**2-q))
    return (eta, psi)

#polar coordinates
phi = np.linspace(0, 2*np.pi, M)
x1 = r*np.cos(phi)
y1 = r*np.sin(phi)

#source coordinates xy and distance of second source
D1 = 50
D2 = 20
x2 = x1 - D1
y2 = y1 - D2
x3 = x1 + D1
y3 = y1 + D2

#source coordinates eta psi
uv_vec = np.vectorize(uv)
psi1 = uv_vec(x1, y1)[1]
psi2 = uv_vec(x2, y2)[1]
psi3 = uv_vec(x3, y3)[1]
eta1 = uv_vec(x1, y1)[0]
eta2 = uv_vec(x2, y2)[0]
eta3 = uv_vec(x3, y3)[0]

#%%
#general target function:
def F_target(x, Ci) -> tuple[float, str]:
    if Ci > 0:
        return (Ci*gamma+Ca)*np.exp(-beta*x), 'r'
    else:
        return (Ci)*np.exp(-beta*x), 'b'

#System of Equations to calculate coefficients
#"perspective" source 1
lst = []                                #empty array

for i in range(0, M):                                    #filling array with all terms of MF for 1st source
    for j in range(0, 1):
        lst.append(Se(j, psi1[i])*Ye(j, eta1[i]))
    for j in range(1, n):
        lst.append(So(j, psi1[i])*Yo(j, eta1[i]))
        lst.append(Se(j, psi1[i])*Ye(j, eta1[i]))
    for j in range(0, 1):                                #filling array with all terms of MF for 2nd source
        lst.append(Se(j, psi2[i])*Ye(j, eta2[i]))
    for j in range(1, n):
        lst.append(So(j, psi2[i])*Yo(j, eta2[i]))
        lst.append(Se(j, psi2[i])*Ye(j, eta2[i]))

F_M1 = []
s = (2*n-1)*2
for k in range(0, len(lst), s):           #appending each line (s elements) as arrays (in brackets) -> achieve right array structure (list of arrays)
    F_M1.append(lst[k:k+s])

#"perspective" source 2
lst2 = []

for i in range(0, M):
    for j in range(0, 1):                                   #filling array with all terms of MF for 1st source
        lst2.append(Se(j, psi3[i])*Ye(j, eta3[i]))
    for j in range(1, n):
        lst2.append(So(j, psi3[i])*Yo(j, eta3[i]))
        lst2.append(Se(j, psi3[i])*Ye(j, eta3[i]))
    for j in range(0, 1):                                   #filling array with all terms of MF for 2nd source
        lst2.append(Se(j, psi1[i])*Ye(j, eta1[i]))
    for j in range(1, n):
        lst2.append(So(j, psi1[i])*Yo(j, eta1[i]))
        lst2.append(Se(j, psi1[i])*Ye(j, eta1[i]))

#%%
F_M2 = []
s = (2*n-1)*2
for k in range(0, len(lst2), s):          #appending each line (s elements) as lists (in brackets) -> achieve right array structure (list of arrays)
    F_M2.append(lst2[k:k+s])

F_M = F_M1 + F_M2                       #combining arrays for "perspective"" 1 and 2

F = []                                  #target function vector

for u in range(0, M):
    F.append(F_target(x1[u], C0)[0])
for u in range(0, M):
    F.append(F_target(x3[u], C1)[0])

Coeff = np.linalg.lstsq(F_M, F, rcond=None)
print(Coeff[0])

#%%
def c(x, y):
    if (x**2+y**2)<=r**2:
        return C0
    if ((x-D1)**2+(y-D2)**2)<=r**2:
        return C1

    psi = uv(x, y)[1]
    eta = uv(x, y)[0]
    psi2 = uv(x-D1, y-D2)[1]
    eta2 = uv(x-D1, y-D2)[0]

    F1 = Coeff[0][0]*Se(0, psi)*Ye(0, eta)
    for w in range(1, n):
        F1 += Coeff[0][2*w-1]*So(w, psi)*Yo(w ,eta) \
            + Coeff[0][2*w]*Se(w, psi)*Ye(w, eta)

    F2 = Coeff[0][2*n-1]*Se(0, psi2)*Ye(0, eta2)
    for b in range(1, n):
        F2 += Coeff[0][(2*n-1)+(2*b-1)]*So(b, psi2)*Yo(b, eta2) \
            + Coeff[0][(2*n-1)+(2*b)]*Se(b, psi2)*Ye(b, eta2)                   #till here F domain

    # return (F1*np.exp(beta*x) + F2*np.exp(beta*x)).round(9)              #from here C domain

    if (F1*np.exp(beta*x) + F2*np.exp(beta*x))> Ca:
        return (((F1*np.exp(beta*x) + F2*np.exp(beta*x))-Ca)/gamma).round(9)
    else:
        return ((F1*np.exp(beta*x) + F2*np.exp(beta*x))-Ca).round(9)
#%%
# #concentration array for plotting purpose

inc = 1
# Define a helper function for `Pool.map`
def compute_conc(point):
    x, y = point
    return c(x, y)

def Conc_array(x_min, x_max, y_min, y_max, inc):
    xaxis = np.arange(x_min, x_max, inc)
    yaxis = np.arange(y_min, y_max, inc)
    X, Y = np.meshgrid(xaxis, yaxis)

    # Flatten the grid for parallel processing
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Prepare the inputs as tuples of (x, y) for the function `c`
    points = list(zip(X_flat, Y_flat))

    # Use multiprocessing.Pool to parallelize the computation of `c(x, y)` over all grid points
    with mp.Pool(mp.cpu_count()) as pool:
        Conc_flat = pool.map(compute_conc, points)

    # Reshape the flat result back to the original grid shape
    Conc = np.array(Conc_flat).reshape(X.shape)

    return xaxis, yaxis, Conc

# Run the function
if __name__ ==  '__main__':
    start = timeit.default_timer()

    result = Conc_array(0, 150+inc, -20, 30+inc, inc)

    stop = timeit.default_timer()
    sec = int(stop - start)
    cpu_time = timedelta(seconds = sec)
    print('Computation time [hh:mm:ss]:', cpu_time)
#%% plotting

    plt.figure(figsize=(16, 9), dpi = 300)
    mpl.rcParams.update({'font.size': 22})
    plt.axis('scaled')
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')

    plt.xticks(range(len(result[0]))[::int(50/inc)], result[0][::int(50/inc)])
    plt.yticks(range(len(result[1]))[::int(10/inc)], result[1][::int(10/inc)])
    Plume = plt.contourf(result[2], levels=10, cmap='coolwarm') #np.linspace(Ca, 43, 10)
    Plume_max = plt.contour(result[2], levels=[0], linewidths=2, colors='k')

    Source0 = plt.Circle((result[0].tolist().index(0), result[1].tolist().index(0)), 2, color = F_target(x1[u], C0)[2])         #adding circles in the plot
    Source1 = plt.Circle((result[0].tolist().index(0)+(D1/inc), result[1].tolist().index(0)+(D2/inc)), 2, color = F_target(x3[v], C1)[2])   #adding circles in the plot
    plt.gca().add_patch(Source0)
    plt.gca().add_patch(Source1)

    #Colorbar
    norm= mpl.colors.Normalize(vmin=Plume.cvalues.min(), vmax=Plume.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap = Plume.cmap)
    sm.set_array([])
    plt.colorbar(Plume, ticks=Plume.levels, label='Concentration [mg/l]', location='bottom', aspect=75)


    # Label = '$C_{D}=C_{A}=0$'
    # Lmax = Plume.get_paths()[0]
    # #plt.clabel(Plume, fmt=Label, manual = [(50, -(2*np.max(result[1])*inc-np.abs(result[0][0])))])
    # #print('Lmax =', int(np.max(Lmax.vertices[:, int((result[1][0]+result[1][-1])/2)])*inc-np.abs(result[0][0]))) #
    # textbox = r'$L_{max} = 549 m$' #+ str(int(np.max(Lmax.vertices[:, int((result[1][0]+result[1][-1])/2)])*inc-np.abs(result[0][0]))) + ' m'
    # plt.text(200, 30, textbox)
    plt.tight_layout()
    plt.savefig('fig52.pdf')
    plt.show()

#%%
#absolut error [mg/l]
    phi2 = np.linspace(0, 2*np.pi, 360)
    x_test = (r + 1e-9) * np.cos(phi2)
    y_test = (r + 1e-9) * np.sin(phi2)

    x2_test = x_test+D1
    y2_test = y_test+D2

    Err = []
    Err2 = []
    for i in range(0,360,1):
        Err.append((c(x_test[i], y_test[i])))
        Err2.append((c(x2_test[i], y2_test[i])))
    #print(Err)
    #print(Err2)
    print('Min =',np.min(Err).round(9), 'mg/l')
    print('Max =',np.max(Err).round(9), 'mg/l')
    print('Mean =',np.mean(Err).round(9), 'mg/l')
    print('Standard Deviation =',np.std(Err).round(9), 'mg/l')
    print('Min2 =',np.min(Err2).round(9), 'mg/l')
    print('Max2 =',np.max(Err2).round(9), 'mg/l')
    print('Mean2 =',np.mean(Err2).round(9), 'mg/l')
    print('Standard Deviation2 =',np.std(Err2).round(9), 'mg/l')

    plt.figure(figsize=(16,9), dpi=300)
    mpl.rcParams.update({'font.size': 22})
    plt.plot(phi2,Err, color='black', linewidth=2, label='element 1')
    plt.plot(phi2,Err2, color='black', linewidth=2, linestyle = '--', label='element 2')
    plt.xlim([0, 2*np.pi])
    plt.xlabel('Angle (°)')
    plt.ylabel('Concentration (mg/l)')
    plt.xticks(np.linspace(0, 2*np.pi, 13), np.linspace(0, 360, 13).astype(int))
    plt.legend()
