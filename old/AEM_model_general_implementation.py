# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:50:26 2025

@author: Anton
"""

import numpy as np
np.set_printoptions(precision=9)
np.seterr(divide='ignore', invalid='ignore', over='ignore')                #dont print warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import mathieu_functions_OG as mf
import timeit
from datetime import timedelta
import multiprocessing as mp

#%% Fixed Domain Parameters
# Parameter
alpha_l = 10
alpha_t = 1
beta = 1 / (2 * alpha_l)
gamma = 3.5
Ca = 8
n = 7
M = 100
#%% Wrapper xy to eta psi
def uv(x, y, d):
    Y = np.sqrt(alpha_l / alpha_t) * y
    B = x**2 + Y**2 - d**2
    p = (-B + np.sqrt(B**2 + 4 * d**2 * x**2)) / (2 * d**2)
    q = (-B - np.sqrt(B**2 + 4 * d**2 * x**2)) / (2 * d**2)

    psi = None
    psi_0 = np.arcsin(np.sqrt(p))

    if Y >= 0 and x >= 0:
        psi = psi_0
    if Y < 0 and x >= 0:
        psi = np.pi - psi_0
    if Y <= 0 and x < 0:
        psi = np.pi + psi_0
    if Y > 0 and x < 0:
        psi = 2 * np.pi - psi_0

    eta = 0.5 * np.log(1 - 2 * q + 2 * np.sqrt(q**2 - q))
    return (eta, psi)

#%%
def Mathieu(order, psi, eta, q):
    m = mf.Mathieu(q)
    Se = m.ce(order, psi).real
    So = m.se(order, psi).real
    Ye = m.Ke(order, eta).real
    Yo = m.Ko(order, eta).real
    return (Se, So, Ye, Yo)

def compute_q(r):
    d = np.sqrt((r * np.sqrt(alpha_l / alpha_t))**2 - r**2)
    q = (d**2 * beta**2) / 4
    return (d, q)
#%% Element function accounting for influence of other elements
def Element(C, r, x, y, elements_list, index):
    """
    Computes the element contribution while accounting for influence from all other elements.
    """
    # d = np.sqrt((r * np.sqrt(alpha_l / alpha_t))**2 - r**2)
    # q = (d**2 * beta**2) / 4
    (d, q) = compute_q(r)     

    phi = np.linspace(0, 2 * np.pi, M)
    x_vals = x + (r * np.cos(phi))
    y_vals = y + (r * np.sin(phi))

    uv_vec = np.vectorize(uv)
    eta = uv_vec(x_vals, y_vals, d)[0]
    psi = uv_vec(x_vals, y_vals, d)[1]

    # Compute F_target for each element: This will be of size (M,)
    F_target = (C * gamma + Ca) * np.exp(-beta * x_vals)

    # Initialize influence matrix for this element
    F = []
    # Element's contribution for each order
    F.append(Mathieu(0, psi, eta, q)[0] * Mathieu(0, psi, eta, q)[2])

    for w in range(1, n):
        even = Mathieu(w, psi, eta, q)[1] * Mathieu(w, psi, eta, q)[3]
        odd = Mathieu(w, psi, eta, q)[0] * Mathieu(w, psi, eta, q)[2]
        F.append(even)
        F.append(odd)

    # Convert F to a numpy array with shape (M, N) where N is the number of terms
    F_array = np.array(F).T  # Transpose so we get (M, N)
    # print(f"Shape of F_array for element {index}: {F_array.shape}")

    # Now, for each other element, we need to consider their influence.
    # For simplicity, we append the influence of other elements directly in the matrix.
    for i, other_params in enumerate(elements_list):
        if i == index:
            continue  # Skip the current element itself

        # Calculate the contribution of the other element
        other_C, other_r, other_x, other_y = other_params
        # Here, we reuse the same structure as above to account for mutual influence:
        d_other = np.sqrt((other_r * np.sqrt(alpha_l / alpha_t))**2 - other_r**2)
        q_other = (d_other**2 * beta**2) / 4

        eta_other = uv_vec(x_vals, y_vals, d_other)[0]
        psi_other = uv_vec(x_vals, y_vals, d_other)[1]

        F_other = []
        F_other.append(Mathieu(0, psi_other, eta_other, q_other)[0] * Mathieu(0, psi_other, eta_other, q_other)[2])

        for w in range(1, n):
            even = Mathieu(w, psi_other, eta_other, q_other)[1] * Mathieu(w, psi_other, eta_other, q_other)[3]
            odd = Mathieu(w, psi_other, eta_other, q_other)[0] * Mathieu(w, psi_other, eta_other, q_other)[2]
            F_other.append(even)
            F_other.append(odd)

        F_other_array = np.array(F_other).T
        # Append the influence of the other element's terms
        F_array = np.hstack([F_array, F_other_array])  # Horizontally stack contributions

    return F_array, F_target, F

#%% Build the full coupled system
def BuildSystem(elements_list):
    MathieuSeriesExpansion = []
    BoundaryConditions = []

    # Loop through each element and compute F and F_target
    for idx, element_params in enumerate(elements_list):
        F, F_target, Influence_function = Element(*element_params, elements_list, idx)  # Call Element for each set of parameters
        MathieuSeriesExpansion.append(F)  # Append the F array (shape: M, N)
        BoundaryConditions.append(F_target)  # Append the corresponding F_target array (shape: M,)

    # Stack the F matrices from all elements (now considering all elements' influence)
    MathieuSeriesExpansion = np.vstack(MathieuSeriesExpansion)  # Stack to get (M * num_elements, N)
    BoundaryConditions = np.concatenate(BoundaryConditions)  # Concatenate to get (M * num_elements,)

    # print(f"Shape of MathieuSeriesExpansion: {MathieuSeriesExpansion.shape}")
    # print(f"Shape of BoundaryConditions: {BoundaryConditions.shape}")

    return MathieuSeriesExpansion, BoundaryConditions, Influence_function

# Solve the coupled system
def SolveSystem(elements_list):
    MathieuSeriesExpansion, BoundaryConditions, Influence_function = BuildSystem(elements_list)

    # Solve least squares
    Coeff = np.linalg.lstsq(MathieuSeriesExpansion, BoundaryConditions, rcond=None)
    return Coeff[0]

# Example usage: Creating multiple coupled Element calls
elements_list = [
    (10, 1, 0, 0),  # Element 1 parameters
    #(10, 1, 50, 20),  # Element 2 parameters
]
result = Element(10,1,1,1,elements_list,0)
coefficients = SolveSystem(elements_list)
print(coefficients)

print(Mathieu(1,uv(1,0,compute_q(1)[1])[1],uv(1,0,compute_q(1)[0])[0],compute_q(1)[1])) #--> Se and So problematic
#%% Apply `c(x, y)` function to grid
#comprehensive solution
def c(x, y):
    # if element_type == 'circle':
    #     if (x**2+y**2)<=r**2:
    #         return C0
    # if element_type == 'line':
    #     if x == 0 and -r < y < r:
    #         return C0

    eta = uv(x, y, compute_q(1)[0])[0]
    psi = uv(x, y, compute_q(1)[0])[1]
    q = compute_q(1)[1]
    F = coefficients[0]*Mathieu(0, psi, eta, q)[0]*Mathieu(0, psi, eta, q)[2]
    midpoint = (n-1)  # Integer division ensures a valid range

    for w in range(1, midpoint + 1):
        term = coefficients[w] * Mathieu(0, psi, eta, q)[0] * Mathieu(0, psi, eta, q)[2]
        if np.isnan(term):
            print(f"NaN detected at w={w} in first loop!")
        F += term

    for w in range(midpoint +1, 2*n-1):
        term = coefficients[w] * Mathieu(0, psi, eta, q)[1] * Mathieu(0, psi, eta, q)[3]
        if np.isnan(term):
            print(f"NaN detected at w={w} in second loop!")
        F += term

    if ((F*np.exp(beta*x)))> Ca:
        return ((((F*np.exp(beta*x)))-Ca)/gamma).round(9)
    else:
        return ((F*np.exp(beta*x))-Ca).round(9)

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

    result = Conc_array(10, 20, -5, 6, inc)

    stop = timeit.default_timer()
    sec = int(stop - start)
    cpu_time = timedelta(seconds = sec)
    print('Computation time [hh:mm:ss]:', cpu_time)

    plt.figure(figsize=(16, 9), dpi = 300)
    mpl.rcParams.update({'font.size': 22})
    plt.axis('scaled')
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.xticks(range(len(result[0]))[::int(50/inc)], result[0][::int(50/inc)].round())
    plt.yticks(range(len(result[1]))[::int(10/inc)], result[1][::int(10/inc)].round())
    Plume_max = plt.contour(result[2], levels=[0], linewidths=1, colors='k')
    plt.show()
