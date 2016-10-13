'''
=====================================================================================
Python implementation of the ALS (Adujusted Least Square) ellipsoid fitting algorithm
=====================================================================================

Sources:
"Consistent least squares fitting of ellipsoids" by Ivan Markovsky, Alexander Kukush, and Sabine Van Huffel
http://eprints.soton.ac.uk/263295/1/ellest_comp_published.pdf

Note this is only for 3 dimensional cases (some of the vectors describing indecies are hard coded)

Author-
Joseph Chrismer

History:
V1.0 - 2/1/16

To do:
- Minimize over var (?)
- Check step (9) to be positive semi definite
'''

import numpy as np
import math


def tensor_function(k_index,input,var):
    if k_index == 0:
        return 1

    elif k_index == 1:
        return input

    elif k_index == 2:
        return (input**2 - var)

    elif k_index == 3:
        return (input**3 - 3*input*var)

    #K = 4 (default case)
    return (input**4 - 6*input**2*var + 3*var**2)

def als_fit(X,var):
    n = np.size(X,0)
    m = np.size(X,1)

    # (1) Form the Tensor
    ten = np.zeros((5,n,m))
    for k in range(0,5):
        for i in range(0,n):
            for l in range(0,m):
                ten[k,i,l] = tensor_function(k,X[i,l],var)

    # (2) Create nβ x 2 matrix M
    M1 = [1,1,2,1,2,3,1,2,3,0]
    M2 = [1,2,2,3,3,3,0,0,0,0]
    M = np.matrix([M1,M2]).transpose()
    n_beta = int((n+1)*n/2 + n + 1)

    # (3) Form the tensor R = nβ x nβ x n
    R = np.zeros((n_beta,n_beta,n))
    for p in range(0,n_beta):
        for q in range(p,n_beta):
            for i in range(0,n):
                #Python starts counting at zero so we need to offset by 1
                offset = i+1
                #Multiplying by 1 forces the type to be integer, otherwise '+' is interpreted as logical OR
                R[p,q,i] = 1*(M[p,0] == offset) + 1*(M[p,1] == offset) + 1*(M[q,0] == offset) +1*(M[q,1] == offset)

    # (4) Compute ηals
    eta_als = np.zeros((n_beta,n_beta))
    sum = 0
    product = 1

    for p in range(0,n_beta):
        for q in range (p,n_beta):
            for l in range(0,m):
                # Compute the product
                for i in range (0,n):
                    product = product * ( ten[R[p,q,i] , i, l] )

                # Update the sum
                sum = sum + product;
                product = 1;

            #Store value in eta_als
            eta_als[p,q] = sum
            sum = 0

    #Define D (usually computed on the fly, but here it's hard coded for ellipsoids)
    D = [1,3,4]

    # (6) Form the symmetric matrix Ψals
    psi_als = np.zeros((n_beta,n_beta))
    for p in range(0,n_beta):
        for q in range (0,n_beta):
            if q >= p:
                if((p in D) and (q in D)):
                    psi_als[p,q] = 4*eta_als[p,q]

                elif((p not in D) and (q not in D)):
                    psi_als[p,q] = eta_als[p,q]

                else:
                    psi_als[p,q] = 2*eta_als[p,q]
            # q < p
            else:
                psi_als[p,q] = psi_als[q,p]

    # (7) find vector and value of Ψals associated with the smallest eigenvalue (again hard coded for ellipsoid only)
    evals,evecs = np.linalg.eig(psi_als)

    #Note numpy's implementation of eig returns an UNORDERED array for it's eigien values
    min_index = np.argmin(evals)
    min_evec = evecs[:,min_index]

    #number of parameters in A
    na = (n+1)*n/2

    #Stack min_evec into a (corresponds to function -) Note that it is done COLUMNWISE w/res to an upper triangle
    a = np.matrix([ [min_evec[0], min_evec[1], min_evec[3]],
                    [0          , min_evec[2], min_evec[4]],
                    [0          , 0          , min_evec[5]] ])

    #Constuct parameters corresponds to the 9 standard components of an ellipsoid (linear, combination and scalar)
    a = a + a.transpose() - np.diag(np.diag(a))
    b = min_evec[6:9]
    d = min_evec[9]

    #(8) Find the center and parameter vector ah
    center = -1/2*np.dot(np.linalg.inv(a),b)
    center_a_centerT = np.dot(np.dot(center,a),center.transpose())

    #Numpy doesnt type a 1x1 matrix into a scalar so center_a_centerT[0,0] has to be used
    ah =  (1/(center_a_centerT[0,0]-d))*a

    #(9) Check for PSD (ADD LATER)

    #(10) Find transformation matrix which maps the given ellipsoid to a sphere
    evals,Q = np.linalg.eig(ah)

    #find the radius (Numpy preforms pointer assignment on '=' so radii needs to be instantiated first)
    radii = np.zeros((3,1))
    for i in range(0,3):
        radii[i] = math.sqrt(1/evals[i])

    min_axis = np.min(radii)
    D_U = np.diag( [math.sqrt(evals[0]),math.sqrt(evals[1]),math.sqrt(evals[2])] )
    D_U = D_U*min_axis

    #Calculate the transformation matrix ALS_inv
    Q_DU = np.dot(Q,D_U)
    ALS_inv = np.dot(Q_DU,(np.linalg.inv(Q)))

    return ALS_inv,center
