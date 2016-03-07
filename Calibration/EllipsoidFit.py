"""
Ellipsoid fiting library
------------------------
Library containing functions which preforms ellipsoid fitting on given input.

History:
V1.0 - 2/1/16   (initial release LLS method only)

To Do:
Implement Adjusted Least Square Fitting (ALS)

Contact:
	joseph.chrismer@gmail.com
Project blog:
	http://aerialarithmetic.blogspot.com/
"""

import numpy as np
import math

def lls_fit(Mag):
    """
    Preforms linear Least Squared Ellipsoid Fitting on given input. 3D only.

    @Input
    Mag : ndarray
        nx3 ndarray containing the data to be fitted. Each entry should have form [x, y, z]

    @Output
    C0 : matrix
        3x1 vector containing the centers of the ellipsoid estimated from Mag

    A_inv : matrix
        3x3 Scaling matrix which maps the estimated ellipsoid into a sphere with raduis
        equal to minimum principle axis.

    @Usage
        C0, A_inv = lls_fit(Data)
        Calibrated = A_inv * (Data - C0)

    @Refereneces
    This function is a variation of Yury Petrov's method:
    http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
    """

    n = np.size(Mag,0)

    #1. Construct nx9 matrix D fro magnetometer data
    Mag_X = Mag[:,0]
    Mag_Y = Mag[:,1]
    Mag_Z = Mag[:,2]

    #Columns of D correspond to: x^2 + y^2 + z^2 + 2xy + 2xz + 2yz + 2x + 2y + 2z
    D = np.zeros((n,9))

    for i in range(0,n):
        #Squared
        D[i,0] = Mag_X[i]**2
        D[i,1] = Mag_Y[i]**2
        D[i,2] = Mag_Z[i]**2

        #Combination
        D[i,3] = 2*Mag_X[i]*Mag_Y[i]
        D[i,4] = 2*Mag_X[i]*Mag_Z[i]
        D[i,5] = 2*Mag_Y[i]*Mag_Z[i]

        #Linear
        D[i,6] = 2*Mag_X[i]
        D[i,7] = 2*Mag_Y[i]
        D[i,8] = 2*Mag_Z[i]

    #2. Create ONE, an nx1 vector of ones
    ONE = np.ones(n)

    #3. Find the parameter vector P for the best fit ellipsoid E_F
    P = np.linalg.inv(np.dot(D.transpose(),D))     #inv(D' * D)
    P = np.dot(P,np.dot(D.transpose(),ONE))

    #4. Create matrix S and vector b from the parameters in P
    #P = [A B C D E F G H I]
    S = np.matrix([ [P[0], P[3], P[4]],
                    [P[3], P[1], P[5]],
                    [P[4], P[5], P[2]] ])

    b = [P[6], P[7], P[8]]

    #5. Find the center C0 of E_f using (2.8):
    C0 = -np.linalg.inv(S)                         #-inv(S)
    C0 = np.dot(C0,b).transpose()                  #-inv(S)*b

    #6. Calculate w_prime using (3.0)
    w_prime = np.dot(C0.transpose(),np.dot(S,C0))  #C0'*S*C0
    w_prime = w_prime + 2*np.dot(C0.transpose(),b) #C0'*S*C0 + 2*C0'*b
    #Numpy wont cast a matrix to a scalar
    w_prime =np.asscalar(w_prime[0]) - 1           #C0'*S*C0 + 2*C0'*b - 1

    #7. Find the coeficient matrix U describing Ef centered at the origin
    U = -1/w_prime * S

    #8-9. Find and the eigenvectors and eigenvalues of U in Q and D_u resp.
    D_U,Q = np.linalg.eig(U)

    #10. Calculate the inverse of the soft iron transformation matrix A
    radii = [1/math.sqrt(D_U[0]), 1/math.sqrt(D_U[1]), 1/math.sqrt(D_U[2])]
    min_radii = min(radii)

    Scale = np.matrix([ [math.sqrt(D_U[0]), 0, 0],
                        [0, math.sqrt(D_U[1]), 0],
                        [0, 0, math.sqrt(D_U[2])] ]) * min_radii

    A_inv = np.dot(np.dot(Q,Scale),np.linalg.inv(Q))

    return C0,A_inv

