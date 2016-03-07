"""
Ellipsoid fiting library Demo
-----------------------------
Demo of ellipsoid fitting functions. Generates an ellipsoid with given noise, orientation, center and size.
Then runs the data through an ellipsoid fit function displaying the results. Note that high noise will cause lls_fit to
become statistically inconsistent and lose accuracy. Later updates to the EllipsoidFit library will address this.

dependency on matplotlib:
	http://matplotlib.org/

Contact:
	joseph.chrismer@gmail.com
Project blog:
	http://aerialarithmetic.blogspot.com/

"""

import numpy as np
import math
import EllipsoidFit as E_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Generate base sphere using spherical coordinates
sample_length = 1000
azimuth = 2*math.pi*np.random.rand(sample_length,1)
e = 2*np.random.rand(sample_length,1)-1
elevation = np.zeros((sample_length,1))
for i in range(0,sample_length):
    elevation[i] = math.asin(e[i])

#Convert from spherical to cartesian
Sphere = np.zeros((3,sample_length))
radius = 1
for i in range(0, sample_length):
    Sphere[0][i] = radius * math.cos(elevation[i]) * math.cos(azimuth[i])       #X
    Sphere[1][i] = radius * math.cos(elevation[i]) * math.sin(azimuth[i])       #Y
    Sphere[2][i] = radius * math.sin(elevation[i])                              #Z

#Create noise
noise_std = .01
noise = np.random.randn(3,sample_length)*noise_std

#Force zero mean (done to avoid a bias being added on top of a user defined one)
Nx_mean = np.mean(noise[0,:])
Ny_mean = np.mean(noise[1,:])
Nz_mean = np.mean(noise[2,:])
noise[0,:] = noise[0,:] - Nx_mean
noise[1,:] = noise[1,:] - Ny_mean
noise[2,:] = noise[2,:] - Nz_mean

#Parameters to warp the sphere, offset it, add noise and rotate it
true_offset_x = 7
true_offset_y = 5
true_offset_z = 1

#Matrix which warps the sphere into an ellipsoid
scaling = np.zeros((3, 3))
np.fill_diagonal(scaling, [7.0,3.0,1.5])

#Rotation parameters
x_rotation = 20.0 * math.pi/180.0
y_rotation = 20.0 * math.pi/180.0
z_rotation = 20.0 * math.pi/180.0
sx = math.sin(x_rotation)
sy = math.sin(y_rotation)
sz = math.sin(z_rotation)
cx = math.cos(x_rotation)
cy = math.cos(y_rotation)
cz = math.cos(z_rotation)
Rx = np.matrix([ [1, 0, 0],
                 [0, cx, -sx],
                 [0, sx, cx] ])
Ry = np.matrix([ [cy, 0, -sy],
                 [0, 1, 0],
                 [sy, 0, cy] ])
Rz = np.matrix([ [cz, -sz, 0],
                 [sz, cz, 0],
                 [0, 0, 1] ])

#Calculate matrix which distorts the sphere
Distortion_matrix = np.dot(Rx,np.dot(Ry,Rz))
Distortion_matrix = np.dot(Distortion_matrix,scaling)

#Distort the sphere into an rotated ellipsoid
Ellipsoid = np.dot(Distortion_matrix,Sphere)

#Add noise and translate from origin
Ellipsoid[0,:] = Ellipsoid[0,:] + true_offset_x + noise[0,:]
Ellipsoid[1,:] = Ellipsoid[1,:] + true_offset_y + noise[1,:]
Ellipsoid[2,:] = Ellipsoid[2,:] + true_offset_z + noise[2,:]

#preform lls fiting
C0, A_inv = E_fit.lls_fit(Ellipsoid.transpose())

#Calibrate
Calib = np.zeros((3,sample_length))
Calib[0,:] = Ellipsoid[0,:] - C0[0]
Calib[1,:] = Ellipsoid[1,:] - C0[1]
Calib[2,:] = Ellipsoid[2,:] - C0[2]
Calib = np.dot(A_inv,Calib)

#Print results
print(A_inv)
print(C0)

#Visualize results
fig_sphere = plt.figure()
ax_sphere = fig_sphere.add_subplot(111, projection='3d')
#Convert Ellipsoid to array to prevent - TypeError: can't multiply sequence by non-int of type 'float'
ax_sphere.scatter(np.array(Sphere[0,:]), np.array(Sphere[1,:]), np.array(Sphere[2,:]))
ax_sphere.set_title('Generated Sphere')

fig_ellipsoid = plt.figure()
ax_ellipsoid = fig_ellipsoid.add_subplot(111, projection='3d')
ax_ellipsoid.scatter(np.array(Ellipsoid[0,:]), np.array(Ellipsoid[1,:]), np.array(Ellipsoid[2,:]))
ax_ellipsoid.set_title('Generated Ellipsoid (with errors)')

fig_calib = plt.figure()
ax_calib = fig_calib.add_subplot(111, projection='3d')
ax_calib.scatter(np.array(Calib[0,:]), np.array(Calib[1,:]), np.array(Calib[2,:]))
ax_calib.set_title('LLS Calibration results Data')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
