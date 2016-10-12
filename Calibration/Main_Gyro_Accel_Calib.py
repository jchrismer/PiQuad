"""
Inertial sensors calibration
------------------------
Library containing functions which preforms ellipsoid fitting on given input.

History:
V1.0 - 10/12/16   (initial release)

To Do:
Add command line input and parsing

references:
Tedaldi, David, Alberto Pretto, and Emanuele Menegatti. "A robust and easy to implement method for IMU calibration without external e10)
quipments." 2014 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2014.

Contact:
	joseph.chrismer@gmail.com

Project blog:
	http://aerialarithmetic.blogspot.com/
"""


import Inertial_Calibration as IC
import numpy as np
import csv
import statistics
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt
import EllipsoidFit

def getData(filename):
    with open(filename, 'rt') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        x=list(csvReader)
        Data =np.array(x).astype('float')
        return Data

# csv file of sensor of recorded data
Accel = getData('/home/joseph/Desktop/Project/Matlab/Calibration/Finished_code/Calibration/Accel_demo.csv')
Gyro = getData('/home/joseph/Desktop/Project/Matlab/Calibration/Finished_code/Calibration/Gyro_demo.csv')

options = [50,1]
Sample_length = len(Accel)
Time = [1.0/100.0] * Sample_length
T = Accel[:,0]
Accel = Accel[:,1:4]
Gyro = Gyro[:,1:4]
i = 1
Valid_intervals_starts, Valid_intervals_ends,visualize,init_staitc = IC.static_invertal_detection(Gyro,Time,options,3.5)

# Pull out stati intervals and find their mean
num_static = len(Valid_intervals_starts)
Static_accel = np.zeros((3,num_static))

# lls fit (theoretically) needs at least 9 points to fit an ellipsoid too. More is better though
if(num_static < 9):
    print("Variance threshold multiplier: %d could not generate minimal required static intervals - SKIPPING" %i)

for i in range(0,num_static):
    ax = statistics.mean(Accel[Valid_intervals_starts[i]:(Valid_intervals_ends[i]+1),0])
    ay = statistics.mean(Accel[Valid_intervals_starts[i]:(Valid_intervals_ends[i]+1),1])
    az = statistics.mean(Accel[Valid_intervals_starts[i]:(Valid_intervals_ends[i]+1),2])

    Static_accel[:,i] = [ax,ay,az]

# Construct initial estimates for theta_accel using LLS fit on the static intervals

plt.plot(Accel[:,0])
plt.hold(True)
plt.plot(Accel[:,1],'r')
plt.plot(Accel[:,2],'g')
plt.plot(visualize,'k')
plt.show()

center,A_inv,radii = EllipsoidFit.lls_fit(Static_accel.transpose())
Grav = 9.81744
radii = [radii[0]/Grav,radii[1]/Grav,radii[2]/Grav]

# TODO: Implement LVM fitting for accelerometer and gyroscope
theta_acc = Parameters()
theta_acc.add('ax', value=0)
theta_acc.add('ay', value=0)
theta_acc.add('az', value=0)
theta_acc.add('kx', value=1.0/radii[0])
theta_acc.add('ky', value=1.0/radii[1])
theta_acc.add('kz', value=1.0/radii[2])
theta_acc.add('bx', value=center[0])
theta_acc.add('by', value=center[1])
theta_acc.add('bz', value=center[2])
out = minimize(IC.accel_resid, theta_acc, args=Static_accel)
#cost = out.chisqr

''' Gyroscope calibration '''
static_interval_count = len(Valid_intervals_starts)
# Creat storage list
Gyro_movements = (static_interval_count-1) * [[[]]]
for i in range(0,static_interval_count-1):
    gyro_start = Valid_intervals_ends[i]+1
    gyro_ends = Valid_intervals_starts[i+1]-1
    Gyro_movements[i] = Gyro[gyro_start:(gyro_ends),:]

#plt.plot(Gyro[:,0])
#plt.hold(True)
#plt.plot(Gyro[:,1],'r')
#plt.plot(Gyro[:,2],'g')
#plt.plot(visualize,'k')
#plt.show()
# Setup gryscope parameters
theta_gyro = Parameters()
theta_gyro.add('gamma_yz', value=0)
theta_gyro.add('gamma_zy', value=0)
theta_gyro.add('gamma_xz', value=0)
theta_gyro.add('gamma_zx', value=0)
theta_gyro.add('gamma_xy', value=0)
theta_gyro.add('gamma_yx', value=0)
theta_gyro.add('sx', value=1.0/6258.0)
theta_gyro.add('sy', value=1.0/6258.0)
theta_gyro.add('sz', value=1.0/6258.0)

# Get gyroscpoe bias from initial static interval
gbx = statistics.mean(Gyro[0:init_staitc,0])
gby = statistics.mean(Gyro[0:init_staitc,1])
gbz = statistics.mean(Gyro[0:init_staitc,2])
Gyro_intervals = IC.GyroData(Gyro_movements,gbx,gby,gbz)
Accel_intervals = IC.AccelData(Static_accel)
Accel_intervals.Accel = Accel_intervals.applyCalib(out.params,Static_accel)
out_gyro = minimize(IC.gyro_resid, theta_gyro, args=(Gyro_intervals,Accel_intervals,Time))
print(out_gyro.params)
print(out.params)