'''
    Gradient descent based Extended Kalman Filter for quaternion state estimation
    --------------------------------------------------------------------------------------------------------------------

    features:
        Adaptive
        Doest use any Matrix inverse

    Resources:
        General idea behind gradient descent for Kalman Filter
        [1] "An Extended Kalman Filter for Quaternion-Based Orientation Estimation Using MARG Sensors" by Marins et. al.

        Gradient descent cost functions
        [2] "An efficient orientation filter for inertial and inertial/magnetic sensor arrays" by Sebastian Madgwick"

        State initialization from (static) accelerometer and magnitometer measurements
        [3] "Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs" by  Valenti et. al.

        [1] : http://calhoun.nps.edu/bitstream/handle/10945/41567/IROS2001.pdf?sequence=1
        [2] : http://www.x-io.co.uk/res/doc/madgwick_internal_report.pdf
        [3] : http://www.mdpi.com/1424-8220/15/8/19302

    Author:
        Joseph Chrismer

    Version:
        1.0 (release)
'''


import numpy as np
import math
import Misc_Kinematics as Kin
class EKF_AHRS(object):
    def __init__(self, P0, Q, R):
        '''
            Class constructer

            @Input
            P0 : ndarray
                4x4 state covariance matrix
            Q : ndarray
                4x4 (diagonal) process covariance matrix
            R : ndarray
                4x4 (diagonal) measurement covariance matrix

        '''

        # KF covariance matricies
        self.P = P0
        self.Q = Q
        self.R = R

        # State quaternion unit vector vector x = [qr, qx, qy, qz]
        self.x = np.array([1.0, 0.0, 0.0, 0.0])                 # Initialized later using

        # Adaptive process and measurement covariance variables
        self.use_adaptive_gyro = True
        self.use_adaptive_accel = True

        # Auxilary variables
        self.is_init = False
        self.mu = 0                                             # learning rate (used in gradient descent)
        self.accel_base = 1.1
        self.gyro_base = 2.4

    def init_states(self,a,m):
        '''
            Initialize states using (static) unit vector for gravity (a) and magnetometer (m) measurements. Stores
            results in self.x

            @Input
            a : list
                3x1 list containing normalized accelerometer data in the form [x, y, z]
            m : list
                3x1 list containing normalized magnetometer data in the form [x, y, z]

        '''

        self.x[0:4] =Kin.init_attitude(a,m)
        self.is_init = True

    # Kalman filter state prediction
    def predict(self,gyro,dt):
        '''
            Extended Kalman Filter prediction phase. Uses gyroscope input and sampling time to propogate the state
            forward.

            @Input
            gyro : list
                3x1 list containing gyroscope data (rad/s) in the form [x, y, z]
            dt : float
                time interval between updates (seconds)
        '''

        wx = gyro[0]
        wy = gyro[1]
        wz = gyro[2]

        scale = 1
        if(self.use_adaptive_gyro):
            magnitude = math.sqrt(wx*wx+ wy*wy + wz*wz)
            scale = self.gyro_base**(magnitude)                 # Magnetometer adaptive gain

        # Predict state using simple integration
        dq = 0.5 * np.array( Kin.quat_product(self.x,[0.0, wx, wy, wz]))
        self.x += dq*dt
        self.x = self.x * 1/np.linalg.norm(self.x)

        #calculate learning rate using quaterion rate
        dqnorm = np.linalg.norm(dq)
        self.mu = 10*dqnorm*dt

        # Predict state covariance
        hdt = dt/2.0

        # Jacobain of state transition function
        F = np.array([[ 1.0    , -hdt*wx, -hdt*wy, -hdt*wz],
                      [hdt*wx, 1.0      , hdt*wz , -hdt*wy],
                      [hdt*wy, -hdt*wz, 1.0      ,  hdt*wx],
                      [hdt*wz, hdt*wy , -hdt*wx  ,  1.0     ]], np.float)

        # Project process covariance
        self.P = np.dot(F,self.P)
        self.P = np.dot(self.P,F.transpose())
        self.P = self.P + scale*self.Q

    def update(self,a,m,magnitude):
        '''
            Extended Kalman Filter update phase. Compares state predictions to normalized magnetometer and accelerometer
            data by projecting them into quaternions using gradient descent. Also uses adaptive filtering on
            accelerometer magnitude. Does not use a matrix inverse.

            @Input
            a : list
                3x1 list containing normalized accelerometer data in the form [x, y, z]
            m : list
                3x1 list containing normalized magnetometer data in the form [x, y, z]
            magnitude:
                Euclidean norm of accelrometer vector (g force)

            references:
                [2]
        '''

        # Run GD
        z = Kin.GD_meas2q(a,m,self.x,self.mu)
        scale = 1.0
        if(self.use_adaptive_accel):
            scale = self.accel_base**(magnitude -1.0)           # accelerometer adaptive gain

        # Measurement update without matrix inversion
        for i in range(0,4):
            P_z_ki = self.P[i][i] + scale*self.R[i][i]

            # Calculate Kalman gain K_k (4x1 vector)
            K_k = self.P[:,i]*1.0/P_z_ki                        # Operates on the ith COLUMNS of P
            self.P = self.P - np.outer(K_k,self.P[i,:])         # Operates on the ith ROWS of P

            # update state
            r = z[i] - self.x[i]                                # z and x are both quaternions
            self.x = self.x + K_k*r                             # Apply Kalman gain to the residual

        # Re-normalize state
        self.x = self.x* 1.0/np.linalg.norm(self.x)
