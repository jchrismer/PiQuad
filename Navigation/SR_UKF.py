'''
    Quaternion based Unscented Kalman Filter (Square Root variation)
    --------------------------------------------------------------------------------------------------------------------

    Resources:
        Main source
        [1] "Unscented Filtering in a Unit Quaternion Space for Spacecraft Attitude Estimation" by Yee-Jin Cheon

    Author:
        Joseph Chrismer

    Version:
        1.0 (release)
'''

import numpy as np
import math
import Misc_Kinematics as Kin
class UKF_AHRS(object):
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

        # Dimensions
        self.n = 7               # State dimension
        self.num_meas = 6        # Measurement dimension
        self.adj_n = self.n-1    # Loss of 1 dof due to unit quaternion

        # Covariance/noise matricies
        self.Q = Q              # 6x6
        self.R = R              # 6x6
        self.P = P0             # 6x6
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Sy = np.zeros((self.num_meas, self.num_meas),dtype=np.double)    # Measurement covariance matrix
        self.Pxy = np.zeros((self.num_meas, self.num_meas),dtype=np.double)   # State and measurement cross covariance
        self.Q_root = np.linalg.cholesky(Q)
        self.R_root = R
        self.S = np.linalg.cholesky(P0)

        # Sigma point weighting
        self.alpha = .3          # .3 gives PSD
        self.beta = 2
        self.kappa = 0

        # Sigma point weights
        self.Wm = [0,0]          # Mean weight
        self.Wc = [0,0]          # Covariance weight
        self.Lambda = (self.alpha**2)*(self.adj_n + self.kappa) - self.adj_n
        self.num_sigmas = 2*self.adj_n+1
        self.Wm[1] = 0.5/(self.adj_n + self.Lambda)
        self.Wc[1] = self.Wm[1]
        self.Wc[0] = self.Lambda/(self.adj_n + self.Lambda) + 1 - self.alpha**2 + self.beta
        self.Wm[0] = self.Lambda/(self.adj_n + self.Lambda)
        self.sqr_gamma = math.sqrt(self.Lambda + self.adj_n)

        # State (f) and measurement (h) sigmapoints are stored in COLUMNS
        self.Sig_f = np.zeros((self.n, self.num_sigmas))
        self.Sig_h = np.zeros((self.num_meas, self.num_sigmas))

        # UT mean values
        self.x_mean = self.x
        self.y_mean = np.zeros(self.num_meas,dtype=np.double)

        # Auxillary variables
        self.isinit = False

    def init_attitude(self,a,m):
            ax = a[0]
            ay = a[1]
            az = a[2]

            # Estimate accelerometer based quaternion (tilt). Needs something to chose the appropriate sign based
            # on shorest path
            if (az>= 0):
                q0 = math.sqrt((az+1)/2.0)
                qacc = [q0, -ay/(2.0*q0), ax/(2*q0), 0.0]

            # az < 0
            else:
                q1 = math.sqrt((1 - az) * 0.5)
                qacc = [-ay/(2.0*q1),q1,0.0,ax/(2.0*q1)]

            # Estimate magnetometer based quaternion (yaw)
            l = Kin.quaternion_rotate([qacc[0],qacc[1],qacc[2],qacc[3]],m)
            lx = l[0]
            ly = l[1]
            Rho = math.sqrt(lx**2 + ly**2)
            Rho_sqrt = math.sqrt(Rho)

            if (lx>=0):
               qmag = [math.sqrt(Rho + lx*Rho_sqrt)/math.sqrt(2.0*Rho),0.0,0.0,ly/(math.sqrt(2.0)*math.sqrt(Rho + lx*Rho_sqrt))]
            #lx < 0
            else:
               qmag = [ly/(math.sqrt(2)*math.sqrt(Rho - lx*Rho_sqrt)),0.0,0.0,math.sqrt(Rho - lx*Rho_sqrt)/math.sqrt(2.0*Rho)]
            Q0 = Kin.quat_product(qacc,qmag)
            Q0 = np.multiply(Q0,1.0/np.linalg.norm(Q0))
            self.isinit = True
            self.x[0:4] = Q0


    def f(self,x,dt,U):

            '''
            State transition function. Projects the state vector forward given a time step and a gyroscope measurement.
            Uses a discretization of the continous time model:
            q' = 1/2 * w x q

            @Input
            U : list
                3x1 list containing gyroscope data (rad/s) in the form [x, y, z]
            dt : float
                time interval between measurements (seconds)
            x : list
                (unit) quaternion describing current orientation

            @return
               7x1 nparray containing the predicted state estimates of the form [qr, qx, qy, qz, wxb, wyb, wzb]
            '''

            q0 = x[0]
            q1 = x[1]
            q2 = x[2]
            q3 = x[3]
            Quat = np.array([q0,q1,q2,q3])

            # Bias is not modeled to change but is included anyways
            wxb = x[4]
            wyb = x[5]
            wzb = x[6]

            # Gyroscope readings
            wx = U[0]
            wy = U[1]
            wz = U[2]

            # Gyroscope magnitude
            W = np.array([(wx-wxb), (wy-wyb), (wz-wzb)])
            delta_w_mag = np.linalg.norm(W)

            Cw = math.cos(0.5*delta_w_mag*dt)
            Psi_w = (math.sin(0.5*delta_w_mag*dt)/delta_w_mag)*W

            Omega_W = np.matrix([ [Cw, 0, 0, Psi_w[0]],
                                  [0 ,Cw, 0, Psi_w[1]],
                                  [0 ,0 , Cw,Psi_w[2]],
                                  [-Psi_w[0] ,-Psi_w[1] ,-Psi_w[2],Cw]
                                ])

            new_quat = np.dot(Omega_W,Quat)
            return np.array([new_quat[0,0],new_quat[0,1],new_quat[0,2],new_quat[0,3],wxb,wyb,wzb])

    # State to measurement function
    def h(self,mag,x):

        '''
            Measurement update function. Uses [2] to predict normalized accelerometer and magnetometer measurements

            @Input
            mag : list
                3x1 list containing normalized magnetometer data in the form [x, y, z]
            x : list
                (unit) quaternion describing current orientation

            @return
                6x1 nparray containing the predicted measurements of the form [ax, ay, az, mx, my, mz]

        '''

        q0 = x[0]
        q1 = x[1]
        q2 = x[2]
        q3 = x[3]

        mx = mag[0]
        my = mag[1]
        mz = mag[2]

        # Auxiliary variables
        q0q0 = q0 * q0
        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q3q3 = q3 * q3

        # Translate magnetometer into Earth frame
        hx = 2.0 * (mx * (0.5 - q2q2 - q3q3) + my * (q1q2 - q0q3) + mz * (q1q3 + q0q2))
        hy = 2.0 * (mx * (q1q2 + q0q3) + my * (0.5 - q1q1 - q3q3) + mz * (q2q3 - q0q1))
        # Compute horizontal intensity
        bx = math.sqrt(hx * hx + hy * hy)
        bz = 2.0 * (mx * (q1q3 - q0q2) + my * (q2q3 + q0q1) + mz * (0.5 - q1q1 - q2q2))

        # Estimated direction of gravity and magnetic field
        ax_est = q1q3 - q0q2
        ay_est = q0q1 + q2q3
        az_est = q0q0 - 0.5 + q3q3

        # Rotate local magnetic field back into the body frame
        mx_est = bx * (0.5 - q2q2 - q3q3) + bz * (q1q3 - q0q2)
        my_est = bx * (q1q2 - q0q3) + bz * (q0q1 + q2q3)
        mz_est = bx * (q0q2 + q1q3) + bz * (0.5 - q1q1 - q2q2)

        return (np.multiply([ax_est,ay_est,az_est,mx_est,my_est,mz_est],2.0))

    def X_residual(self,left,right):
        '''
            State vector residual function. Used to find the difference between sigma points and state mean
                left - right
            Since both vectors contains quaternions quaternion multiplication must be used to keep
            errors consistent with quaternion space. Regular vector addition is used on the remaining non-quaternion
            elements

            @Input
            left : list
                7x1 list containing left state vector opperand
            right : list
                7x1 list containing right state vector opperand

            @return
                7x1 nparray containing the state residual

        '''
        right_conj = [right[0], -right[1], -right[2], -right[3]]
        Q_residual = Kin.quat_product(left,right_conj)

        # Subtract the rest
        remaining = left[4:self.n] - right[4:self.n]

        return np.concatenate((Q_residual[1:self.n],remaining))

    def vec2q(self,vin):
        '''
            Uses the vector componenet to create a (unit) quaternion. Used in sigma point selection from the process
            covariance matrix

            @Input
            vin : list
                vector portion of a (unit) quaternion in the form [qx, qy, qz]

            @return
                4x1 list representing the unit quaternion corresponding to the vector component [qr, qx, qy, qz]

        '''

        scalar_part = vin[0]**2 + vin[1]**2 + vin[2]**2

        if(scalar_part >= 1):
            scalar_part = 1
            vin = vin/np.linalg.norm(vin)

        return [math.sqrt(1 - scalar_part),vin[0],vin[1],vin[2] ]

    def UT_predict(self):
        '''
            Unscented transform used in the prediction step. Differs from the normal unscented transform in that
            quaternion multiplication is used to generate sigma points and find their mean and covariance. Square root
            formulation

            operates on class members - no input or output

        '''

        # Construct mean
        self.x_mean = self.Wm[0]*self.Sig_f[:,0]
        for i in range(1,self.num_sigmas):
            self.x_mean = self.x_mean + self.Wm[1]*self.Sig_f[:,i]

        # Normalize the barycentric mean of quaternions (41d)
        self.x_mean[0:4] = self.x_mean[0:4]/np.linalg.norm(self.x_mean[0:4])

        # The first sigma point is omitted, hence obj.num_sigmas-1 is used
        resid_mat_length = (self.num_sigmas-1 + self.adj_n)
        Residual_Matrix = np.zeros((self.adj_n,resid_mat_length),dtype=float)

        # Construct array of residuals
        residual = self.X_residual(self.Sig_f[:,0],self.x_mean)
        self.P = self.Wc[0]*np.outer(residual,residual.transpose())
        for k in range(1,self.num_sigmas):
            residual = math.sqrt(self.Wc[1])*self.X_residual(self.Sig_f[:,k],self.x_mean)
            Residual_Matrix[:,k-1] = residual

        # Augment the root of the state covariance matrix Q to the residual matrix
        Residual_Matrix[:,(resid_mat_length-6):resid_mat_length] = self.Q_root

        # Note: r is returned in UPPER triangle form
        q, r = np.linalg.qr(Residual_Matrix.transpose())

        mean_resid = self.X_residual(self.Sig_f[:,0],self.x_mean)
        wc_root = math.sqrt(abs(self.Wc[0]))

        # S is in upper triangle form in the current method
        rr = np.dot(r.transpose(),r)
        mean_resid_squared = np.sign(self.Wc[0]) * wc_root*np.outer(mean_resid,mean_resid.transpose())
        self.S = np.linalg.cholesky(rr + mean_resid_squared)
        self.x = self.x_mean

    def UT_update(self):
        '''
            Unscented transform used in the update step. Standard algorithm since updates happen in regular vector space.
            Square root version.

            operates on class members - no input or output

        '''

        self.y_mean = self.Wm[0]*self.Sig_h[:,0]
        # Compute sigma point mean
        for i in range(1,self.num_sigmas):
            self.y_mean = self.y_mean + self.Wm[1]*self.Sig_h[:,i]
        e_size = (self.num_meas, self.num_sigmas + self.num_meas)
        e = np.zeros(e_size)
        # Compute Measurement covariance
        residual = self.Sig_h[:,0] - self.y_mean   # measurements are in vector space so no special operations needed

        # residual matrix
        e[:,0] = residual
        for k in range(1,self.num_sigmas):
            residual = self.Sig_h[:,k] - self.y_mean
            e[:,k] = math.sqrt(self.Wc[1])*residual

        e[:,(e_size[1] - self.num_meas):e_size[1]] = self.R_root
        q, Sy = np.linalg.qr(e[:,1:e_size[1]].transpose())
        test = math.sqrt(self.Wc[1])*e[:,1:e_size[1]].transpose()
        wc_root = math.sqrt(abs(self.Wc[0]))
        Sy2 = np.dot(Sy.transpose(),Sy)
        temp_mean = np.sign(self.Wc[0])*wc_root * np.outer(e[:,0],e[:,0].transpose())
        self.Sy = np.linalg.cholesky(Sy2 + temp_mean)

    def predict(self,U,dt):

        '''
            UKF state prediction.

            @Input
            U : list
                3x1 list containing gyroscope data (rad/s) in the form [x, y, z]
            dt : float
                time interval between measurements (seconds)
        '''

        # Take the root of P
        P_root_G = self.sqr_gamma*self.S

        # generate sigma points
        self.Sig_f[:,0] = self.x
        for i in range(0,self.adj_n):
            p_pos = self.vec2q(P_root_G[0:3,i])
            p_neg = self.vec2q(-P_root_G[0:3,i])

            # Positive quaternion
            self.Sig_f[0:4,i+1] = Kin.quat_product(p_pos,self.x)
            # negative quaternion
            self.Sig_f[0:4,i+1+self.adj_n] = Kin.quat_product(p_neg,self.x)

            # Remainding states
            self.Sig_f[4:7,i+1] = self.x[4:7] + P_root_G[3:6,i]
            self.Sig_f[4:7,i+1+self.adj_n] = self.x[4:7] - P_root_G[3:6,i]

        # Pass each sigma point through the state transition model
        for i in range(0,self.num_sigmas):
            self.Sig_f[:,i] = self.f(self.Sig_f[:,i],dt,U)

        # update state and covariance using the unscented transform
        self.UT_predict()

    def update(self,z):

        '''
            UKF measurement update.

            @Input
            z : list
                6x1 list containing the normalized observation vector of gyroscope and accelerometer data. Has form
                [ax, ay, az, mx, my, mz]

        '''

        mag = z[3:6]

        # Pass sigma points through the measurement model
        for i in range(0,self.num_sigmas):
            self.Sig_h[:,i] = self.h(mag,self.Sig_f[:,i])

        # Pass measurement sigma points through the UT
        self.UT_update()

        # Compute the cross covariance matrix
        #Todo: State sigma points are not recomputed therefore store residual in matrix to save time
        xresid = self.X_residual(self.Sig_f[:,0],self.x)
        yresid = self.Sig_h[:,0] - self.y_mean
        self.Pxy = self.Wc[0]*np.outer(xresid,yresid.transpose())
        for k in range(1,self.num_sigmas):
            xresid = self.X_residual(self.Sig_f[:,k],self.x)
            yresid = self.Sig_h[:,k] - self.y_mean
            self.Pxy = self.Pxy + self.Wc[1]*np.outer(xresid,yresid.transpose())

        # Compute Kalman gain
        PxinvSyT = np.dot(self.Pxy,np.linalg.inv(self.Sy.transpose()))
        K = np.dot(PxinvSyT,np.linalg.inv(self.Sy))   # K = Pxy*(Pyy)^(-1)
        meas_err = z - self.y_mean
        correction = K.dot(meas_err)                  # K*(z - zest)
        # Turn the correction into a quaternion
        q_corrected = self.vec2q(correction[0:3])

        # Update state esitmation (use quaternion multiplication for quaternion states)
        self.x[0:4] = Kin.quat_product(q_corrected,self.x[0:4])
        self.x[4:7] = self.x[4:7] + correction[3:6]

        # Update state covariance (square root KF)
        U = np.dot(K,self.Sy)
        for k in range(0,self.adj_n):
            S2 = np.dot(self.S,self.S.transpose())
            U2 = np.outer(U[:,k],U[:,k].transpose())
            self.S = np.linalg.cholesky(S2 - U2)