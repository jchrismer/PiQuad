__author__ = 'joseph'
'''
Contains various utility functions for use in modeling rotating bodies
'''
import math
import numpy as np

# Convert unit quaternion to Euler angles
def quaterion2Euler(q):

    #Compute estimated rotation matrix elements
    R11 = 2.*q[0]**2-1 + 2.*q[1]**2
    R21 = 2.*(q[1]*q[2] - q[0]*q[3])
    R31 = 2.*(q[1]*q[3] + q[0]*q[2])
    R32 = 2.*(q[2]*q[3] - q[0]*q[1])
    R33 = 2.*q[0]**2 - 1 + 2*q[3]**2

    phi = math.atan2(R32, R33 )*180/math.pi                         # Roll
    theta = -math.atan(R31 / math.sqrt(1-R31**2) )*180/math.pi      # Pitch
    psi = math.atan2(R21, R11 )*180/math.pi                         # Yaw

    return [phi,theta,psi]

#Define body frame later
def quaternion_rotate(q,a):
    '''
    Rotates a by the given quaternion q
    :param q: (unit) quaternion rotating a
    :param a: vector to be rotated by q
    :return: the rotated vector
    '''

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    x_rot = a[0]*(q0**2 + q1**2 - q2**2 - q3**2) + 2*a[1]*(q1*q2 - q0*q3) + 2*a[2]*(q1*q3 + q0*q2)
    y_rot = 2*a[0]*(q1*q2 + q0*q3) + a[1]*(q0**2 - q1**2 + q2**2 - q3**2) + 2*a[2]*(q2*q3 - q0*q1)
    z_rot = 2*a[0]*(q1*q3-q0*q2) + 2*a[1]*(q2*q3 + q0*q1) + a[2]*(q0**2 - q1**2 - q2**2 + q3**2)

    return np.array([x_rot,y_rot,z_rot])

def quat_product(a,b):
    '''
    Preforms quaternion multiplication (cross product) on the given inputs
    :param a: left operand
    :param b: right operand
    :return ab: quaternion product of a and b
    '''

    ab = [0,0,0,0]
    ab[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    ab[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    ab[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    ab[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]

    return ab


def init_attitude(a,m):
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
            l = quaternion_rotate([qacc[0],qacc[1],qacc[2],qacc[3]],m)
            lx = l[0]
            ly = l[1]
            Rho = math.sqrt(lx**2 + ly**2)
            Rho_sqrt = math.sqrt(Rho)

            if (lx>=0):
               qmag = [math.sqrt(Rho + lx*Rho_sqrt)/math.sqrt(2.0*Rho),0.0,0.0,ly/(math.sqrt(2.0)*math.sqrt(Rho + lx*Rho_sqrt))]
            #lx < 0
            else:
               qmag = [ly/(math.sqrt(2)*math.sqrt(Rho - lx*Rho_sqrt)),0.0,0.0,math.sqrt(Rho - lx*Rho_sqrt)/math.sqrt(2.0*Rho)]
            Q0 = quat_product(qacc,qmag)
            Q0 = np.multiply(Q0,1.0/np.linalg.norm(Q0))
            return Q0

# Gradient descent approach to convert accelerometer and magnetometer measurement to quaternion
def GD_meas2q(a,m,q,mu,iteration = 10):
    q1=q[0]
    q2=q[1]
    q3=q[2]
    q4=q[3]

    i = 1
    while (i <= iteration):

        # Rotate measured magnetic sensor into Earth frame
        q_coniug=[q[0], -q[1],-q[2], -q[3]]
        hTemp = quat_product(q,[0,m[0],m[1],m[2]])
        h = quat_product(hTemp,q_coniug)

        # Reconstruct local magnetic field using vert. & horz. component to use as setpoint
        b = np.array([math.sqrt(h[1]**2+h[2]**2), 0, h[3]])
        b = b * 1.0/np.linalg.norm(b)

        # Gravity and magnetometer cost function vector
        Fgb = np.array([(2.0*(q2*q4-q1*q3) - a[0]),
                       (2.0*(q1*q2+q3*q4) - a[1]),
                       (2.0*(0.5-q2**2-q3**2) - a[2]),
                       (2.0*b[0]*(0.5-q3**2-q4**2) + 2*b[2]*(q2*q4-q1*q3) - m[0]),
                       (2.0*b[0]*(q2*q3-q1*q4) + 2.0*b[2]*(q1*q2+q3*q4) - m[1]),
                       (2.0*b[0]*(q1*q3+q2*q4) + 2.0*b[2]*(0.5-q2**2-q3**2.0) - m[2])],
                       np.float)

        # Cost function Jacobian w/res to quaternions
        Jgb = np.array([[-2.0*q3,      2.0*q4,             -2.0*q1, 2.0*q2],
                        [2.0*q2,       2.0*q1,              2.0*q4, 2.0*q3],
                        [0.0,         -4.0*q2,             -4.0*q3, 0.0],
                        [-2.0*b[2]*q3          , 2.0*b[2]*q4          ,-4.0*b[0]*q3-2.0*b[2]*q1, -4.0*b[0]*q4+2.0*b[2]*q2],
                        [-2.0*b[0]*q4+2*b[2]*q2, 2.0*b[0]*q3+2*b[2]*q1, 2.0*b[0]*q2+2.0*b[2]*q4, -2.0*b[0]*q1+2.0*b[2]*q3],
                        [2.0*b[0]*q3           , 2.0*b[0]*q4-4*b[2]*q2, 2.0*b[0]*q1-4.0*b[2]*q3, 2.0*b[0]*q2]],
                       np.float)

        # Preform GD
        Df = np.dot(Jgb.transpose(),Fgb)
        q_result=q-mu*Df*1.0/np.linalg.norm(Df)
        q_result=q_result*1.0/np.linalg.norm(q_result);

        q1=q_result[0]
        q2=q_result[1]
        q3=q_result[2]
        q4=q_result[3]
        q=[q1, q2, q3, q4]

        i+= 1

    return q