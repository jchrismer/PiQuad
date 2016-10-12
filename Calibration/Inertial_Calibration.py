__author__ = 'joseph'
import statistics
import numpy as np

class AccelData(object):

    def __init__(self,Accel):
        #Static accelerometer data
        self.Accel = Accel

    def applyCalib(self,params,Accel):
        ax = params['ax']
        ay = params['ay']
        az = params['az']

        scaling_Matrix = np.diag([params['kx'], params['ky'],params['kz']])
        misalignment_Matrix = np.array([[1.0, -ax,ay],
                       [0, 1.0, -az],
                       [0,0,1.0]])

        adjustment_matrix = np.dot(misalignment_Matrix,scaling_Matrix)
        bx = params['bx']
        by = params['by']
        bz = params['bz']

        # apply theta to the accelerometer
        Accel[0,:] = Accel[0,:] - bx
        Accel[1,:] = Accel[1,:] - by
        Accel[2,:] = Accel[2,:] - bz

        Accel = np.dot(adjustment_matrix,Accel)

        return Accel    # probally not necessary as it may of been passed by reference

class GyroData(object):

    def __init__(self,Gyro,bx,by,bz):
        self.bx = bx
        self.by = by
        self.bz = bz
        self.Gyro = Gyro

    def applyCalib(self,params,Gyro):

        scaling_Matrix = np.diag([params['sx'], params['sy'],params['sz']])
        misalignment_Matrix = np.array([
                                       [1, params['gamma_yz'],params['gamma_zy']],
                                       [params['gamma_xz'], 1, params['gamma_zx']],
                                       [params['gamma_xy'],params['gamma_yx'],1]])

        adjustment_matrix = np.dot(misalignment_Matrix,scaling_Matrix)
        Gyro[0,:] = Gyro[0,:] - self.bx
        Gyro[1,:] = Gyro[1,:] - self.by
        Gyro[2,:] = Gyro[2,:] - self.bz

        Gyro = np.dot(adjustment_matrix,Gyro)
        return Gyro

class RollingStatistic(object):

    def __init__(self, window_size):
        self.N = window_size
        self.window = window_size * [0]
        self.average = 0
        self.variance = 0
        self.stddev = 0
        self.index = 0

    def update(self,new):
        # Preload
        if(self.index < self.N):
            self.window[self.index] = new
            self.index += 1

            # If Window preloaded - start rolling statistics
            if(self.index == self.N):
                self.average = statistics.mean(self.window)
                self.variance = statistics.variance(self.window)
            return

        # Push element into window list and remove the old element
        old = self.window[0]
        self.window.pop(0)
        self.window.append(new)

        oldavg = self.average
        newavg = oldavg + (new - old)/self.N
        self.average = newavg
        if(self.N > 1):
            self.variance += (new-old)*(new-newavg+old-oldavg)/(self.N-1)

    def getVar(self):
        if(self.index == 1):
            return 0
        elif(self.index < self.N):
            return statistics.variance(self.window[0:self.index]) # Make return 0?

        return self.variance

    def reset(self):
        self.index = 0

def static_invertal_detection(Data_in, Time, options,var_mult):
    total_samples = len(Time)
    Initial_Static = options[0]
    index = 0
    static_timer = 0
    static_window = options[1]

    running_var_x = RollingStatistic(25)
    running_var_y = RollingStatistic(25)
    running_var_z = RollingStatistic(25)

    # Find the total number of entries in the initial wait period
    while (static_timer <= Initial_Static):
        static_timer = static_timer + Time[index]
        index = index +1

    Static_var_X = statistics.variance(Data_in[0:index,0])
    Static_var_Y = statistics.variance(Data_in[0:index,1])
    Static_var_Z = statistics.variance(Data_in[0:index,2])

    Static_Th = Static_var_X**2 + Static_var_Y**2 + Static_var_Z**2     #Static threshold

    static_timer = 0
    current_interval_start = 1
    current_interval_end = current_interval_start + 1

    Valid_intervals_starts = []
    Valid_intervals_ends = []
    num_static = 0
    Max = -999999
    Min = 999999
    #loop through the dataset and map the static intervals
    for i in range(0,total_samples):
        # update time
        static_timer = static_timer + Time[i]

        running_var_x.update(Data_in[i,0])
        running_var_y.update(Data_in[i,1])
        running_var_z.update(Data_in[i,2])

        m = max([Data_in[i,0],Data_in[i,1],Data_in[i,2]])
        mn = min([Data_in[i,0],Data_in[i,1],Data_in[i,2]])
        # Store maximum for constructing the visualization of this later
        if(m > Max):
            Max = m
        if(mn < Min):
            Min = mn
        # Check current (rolling) variance
        current_norm = running_var_x.getVar()**2 + running_var_y.getVar()**2 + running_var_z.getVar()**2
        if(current_norm > Static_Th*var_mult):
            #check if the latest interval is valid length
            if(static_timer >= static_window):
                num_static += 1
                current_interval_end = i -1              # skip the point that caused it to go beyond threshold
                Valid_intervals_starts.append(current_interval_start)
                Valid_intervals_ends.append(current_interval_end)

            # Reset running variances
            running_var_x.reset()
            running_var_y.reset()
            running_var_z.reset()

            # Reset the current static interval starting and ending index
            current_interval_end = i
            current_interval_start = current_interval_end

            # Reset timer
            static_timer = 0

    # Main loop ended
    visualize = total_samples * [28000]
    for i in range(0,num_static):
        length = Valid_intervals_ends[i] - Valid_intervals_starts[i] + 1
        visualize[Valid_intervals_starts[i]:(Valid_intervals_ends[i]+1)] = [.6*Max]*length

    return Valid_intervals_starts, Valid_intervals_ends, visualize, index

def accel_resid(params, accel_staticx,accel_staticy,accel_staticz):

    scaling_Matrix = np.diag([params['kx'], params['ky'],params['kz']])
    misalignment_Matrix = np.array([[1, -params['ax'],params['ay']],
                   [0, 1, -params['az']],
                   [0,0,1]])

    adjustment_matrix = np.dot(misalignment_Matrix,scaling_Matrix)
    local_gravity = 9.81744
    bx = params['bx']
    by = params['by']
    bz = params['bz']

    # apply theta to the accelerometer
    accel_static = np.zeros((3,len(accel_staticx)))
    accel_static[0,:] = accel_staticx - bx
    accel_static[1,:] = accel_staticy - by
    accel_static[2,:] = accel_staticz - bz

    accel_static = np.dot(adjustment_matrix,accel_static)

    residual = len(accel_staticx)*[0.0]
    for i in range (0,len(accel_staticx)):
       residual[i] = (local_gravity**2 - (accel_static[0,i]**2 + accel_static[1,i]**2 + accel_static[2,i]**2))

    return residual

def gyro_resid(params,GyroData,AccelData,Time):
    index = 0
    interval_count = len(GyroData.Gyro)
    resid = interval_count*[0.0]
    for Gyro in GyroData.Gyro:
        # Apply calibration of the gyroscope
        dt = Time[index]
        G = np.array(Gyro)
        G_calib = GyroData.applyCalib(params,G.transpose())
        R = quaternion_RK4(G_calib,dt)

        # Extract gravity vector from accelerometer
        a = AccelData.Accel[:,index]
        Ua = AccelData.Accel[:,index+1]

        # Apply predicted rotation to accelerometer and compare to observed
        Ug = np.dot(R,a)
        diff = Ua - Ug

        # store the magnitude of the difference and update the static interval index
        resid[index] = diff[0]**2 + diff[1]**2 + diff[2]**2
        index += 1

    return resid


#TODO: Move to misc. kinematics
def quaternion_RK4(gyro,dt):

    num_samples = gyro.shape[1]

    q_k = np.array([1,0,0,0])

    # RK loop
    for i in range(0,(num_samples-1)):

        q1 = q_k
        S1 = gyro_cross4(gyro[:,i])
        k_1 = 1.0/2.0*np.dot(S1,q1)

        q2 = q_k + dt*1.0/2.0*k_1
        half_gyro_left = 1.0/2.0*(gyro[:,i] + gyro[:,i+1])
        S_half = gyro_cross4(half_gyro_left)
        k_2 = 1.0/2.0*np.dot(S_half,q2)

        q3 = q_k + dt*1.0/2.0*k_2
        k_3 = 1.0/2.0*np.dot(S_half,q3)

        q4 = q_k + dt*k_3
        S_2 = gyro_cross4(gyro[:,i+1])
        k_4 = 1.0/2.0*np.dot(S_2,q4)

        q_k = q_k + dt*(1.0/6.0*k_1 + 1.0/3.0*k_2 + 1.0/3.0*k_3 + 1.0/6.0*k_4)
        q_k = q_k*1.0/np.linalg.norm(q_k)

    # Convert quaternion to rotation matrix
    # TODO: MOVE TO MISC KIN
    r11 = q_k[0]**2 + q_k[1]**2 - q_k[2]**2 - q_k[3]**2
    r12 = 2.0*(q_k[1]*q_k[2] - q_k[0]*q_k[3])
    r13 = 2.0*(q_k[1]*q_k[3] + q_k[0]*q_k[2])
    r21 = 2.0*(q_k[1]*q_k[2] + q_k[0]*q_k[3])
    r22 = q_k[0]**2 - q_k[1]**2 + q_k[2]**2 - q_k[3]**2
    r23 = 2.0*(q_k[2]*q_k[3] - q_k[0]*q_k[1])
    r31 = 2.0*(q_k[1]*q_k[3] - q_k[0]*q_k[2])
    r32 = 2.0*(q_k[2]*q_k[3] + q_k[0]*q_k[1])
    r33 = q_k[0]**2 - q_k[1]**2 - q_k[2]**2 + q_k[3]**2

    # Note that R is actually the transpose of what it would normally be
    R = np.array([[r11, r21, r31],
                 [r12, r22, r32],
                 [r13, r23, r33]])

    return R

def gyro_cross4(gyro):
    gx = gyro[0]
    gy = gyro[1]
    gz = gyro[2]

    Sx = np.array([[0, -gx,  -gy, -gz],
                   [gx,  0,   gz, -gy],
                   [gy, -gz,   0,  gx],
                   [gz,  gy, -gx,   0]])
    return Sx