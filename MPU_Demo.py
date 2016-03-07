"""

MPU9150 demo program
-----------------------------
Testing if the MPU9150 is readable on the raspberry pi. Attempts to connect with the 
hardware. If successful, data from the MPU9150 will be output onto the terminal for
10 seconds. Otherwise the program exits.

Dependency on RTIMU:
https://github.com/richards-tech/RTIMULib

Contact:
	joseph.chrismer@gmail.com
Project blog:
	http://aerialarithmetic.blogspot.com/

"""

import RTIMU
import os.path
import time
import math

#Setup hardware
SETTINGS_FILE = "RTIMULib"

if not os.path.exists(SETTINGS_FILE + ".ini"):
  print("Settings file does not exist, will be created")

s = RTIMU.Settings(SETTINGS_FILE)
imu = RTIMU.RTIMU(s)

if (not imu.IMUInit()):
    print("IMU Init Failed")    
    print("exiting NOW")
    sys.exit(1)

imu.setSlerpPower(0.02)
imu.setGyroEnable(True)
imu.setAccelEnable(True)
imu.setCompassEnable(True)

poll_interval = imu.IMUGetPollInterval()

now = time.time()
start = time.time();

print("Starting...")
count = 0
while ((time.time() - start) < 10):
  if imu.IMURead():
            
    data = imu.getIMUData()    
    dt = time.time() - now
    
    accel = data['accel']
    gyro = data['gyro']
    compass = data['compass']
    
    IMU_list = accel+gyro+compass
    msg = ""
    #Convert readings to a string (used later for tcp)    
    for x in IMU_list:
        msg += '{:.4f}'.format(x)+','        
    msg+=str(dt)
    print(msg)

    now = time.time()
    count+=1   
    time.sleep(poll_interval*1.0/1000.0)
