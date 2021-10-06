import os
import time
import atexit
from motor_class import Motor

motora = Motor(21,20,16,26)
minute = 60
motora.init()

def finish():
    motora.stop()
    return

atexit.register(finish)

motora.lock()

#print("revolution = " + str(Motor.REVOLUTION))
#print(str(motora.goto(200)))
#time.sleep(2)
#print(str(motora.goto(100)))
#time.sleep(2)
#print(str(motora.goto(900)))
#time.sleep(2)
#print(str(motora.goto(0)))
#time.sleep(2)

speed = Motor.NORMAL

while True:
    print("clockwise")
    t1 = time.time()
    motora.turn(.01*Motor.REVOLUTION, Motor.CLOCKWISE)
    t2 = time.time()
    print("time " + str(t2-t1))
    motora.lock()
    time.sleep(.1)
    
    
    print("Motor A Anti-Clockwise")
    t1 = time.time()
    motora.turn(.01*Motor.REVOLUTION, Motor.ANTICLOCKWISE)
    t2 = time.time()
    #print("time " + str(t2-t1))
    motora.lock()
    time.sleep(.1)
    
    
    if speed == Motor.NORMAL:
        speed = Motor.SLOW
        
    else:
        speed = Motor.NORMAL
        motora.setSpeed(speed)
