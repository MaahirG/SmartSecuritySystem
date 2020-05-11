import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)    # pin #'s are gpio spots

GPIO.setup(11, GPIO.OUT)
servo = GPIO.PWM(11,50)      # 50 is pulse rate (Hz)

servo.start(0) # pulse off

try:
    while True:
        servo.ChangeDutyCycle(7)  # turn towards 90 degree
        time.sleep(2) # sleep 1 second
        servo.ChangeDutyCycle(2)  # turn towards 0 degree
        time.sleep(2) # sleep 1 second
        servo.ChangeDutyCycle(12) # turn towards 180 degree
        time.sleep(2) # sleep 1 second 
except KeyboardInterrupt:
    servo.ChangeDutyCycle(2)
    time.sleep(1)
    servo.stop()
    GPIO.cleanup()

# while duty <= 12:
#     servo.ChangeDutyCycle(duty)
#     time.sleep(1)
#     duty = duty + 1

# time.sleep(2)

# servo.ChangeDutyCycle(7)
# time.sleep(2)

# servo.ChangeDutyCycle(2)
# time.sleep(0.5)
# servo.ChangeDutyCycle(0)

# servo.stop()
# GPIO.cleanup()
print("BYE")

