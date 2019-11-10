import serial
import time

arduinoData = serial.Serial('/dev/cu.usbmodem143101',9600)
signal = int(input("please type command: "))
print("signal = ", signal)
time.sleep(2)
if signal == 1:
	arduinoData.write(b'1')
else:
	arduinoData.write(b'0')
print("done")