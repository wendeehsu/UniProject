import serial
import time

prev = 0
arduinoData = serial.Serial('/dev/cu.usbmodem141201',9600)
movement = [0,180,0]
for i in movement:
	time.sleep(1)
	print("signal = ", i, " ", 3*i/float(180))
	arduinoData.write(str(i).encode())
	time.sleep(float(abs(i-prev))/60)
	prev = i