import serial
import time

arduinoData = serial.Serial('/dev/cu.usbmodem143101',9600)
# if signal == 1:
# 	arduinoData.write(b'1')
# else:
# 	arduinoData.write(b'0')
while True:
	signal = input("please type command: ")
	if signal.isdigit():
		print("signal = ", signal)
		arduinoData.write(signal.encode())
	else:
		break