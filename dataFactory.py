#gyro&accel
import RPi.GPIO as gpio
import smbus
import math
import re

#mqtt
import paho.mqtt.client as mqtt
import random
import time

#gps
import serial
import pynmea2

#------------------------------------------------ gps -----------------------------------------
def parseGPS(mystr):
	if mystr.find('GGA') > 0:
		msg = pynmea2.parse(mystr)
		return msg
	else :
		return None
serialPort = serial.Serial("/dev/ttyS0", 9600, timeout=0.5)

#------------------------------------------------ mqtt --------------------------------------
'''
mqttc = mqtt.Client()
#mqttc.on_connect = on_connect
#mqttc.on_publish = on_publish

# YOU NEED TO CHANGE THE IP ADDRESS OR HOST NAME
mqttc.connect("192.168.0.48")
#mqttc.connect("localhost")

def on_connect2(client, userdata, rc):
	print "connected with result code ", rc
	client.subscribe("test/userID_1")

def on_message2(client, userdata, msg):
	if msg.topic == "test/userID_1":
		global isCancel
		isCancel = 1
		
isCancel = 0
mqtts = mqtt.Client()
mqtts.on_connect = on_connect2
mqtts.on_message = on_message2

mqttc.connect("192.168.0.48")
'''
#------------------------------- ultrasonic ---------------------------------------
#-------------------------------- gyro&accel ----------------------------------------
# Power management registers
power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c

def read_byte(adr):
	return bus.read_byte_data(address, adr)

def read_word(adr):
	high = bus.read_byte_data(address, adr)
	low = bus.read_byte_data(address, adr+1)
	val = (high << 8) + low
	return val

def read_word_2c(adr):
	val = read_word(adr)
	if (val >= 0x8000):
		return -((65535 - val) + 1)
	else:
		return val
def dist(a,b):
	return math.sqrt((a*a)+(b*b))

def get_y_rotation(x,y,z):
	radians = math.atan2(x, dist(y,z))
	return -math.degrees(radians)

def get_x_rotation(x,y,z):
	radians = math.atan2(y, dist(x,z))
	return math.degrees(radians)

bus = smbus.SMBus(1) # or bus = smbus.SMBus(1) for Revision 2 boards
address = 0x68       # This is the address value read via the i2cdetect command

# Now wake the 6050 up as it starts in sleep mode
bus.write_byte_data(address, power_mgmt_1, 0)

#---------------------------------- main -------------------------------------------
count = 0

gyro_xout_prev = 0.
gyro_yout_prev = 0.
gyro_zout_prev = 0.

accel_xout_prev = 0.
accel_yout_prev = 0.
accel_zout_prev = 0.

accel_xout_scaled_prev = 0. 
accel_yout_scaled_prev = 0.
accel_zout_scaled_prev = 0.


#mqtts.loop_start()

try:
	while True:
		outfile = open("rightNoAccident.txt","a+")
		
		#step1 sleep
		print "step1"
		time.sleep(0.05)

		#step2 get gyro&aceel sensor value 
		print "step2"
		gyro_xout = read_word_2c(0x43)
		gyro_yout = read_word_2c(0x45)
		gyro_zout = read_word_2c(0x47)
		outfile.write(str(gyro_xout)+" ")
		outfile.write(str(gyro_yout)+" ")
		outfile.write(str(gyro_zout)+" ")				
		print "gyro_xout: ", gyro_xout, " scaled: ", (gyro_xout / 131)
		print "gyro_yout: ", gyro_yout, " scaled: ", (gyro_yout / 131)
		print "gyro_zout: ", gyro_zout, " scaled: ", (gyro_zout / 131)
	
		accel_xout = read_word_2c(0x3b)
		accel_yout = read_word_2c(0x3d)
		accel_zout = read_word_2c(0x3f)
		outfile.write(str(accel_xout)+" ")
		outfile.write(str(accel_yout)+" ")
		outfile.write(str(accel_zout)+"\n")
		accel_xout_scaled = accel_xout / 16384.0
		accel_yout_scaled = accel_yout / 16384.0
		accel_zout_scaled = accel_zout / 16384.0
#		outfile.write(str(accel_xout_scaled)+" ")
#		outfile.write(str(accel_yout_scaled)+" ")
#		outfile.write(str(accel_zout_scaled)+" ")
		outfile.close()
		#skip first step	
		if count == 0:
			gyro_xout_prev = gyro_xout
			gyro_yout_prev = gyro_yout
			gyro_zout_prev = gyro_zout
	
			accel_xout_prev = accel_xout
			accel_yout_prev = accel_yout
			accel_zout_prev = accel_zout
	
			accel_xout_scaled_prev = accel_xout_scaled 
			accel_yout_scaled_prev = accel_yout_scaled 
			accel_zout_scaled_prev = accel_zout_scaled
			count += 1
			continue
		'''	
		#step3 check if value is validate
		print "step3"
		if math.sqrt(pow(accel_xout_prev - accel_xout,2)+pow(accel_yout_prev - accel_yout,2)+pow(accel_zout_prev - accel_zout,2)) < 5000 :
			continue
	
		#step4 wait unitle gap between two variables is small enough
		print "step4"
		waitCount = 0
		skipCount = 0
		gyro_xout_prev2 = 0.
		gyro_yout_prev2 = 0.
		gyro_zout_prev2 = 0.
	
		accel_xout_prev2 = 0.
		accel_yout_prev2 = 0.
		accel_zout_prev2 = 0.
	
		accel_xout_scaled_prev2 = 0. 
		accel_yout_scaled_prev2 = 0.
		accel_zout_scaled_prev2 = 0.
		waitCount = 0
		skipCount = 0
		while True :
			print "step44"
			outfile = open("data.txt","a+")
			time.sleep(0.05)
	
			gyro_xout2 = read_word_2c(0x43)
			gyro_yout2 = read_word_2c(0x45)
			gyro_zout2 = read_word_2c(0x47)
	
			accel_xout2 = read_word_2c(0x3b)
			accel_yout2 = read_word_2c(0x3d)
			accel_zout2 = read_word_2c(0x3f)
	
			accel_xout_scaled2 = accel_xout2 / 16384.0
			accel_yout_scaled2 = accel_yout2 / 16384.0
			accel_zout_scaled2 = accel_zout2 / 16384.0
	
			vector =  math.sqrt(pow(accel_xout_prev2 - accel_xout2,2)+pow(accel_yout_prev2 - accel_yout2,2)+pow(accel_zout_prev2 - accel_zout2,2))
			x_rotation = get_x_rotation(accel_xout_scaled2, accel_yout_scaled2, accel_zout_scaled2)
			y_rotation = get_y_rotation(accel_xout_scaled2, accel_yout_scaled2, accel_zout_scaled2)
	#		outfile.write(str(x_rotation)+" ")
        #               outfile.write(str(y_rotation)+" ")
        #               outfile.write(str(vector)+" ")				
                        outfile.close()

			now = time.localtime()
			nowDatetime = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
			mqttc.loop_start()
			sendMsg = str(vector/60)+'*'+str(x_rotation)+'*'+str(y_rotation)+'*'+str(nowDatetime)
	
			print sendMsg
			mqttc.publish("gyroSensor",sendMsg)	
	
			if (vector < 5000) and (abs(x_rotation) > 60) or (abs(y_rotation) > 60) :
				waitCount += 1
				print "waitcountDown ", waitCount
			else :
				print "skipcountDown ", skipCount
				skipCount += 1
				waitCount = 0
			#count 30seconds		
			if skipCount > 10000 :
				count = 0
				break	
			#count 10secons
			if waitCount > 10000 :
				testa = 1	
				#step6 get gps value
				print "step6"
				countGps = 0
				while True:
					time.sleep(0.1)
					mystr = serialPort.readline()
					msg = parseGPS(mystr)
					if countGps > 100 :
						break
					else :
						countGps += 1
					if msg :
						if msg.lat and msg.lon :
							#step7 publis gps value 
							print "step7"
	
							print msg.lon+'----------'+msg.lat
							b = float(msg.lat)/100
							c = float(msg.lon)/100
							msg.lat = "%f" % (b)
							msg.lon = "%f" % (c)
	
							positionLat = msg.lat.find('.')
							positionLon = msg.lon.find('.')
			
							latDegree = int(msg.lat[:positionLat])
							lonDegree = int(msg.lon[:positionLon])
							latMinute = float(msg.lat[positionLat:])/60*100
							lonMinute = float(msg.lon[positionLon:])/60*100
								
							latMerge = latDegree + latMinute
							msg.lat = "%f" % (latMerge)
							lonMerge = lonDegree + lonMinute
							msg.lon = "%f" % (lonMerge)
	
							#step8 publis gps value 
							print "step8"
							mqttc.loop_start()
							sendMsg = 'userID_1,'+msg.lon+','+msg.lat
							print sendMsg
							mqttc.publish("accident",sendMsg)	
							mqttc.loop_stop()				
							while isCancel == 0:
								print "waiting"
								time.sleep(1)
							count = 0
							isCancel = 0
							break
							
					else :
						continue
				break
				
			gyro_xout_prev2 = gyro_xout2
			gyro_yout_prev2 = gyro_yout2
			gyro_zout_prev2 = gyro_zout2
		
			accel_xout_prev2 = accel_xout2
			accel_yout_prev2 = accel_yout2
			accel_zout_prev2 = accel_zout2
	
			accel_xout_scaled_prev2 = accel_xout_prev2 / 16384.0
			accel_yout_scaled_prev2 = accel_yout_prev2 / 16384.0
			accel_zout_scaled_prev2 = accel_zout_prev2 / 16384.0
			'''	
	
		gyro_xout_prev = gyro_xout
		gyro_yout_prev = gyro_yout
		gyro_zout_prev = gyro_zout
		
		accel_xout_prev = accel_xout
		accel_yout_prev = accel_yout
		accel_zout_prev = accel_zout
	
		accel_xout_scaled_prev = accel_xout_prev / 16384.0
		accel_yout_scaled_prev = accel_yout_prev / 16384.0
		accel_zout_scaled_prev = accel_zout_prev / 16384.0
		
except KeyboardInterrupt as e:
	gpio.cleanup()
	mqtts.loob_stop
	mqtts.unsubscribe("test/userID_1")
	mqtts.disconnect()
#outfile.close()
