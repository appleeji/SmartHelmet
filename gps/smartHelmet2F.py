#deeplearning
import tensorflow as tf
import numpy as np
#gyro&accel
import RPi.GPIO as gpio
import smbus
import math

#mqtt
import paho.mqtt.client as mqtt
import random
import time

#gps
import serial
import pynmea2
#led
import time

from neopixel import *
#------------------------------------------------ led -----------------------------------------#\
# LED strip configuration:
LED_COUNT      = 10      # Number of LED pixels.
LED_PIN        = 18      # GPIO pin connected to the pixels (18 uses PWM!).
#LED_PIN        = 10      # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 5       # DMA channel to use for generating signal (try 5)
LED_BRIGHTNESS = 32     # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53
LED_STRIP      = ws.WS2811_STRIP_GRB   # Strip type and colour ordering\

# Define functions which animate LEDs in various ways.
def colorWipe(strip, color, wait_ms=50):
	"""Wipe color across display a pixel at a time."""
	for i in range(strip.numPixels()):
		strip.setPixelColor(i, color)
		strip.show()
		time.sleep(wait_ms/100000.0)
	

#------------------------------------------------ gps -----------------------------------------
def parseGPS(mystr):
	if mystr.find('GGA') > 0:
		msg = pynmea2.parse(mystr)
		return msg
	else :
		return None
serialPort = serial.Serial("/dev/ttyS0", 9600, timeout=0.5)

#------------------------------------------------ mqtt --------------------------------------
mqttc = mqtt.Client()
#mqttc.on_connect = on_connect
#mqttc.on_publish = on_publish

# YOU NEED TO CHANGE THE IP ADDRESS OR HOST NAME
mqttc.connect("13.124.1.204")
#mqttc.connect("localhost")



ID = -1
def on_connect2(self,client, userdata, rc):
	print ("connected with result code ", rc)
	mqtts.subscribe("user/woojin/id/userID_1/test")
	print("test")
def on_message2(client, userdata, msg):
        if msg.topic == "user/woojin/id/userID_1/test":
		global isCancel
		isCancel = 1
		print(msg.payload)
		global ID 
		ID = msg.payload
		print(ID+"test3333")
		#another code
		#def get_Distance(msg2):
		#	print(msg2)		
		#get_Distance('ttt')

isCancel = 0
mqtts = mqtt.Client()
mqtts.on_connect = on_connect2
mqtts.on_message = on_message2

mqtts.connect("13.124.1.204")

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
learning_rate = 0.1
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_normal([7, 4]), name='weight1')
b1 = tf.Variable(tf.random_normal([4]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([4, 2]), name='weight2')
b2 = tf.Variable(tf.random_normal([2]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
'''
W3 = tf.Variable(tf.random_normal([6, 4]), name='weight3')
b3 = tf.Variable(tf.random_normal([4]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2,W3)+b3)

W4 = tf.Variable(tf.random_normal([3, 2]), name='weight5')
b4 = tf.Variable(tf.random_normal([2]), name='bias5')
layer4 = tf.nn.relu(tf.matmul(layer3,W4)+b4)
'''		

W5 = tf.Variable(tf.random_normal([2, 1]), name='weight6')
b5 = tf.Variable(tf.random_normal([1]), name='bias6')

#W4 = tf.Variable(tf.random_normal([6, 1]), name='weight2')
#b4 = tf.Variable(tf.random_normal([1]), name='bias2')
			
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(layer2, W5) + b5)
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
				       tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))



count = 0
time_count = 0

gyro_xout_prev = 0.
gyro_yout_prev = 0.
gyro_zout_prev = 0.

accel_xout_prev = 0.
accel_yout_prev = 0.
accel_zout_prev = 0.

accel_xout_scaled_prev = 0. 
accel_yout_scaled_prev = 0.
accel_zout_scaled_prev = 0.
T=0
mqtts.loop_start()

try:
# Launch graph
	saver = tf.train.Saver()
	print "-------------"
	strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL, LED_STRIP)
	# Intialize the library (must be called once before other functions).
	strip.begin()
	print ('Color wipe animations.')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print "222222"
	    	saver.restore(sess, 'frontBack125.ckpt')
		ledCount = 0
		
		while True:
			#step1 sleep
			print "step1"
			time.sleep(0.01)
			if ledCount == 0:
				colorWipe(strip, Color(0, 0, 255)) 
				ledCount += 1
	
			#step2 get gyro&aceel sensor value 
			print "step2"
			gyro_xout = read_word_2c(0x43)
			gyro_yout = read_word_2c(0x45)
			gyro_zout = read_word_2c(0x47)
		
			print "gyro_xout: ", gyro_xout, " scaled: ", (gyro_xout / 131)
			print "gyro_yout: ", gyro_yout, " scaled: ", (gyro_yout / 131)
			print "gyro_zout: ", gyro_zout, " scaled: ", (gyro_zout / 131)
		
			accel_xout = read_word_2c(0x3b)
			accel_yout = read_word_2c(0x3d)
			accel_zout = read_word_2c(0x3f)
		
			print "accel_xout: ", accel_xout
			print "accel_yout: ", accel_yout
			print "accel_zout: ", accel_zout

			accel_xout_scaled = accel_xout / 16384.0
			accel_yout_scaled = accel_yout / 16384.0
			accel_zout_scaled = accel_zout / 16384.0

			time_count +=1
			if(time_count%30)==1:
				mqttc.loop_start()

				sendMsg = str(gyro_xout)+'*'+str(gyro_yout)+'*'+str(gyro_yout)
				print sendMsg
				mqttc.publish("gyroSensor1",sendMsg)	
				mystr = serialPort.readline()
				mqttc.publish("gyroSensor1",sendMsg)	

				msg = parseGPS(mystr)
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
						sendMsg = (msg.lat)+'*'+str(msg.lon)
						print sendMsg
						mqttc.publish("gpsLocation1",sendMsg)	
						mqttc.loop_stop()						
							
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
	
			#step3 check if value is validate
			x_data2 = [[0.7,0.7,0.7,0.7,0.7,0.9,0.5]]
			y_data2 = [[0]]
			if msg:
				if msg.lat and msg.lon :
					if(msg.lat==37.11211 and msg.lon == 126.111):
						x_data2[0][6] = 0.0
			x_data2[0][0] = (gyro_xout +32768) / 82770.
			x_data2[0][1] = (gyro_yout +32768) / 82770.
			x_data2[0][2] = (gyro_zout +32768) / 82771.
			x_data2[0][3] = (accel_xout +32768) / 82770.
			x_data2[0][4] = (accel_yout +32768) / 112768.
			x_data2[0][5] = (accel_zout +32768) / 72777.
	
			print("fdsafladsf")
			print(x_data2)
			x_data2 = np.array(x_data2, dtype=np.float32)
			y_data2 = np.array(y_data2, dtype=np.float32)
			
			print("qqqqqqqqqqqqqqqqq")
			print(x_data2)
			print "1111111"

			print "3333333"
			# Accuracy report
			
			print("eeeeeeeeeeeeeee")
			print(x_data2)
			h, c, a = sess.run([hypothesis, predicted, accuracy],
				       feed_dict={X: x_data2, Y: y_data2})
			print "44444444"
			print(c[0][0])
			print(h[0][0])
			global T
			T = 1
			now = time.localtime()
			print((now.tm_hour+1))
			if(now.tm_hour>=20):
                                		h[0][0]-=0.1
			print (h[0][0])
			print "55555555"
			if h[0][0]>=0.5 :
				continue

	
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
			#step4 wait unitle gap between two variables is small enough
				print "step44"
				time.sleep(0.1)
				
				colorWipe(strip, Color(255, 117, 0)) 
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
				
				now = time.localtime()
				nowDatetime = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
				mqttc.loop_start()
				sendMsg = str(vector/60)+'*'+str(x_rotation)+'*'+str(y_rotation)+'*'+str(nowDatetime)
		
				print sendMsg
				mqttc.publish("gyroSensor",sendMsg)	
				if (vector < 10000) and (abs(x_rotation) > 45) or (abs(y_rotation) > 45) : 
					waitCount += 1
					print "waitcountDown ", waitCount
				else :
					print "skipcountDown ", skipCount
					skipCount += 1
					waitCount = 0
				#count 30seconds		
				if skipCount > 300 :
					break	
				#count 10secons
				if waitCount > 100 :
					testa = 1	
					#step6 get gps value
					print "step6"
					colorWipe(strip, Color(255, 0, 0))
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
	colorWipe(strip, Color(0, 0, 0),50) 
	mqtts.loob_stop
	mqtts.unsubscribe("test/userID_1")
	mqtts.disconnect()
