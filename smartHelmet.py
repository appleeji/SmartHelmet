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
'''
while True:
	mqtts.loop_start()
	#print(ID)
	
	if ID !=-1:
		print("aaaaaaa")
		break
		#while exit

	#mqtts.loop_forever()
print("test12323333")
'''
#------------------------------- ultrasonic ---------------------------------------

trig_pin = 13
echo_pin = 19

gpio.setmode(gpio.BCM)
gpio.cleanup()

gpio.setup(trig_pin, gpio.OUT)
gpio.setup(echo_pin, gpio.IN)

def getDistance():
	gpio.output(trig_pin, False)
	time.sleep(0.1)
	gpio.output(trig_pin, True)
	time.sleep(0.00001)
	gpio.output(trig_pin, False)
	
	while gpio.input(echo_pin) ==0:
		pulse_start = time.time()

	while gpio.input(echo_pin) == 1:
		pulse_end = time.time()

	pulse_duration = pulse_end - pulse_start
	distance = pulse_duration * 17000
	distance = round(distance, 2)
	return distance

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
W1 = tf.Variable(tf.random_normal([6, 4]), name='weight1')
b1 = tf.Variable(tf.random_normal([4]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([4, 3]), name='weight1')
b2 = tf.Variable(tf.random_normal([3]), name='bias1')
layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)

W3 = tf.Variable(tf.random_normal([3, 2]), name='weight1')
b3 = tf.Variable(tf.random_normal([2]), name='bias1')
layer3 = tf.nn.relu(tf.matmul(layer2,W3)+b3)

W4 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b4 = tf.Variable(tf.random_normal([1]), name='bias2')
		
#W4 = tf.Variable(tf.random_normal([6, 1]), name='weight2')
#b4 = tf.Variable(tf.random_normal([1]), name='bias2')
			
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
				       tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))



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
T=0
mqtts.loop_start()

try:
	while True:
		#step1 sleep
		print "step1"
		#time.sleep(0.3)
#		distance = getDistance()
#		if distance > 15:
#			continue

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
		x_data2 = [[0.7,0.7,0.7,0.7,0.7,0.9]]
		y_data2 = [[0]]

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
		# Launch graph
		saver = tf.train.Saver()
		print "-------------"
		with tf.Session() as sess:
		    # Initialize TensorFlow variables
		    if T!=1:
		   	 sess.run(tf.global_variables_initializer())
		    print "222222"
                    if T!=1:			    
		    	saver.restore(sess, 'NoTest125.ckpt')
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
		  #  if(c[0][0]==0.0):
		#	print("test !!!")
		#    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
	#	tf.reset_default_graph()
		print "55555555"
		if c[0][0]!=0.0 :
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
			if (vector < 5000) and (abs(x_rotation) > 60) or (abs(y_rotation) > 60) : 
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
	gpio.cleanup()
	mqtts.loob_stop
	mqtts.unsubscribe("test/userID_1")
	mqtts.disconnect() 
