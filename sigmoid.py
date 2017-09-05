# Lab 9 XOR
import tensorflow as tf
import numpy as np
import math
import re

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

#dataAccident
#dataNoAccident

infile = open("dataNoAccident.txt","r")
s = infile.read()
numbers = re.split("['\n' ]",s)

infile2 = open("dataAccident.txt","r")
s2 = infile2.read()
numbers2 = re.split("['\n' ]",s2)

leng = len(numbers)/6
leng2 = len(numbers2) /6
#leng = 2
x_data = [[0 for col in range(6)] for row in range(leng+leng2)]
size = 0
for i in range((leng)*6):
#for i in range(2):
        
     	x_data[size][i%6]=float(numbers[i])
        if i % 6 == 5 :
                size=size+1
for i in range(leng2*6):
	x_data[size][i%6]=float(numbers2[i])
	if i % 6 == 5 :
		size=size+1
y_data = [[1 for col in range(1)] for row in range(leng)]
y_data2 = [[0 for col in range(1)] for row in range(leng2)]
y_data = y_data + y_data2
'''
x_data_mean = []
temp = 0
for i in range(9):
        for j in range(leng) :
                temp+=x_data[j][i]
        x_data_mean.append(temp/leng)
        temp=0

x_data_double = []
for i in range(9):
        for j in range(leng) :
                temp +=x_data[j][i]*x_data[j][i]
        x_data_double.append(temp/leng)
        temp=0
x_data_stddev= []
for i in range(9):
        x_data_stddev.append(math.sqrt(x_data_double[i] - x_data_mean[i]*x_data_mean[i]))
'''
'''
x_data[leng][0] = 50001
x_data[leng][1] = 50002
x_data[leng][2] = 50003
x_data[leng][3] = 50001
x_data[leng][4] = 80000
x_data[leng][5] = 40000

x_data[leng][6] = 1 
x_data[leng][7] = 1 
x_data[leng][8] = 1 

x_data[leng+1][0] = 1000
x_data[leng+1][1] = -2000
x_data[leng+1][2] = -1000
x_data[leng+1][3] = 2000
x_data[leng+1][4] = 20000
x_data[leng+1][5] = 1000

x_data[leng+2][0] = 1002
x_data[leng+2][1] = 20002
x_data[leng+2][2] = 1001
x_data[leng+2][3] = 2002
x_data[leng+2][4] = 2000
x_data[leng+2][5] = 1000

x_data[leng+3][0] = 10000
x_data[leng+3][1] = 2000
x_data[leng+3][2] = 1000
x_data[leng+3][3] = 1000
x_data[leng+3][4] = 2000
x_data[leng+3][5] = 2009

x_data[leng+4][0] = 1000
x_data[leng+4][1] = 1002
x_data[leng+4][2] = 2003
x_data[leng+4][3] = 1001
x_data[leng+4][4] = 20000
x_data[leng+4][5] = 1000

x_data[leng+5][0] = 1000
x_data[leng+5][1] = 2000
x_data[leng+5][2] = 1000
x_data[leng+5][3] = 3000
x_data[leng+5][4] = 2000
x_data[leng+5][5] = 10000

x_data[leng+6][0] = 1002
x_data[leng+6][1] = -20002
x_data[leng+6][2] = 1501
x_data[leng+6][3] = 1002
x_data[leng+6][4] = 2300
x_data[leng+6][5] = 1300

x_data[leng+7][0] = 1400
x_data[leng+7][1] = -25000
x_data[leng+7][2] = 1600
x_data[leng+7][3] = 1200
x_data[leng+7][4] = 1500
x_data[leng+7][5] = 1609

x_data[leng+8][0] = 15000
x_data[leng+8][1] = 14000
x_data[leng+8][2] = 1700
x_data[leng+8][3] = 1800
x_data[leng+8][4] = 1200
x_data[leng+8][5] = 1600

x_data[leng+9][0] = 1702
x_data[leng+9][1] = -1802
x_data[leng+9][2] = -17001
x_data[leng+9][3] = 20002
x_data[leng+9][4] = 25000
x_data[leng+9][5] = 1900

x_data[leng+10][0] = -2500
x_data[leng+10][1] = -2500
x_data[leng+10][2] = 25000
x_data[leng+10][3] = 2000
x_data[leng+10][4] = 3500
x_data[leng+10][5] = 1009

x_data[leng+11][0] = 2300
x_data[leng+11][1] = -23000
x_data[leng+11][2] = 2200
x_data[leng+11][3] = 2500
x_data[leng+11][4] = 1500
x_data[leng+11][5] = 1800

x_data[leng+12][0] = 1502
x_data[leng+12][1] = -15002
x_data[leng+12][2] = 1501
x_data[leng+12][3] = -2502
x_data[leng+12][4] = 3500
x_data[leng+12][5] = 1800

x_data[leng+13][0] = 16000
x_data[leng+13][1] = -16000
x_data[leng+13][2] = 16000
x_data[leng+13][3] = 16000
x_data[leng+13][4] = -26000
x_data[leng+13][5] = 18009

x_data[leng+14][0] = 17000
x_data[leng+14][1] = -18000
x_data[leng+14][2] = 17000
x_data[leng+14][3] = 18000
x_data[leng+14][4] = 25000
x_data[leng+14][5] = -14000

x_data[leng+15][0] = 18002
x_data[leng+15][1] = -19002
x_data[leng+15][2] = 16001
x_data[leng+15][3] = -15002
x_data[leng+15][4] = 23000
x_data[leng+15][5] = 14000

x_data[leng+16][0] = 16000
x_data[leng+16][1] = -16000
x_data[leng+16][2] = 16000
x_data[leng+16][3] = 16000
x_data[leng+16][4] = -37000
x_data[leng+16][5] = 19009


x_data[leng+17][0] = -840
x_data[leng+17][1] = 95
x_data[leng+17][2] = 125
x_data[leng+17][3] = 1504
x_data[leng+17][4] = 5248
x_data[leng+17][5] = 13900

x_data[leng+18][0] = -840
x_data[leng+18][1] = 95
x_data[leng+18][2] = 125
x_data[leng+18][3] = 1504
x_data[leng+18][4] = 5248
x_data[leng+18][5] = 13900

x_data[leng+19][0] = -840
x_data[leng+19][1] = 95
x_data[leng+19][2] = 125
x_data[leng+19][3] = 1504
x_data[leng+19][4] = 5248
x_data[leng+19][5] = 13900

x_data[leng+20][0] = -840
x_data[leng+20][1] = 95
x_data[leng+20][2] = 125
x_data[leng+20][3] = 1504
x_data[leng+20][4] = 5248
x_data[leng+20][5] = 13900

x_data[leng+21][0] = -840
x_data[leng+21][1] = 95
x_data[leng+21][2] = 125
x_data[leng+21][3] = 1504
x_data[leng+21][4] = 5248
x_data[leng+21][5] = 13900

x_data[leng+22][0] = -840
x_data[leng+22][1] = 95
x_data[leng+22][2] = 125
x_data[leng+22][3] = 1504
x_data[leng+22][4] = 5248
x_data[leng+22][5] = 13900
'''

'''
y_data[leng][0] = 0
y_data[leng+1][0] = 0
y_data[leng+2][0] = 0
y_data[leng+3][0] = 0
y_data[leng+4][0] = 0
y_data[leng+5][0] = 0
y_data[leng+6][0] = 0
y_data[leng+7][0] = 0
y_data[leng+8][0] = 0
y_data[leng+9][0] = 0
y_data[leng+10][0] = 0
y_data[leng+11][0] = 0
y_data[leng+12][0] = 0
y_data[leng+13][0] = 0
y_data[leng+14][0] = 0
y_data[leng+15][0] = 0
y_data[leng+16][0] = 0
'''
'''
y_data[leng+17][0] = 0
y_data[leng+18][0] = 0
y_data[leng+19][0] = 0
y_data[leng+20][0] = 0
y_data[leng+21][0] = 0
y_data[leng+22][0] = 0
'''

x_data2 = [[0.7,0.7,0.7,0.7,0.7,0.9],[1,1,1,1,1,1]]
y_data2 = [[0],[0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)
x_data2 = np.array(x_data2, dtype=np.float32)
y_data2 = np.array(y_data2, dtype=np.float32)

x_data_min = x_data.min(axis=0)
x_data_max = x_data.max(axis=0)
x_data_leng = abs(x_data_max-x_data_min)
x_data = (x_data - x_data_min)/x_data_leng
print(x_data_min)
print(x_data_leng)

print (x_data)
print (y_data)

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
'''
W4 = tf.Variable(tf.random_normal([6, 1]), name='weight2')
b4 = tf.Variable(tf.random_normal([1]), name='bias2')
'''
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
'''
hypothesis = tf.sigmoid(tf.matmul(X, W4) + b4)
'''
# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
saver = tf.train.Saver()
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
    saver.save(sess, 'test1.ckpt')
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print(c[0][0])
    if c[1][0]==0.0:
	print("test !!!")
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
'''
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1,W2]))
'''
'''
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
'''

'''
Hypothesis:  [[ 0.5]
 [ 0.5]
 [ 0.5]
 [ 0.5]]
Correct:  [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
Accuracy:  0.5
'''
