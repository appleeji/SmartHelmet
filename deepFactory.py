# Lab 9 XOR
import tensorflow as tf
import numpy as np
import math
import re

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

#dataAccident
#dataNoAccident

infile = open("frontBack.txt","r")
s = infile.read()
numbers = re.split("['\n' ]",s)

infile2 = open("newAccident.txt","r")
s2 = infile2.read()
numbers2 = re.split("['\n' ]",s2)

leng = len(numbers)/6/(10/7)
leng2 = len(numbers2) /6/4/(10/7)
leng3 = len(numbers)/6/(10/3)
leng4 = len(numbers2) /6/4/(10/3)
#leng = 2
x_data = [[0 for col in range(6)] for row in range(leng+leng2)]
x_data2 = [[0 for col in range(6)] for row in range(leng3+leng4)]
size = 0
for i in range((leng)*6):
        x_data[size][i%6]=float(numbers[i])
        if i % 6 == 5 :
                size=size+1
newSize = size
for i in range(leng2*6):
        x_data[size][i%6]=float(numbers2[i])
        if i % 6 == 5 :
                size=size+1
size=0
for i in range((leng*6),len(numbers)-1):
        x_data2[size][i%6]=float(numbers[i])
        if i % 6 == 5 :
                size=size+1
for i in range((leng2*6),(leng2*6+leng4)):
        x_data2[size][i%6]=float(numbers2[i])
        if i % 6 == 5 :
                size=size+1
y_data = [[1 for col in range(1)] for row in range(leng)]
y_data2 = [[0 for col in range(1)] for row in range(leng2)]
y_data = y_data + y_data2
t_data = [[1 for col in range(1)] for row in range(leng3)]
t_data2 =  [[0 for col in range(1)] for row in range(leng4)]
t_data = t_data + t_data2
print(len(x_data),"  ",len(y_data))
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

#x_data2 = [[0.7,0.7,0.7,0.7,0.7,0.9],[1,1,1,1,1,1]]
#y_data2 = [[0],[0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)
x_data2 = np.array(x_data2, dtype=np.float32)
y_data2 = np.array(y_data2, dtype=np.float32)
t_data = np.array(t_data, dtype=np.float32)

x_data_min = x_data.min(axis=0)
x_data_max = x_data.max(axis=0)
x_data_leng = abs(x_data_max-x_data_min)
x_data = (x_data - x_data_min)/x_data_leng

x_data_min2 = x_data2.min(axis=0)
x_data_max2 = x_data2.max(axis=0)
x_data_leng2 = abs(x_data_max2-x_data_min2)
x_data2 = (x_data2 - x_data_min2)/x_data_leng2
print(x_data_min)
print(x_data_leng)

print (x_data)
print (y_data)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_normal([6, 8]), name='weight1')
b1 = tf.Variable(tf.random_normal([8]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([8, 6]), name='weight2')
b2 = tf.Variable(tf.random_normal([6]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)

W3 = tf.Variable(tf.random_normal([6, 4]), name='weight3')
b3 = tf.Variable(tf.random_normal([4]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2,W3)+b3)
'''
W4 = tf.Variable(tf.random_normal([2, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
'''
W4 = tf.Variable(tf.random_normal([4, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')

'''
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
'''
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)


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
    saver.save(sess, 'frontBack.ckpt')
   # saver.restore(sess, 'rtttttttt.ckpt')
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data2, Y: t_data})
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
