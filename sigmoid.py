# Lab 9 XOR
import tensorflow as tf
import numpy as np
import math
tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

infile = open("data.txt","r")
s = infile.read()
numbers = [eval(x) for x in s.split()]
leng = len(numbers)/9
#leng = 2
x_data = [[0 for col in range(6)] for row in range(leng+17)]
size = 0
for i in range(leng*9):
#for i in range(2):
        if i%9<6:
     	   x_data[size][i%6]=float(numbers[i])
        if i % 9 == 8 :
                size=size+1
y_data = [[1 for col in range(1)] for row in range(leng+17)]
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

x_data[leng][0] = 50001
x_data[leng][1] = 50002
x_data[leng][2] = 50003
x_data[leng][3] = 50001
x_data[leng][4] = 80000
x_data[leng][5] = 40000
'''
x_data[leng][6] = 1 
x_data[leng][7] = 1 
x_data[leng][8] = 1 
'''
x_data[leng+1][0] = 50000
x_data[leng+1][1] = 50000
x_data[leng+1][2] = 50000
x_data[leng+1][3] = 50000
x_data[leng+1][4] = 80000
x_data[leng+1][5] = 40000

x_data[leng+2][0] = 50002
x_data[leng+2][1] = 50002
x_data[leng+2][2] = 50001
x_data[leng+2][3] = 50002
x_data[leng+2][4] = 80000
x_data[leng+2][5] = 40000

x_data[leng+3][0] = 50000
x_data[leng+3][1] = 50000
x_data[leng+3][2] = 50000
x_data[leng+3][3] = 50000
x_data[leng+3][4] = 80000
x_data[leng+3][5] = 40009

x_data[leng+4][0] = 40000
x_data[leng+4][1] = 40002
x_data[leng+4][2] = 40003
x_data[leng+4][3] = 40001
x_data[leng+4][4] = 60000
x_data[leng+4][5] = 30000
'''
x_data[leng][6] = 1 
x_data[leng][7] = 1 
x_data[leng][8] = 1 
'''
x_data[leng+5][0] = 40000
x_data[leng+5][1] = 40000
x_data[leng+5][2] = 40000
x_data[leng+5][3] = 40000
x_data[leng+5][4] = 60000
x_data[leng+5][5] = 30000

x_data[leng+6][0] = 40002
x_data[leng+6][1] = 40002
x_data[leng+6][2] = 40001
x_data[leng+6][3] = 40002
x_data[leng+6][4] = 60000
x_data[leng+6][5] = 30000

x_data[leng+7][0] = 40000
x_data[leng+7][1] = 40000
x_data[leng+7][2] = 40000
x_data[leng+7][3] = 40000
x_data[leng+7][4] = 60000
x_data[leng+7][5] = 30009

x_data[leng+8][0] = 40000
x_data[leng+8][1] = 40000
x_data[leng+8][2] = 40000
x_data[leng+8][3] = 40000
x_data[leng+8][4] = 60000
x_data[leng+8][5] = 30000

x_data[leng+9][0] = 40002
x_data[leng+9][1] = 40002
x_data[leng+9][2] = 40001
x_data[leng+9][3] = 40002
x_data[leng+9][4] = 60000
x_data[leng+9][5] = 30000

x_data[leng+10][0] = 35000
x_data[leng+10][1] = 35000
x_data[leng+10][2] = 35000
x_data[leng+10][3] = 35000
x_data[leng+10][4] = 55000
x_data[leng+10][5] = 28009

x_data[leng+11][0] = 35000
x_data[leng+11][1] = 35000
x_data[leng+11][2] = 35000
x_data[leng+11][3] = 35000
x_data[leng+11][4] = 55000
x_data[leng+11][5] = 28000

x_data[leng+12][0] = 35002
x_data[leng+12][1] = 35002
x_data[leng+12][2] = 35001
x_data[leng+12][3] = 35002
x_data[leng+12][4] = 55000
x_data[leng+12][5] = 28000

x_data[leng+13][0] = 35000
x_data[leng+13][1] = 35000
x_data[leng+13][2] = 35000
x_data[leng+13][3] = 35000
x_data[leng+13][4] = 55000
x_data[leng+13][5] = 28009

x_data[leng+14][0] = 36000
x_data[leng+14][1] = 36000
x_data[leng+14][2] = 36000
x_data[leng+14][3] = 36000
x_data[leng+14][4] = 57000
x_data[leng+14][5] = 29000

x_data[leng+15][0] = 36002
x_data[leng+15][1] = 36002
x_data[leng+15][2] = 36001
x_data[leng+15][3] = 36002
x_data[leng+15][4] = 57000
x_data[leng+15][5] = 29000

x_data[leng+16][0] = 36000
x_data[leng+16][1] = 36000
x_data[leng+16][2] = 36000
x_data[leng+16][3] = 36000
x_data[leng+16][4] = 57000
x_data[leng+16][5] = 29009


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
'''
x_data_min = x_data.min(axis=0)
x_data_max = x_data.max(axis=0)
x_data_leng = abs(x_data_max-x_data_min)
x_data = (x_data - x_data_min)/x_data_leng
'''
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
