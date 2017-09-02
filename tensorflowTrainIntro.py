"""Tensorflow training intro using tf.train API"""
import tensorflow as tf
sess = tf.Session()
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b
init = tf.global_variables_initializer()
sess.run(init)
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model -y)
loss = tf.reduce_sum(squared_deltas)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(1000):
    sess.run(train, {x: [1,2,3,4],y: [0,-1,-2,-3]})
print(sess.run([W,b]))
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: [1,2,3,4], y: [0,-1,-2,-3]})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
