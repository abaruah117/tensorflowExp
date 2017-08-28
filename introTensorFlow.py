import tensorflow as tf
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1,node2) #prints the nodes
sess = tf.Session()
print(sess.run([node1,node2])) #prints the evaluation of the nodes
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
a = tf.placeholder(tf.float32) 
b = tf.placeholder(tf.float32) #placeholders, accept external input
adder_node = a + b
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
add_and_triple = adder_node*3. #new function, adds (adder_node) then mult by 3
print(sess.run(add_and_triple, {a:3,b:4.5}))
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b #line = mx+b
init = tf.global_variables_initializer() #initialize Variables (all)
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))
#loss function
def loss(linear_model, y):
    return tf.reduce_sum(tf.square(linear_model-y))
y = tf.placeholder(tf.float32)
print(sess.run(loss(linear_model, [0,-1,-2,-3]),{x:[1,2,3,4]}))
#alt:
#y = tf.placeholder(tf.float32)
#squared_deltas = tf.square(linear_model - y)
#loss = tf.reduce_sum(squared_deltas)
#print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
fixW = tf.assign(W,[-1.])#new value
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb])
print(sess.run(loss(linear_model, [0,-1,-2,-3]),{x:[1,2,3,4]}))
