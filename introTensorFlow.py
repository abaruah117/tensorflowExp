import tensorflow as tf
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)
print(node1,node2) #prints the nodes
sess = tf.Session()
print(sess.run([node1,node2])) #prints the evaluation of the nodes
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
node3: Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
a = tf.placeholder(tf.float32) 
b = tf.placeholder(tf.float32) #placeholders, accept external input
adder_node = a + b
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
add_and_triple = adder_node*3.
print(sess.run(add_and_triple, {a:3,b:4.5}))
