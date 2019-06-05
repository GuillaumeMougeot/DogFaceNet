import tensorflow as tf 

x = tf.get_variable('x', initializer=tf.constant(2))
for i in range(2): 
    x = x * 2
z = x + 2
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(x))
    print(sess.run(z))