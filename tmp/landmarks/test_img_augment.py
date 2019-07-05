import numpy as np 
import matplotlib.pyplot as plt
import skimage as sk
import tensorflow as tf 

img_np = sk.io.imread('coucou.jpg')
img = tf.constant(img_np, dtype=tf.float32)
img = tf.expand_dims(img, 0)
img = (img-127.5)/127.5

m = 1.0
a = -m
b = m
filt = 0.5*tf.random_uniform((3,3,3,1))
# filt = tf.ones((3,3,3,1))
# filt = tf.zeros((3,3,3,1))
# filt = (b-a)*tf.random_uniform((3,3,3,1))+a
# s = [[0,1,0],
#     [1,-4,1],
#     [0,1, 0]]

# filt = tf.constant(np.expand_dims([s,s,s], -1),dtype=tf.float32)/4
# filt = tf.constant(filt_np, dtype=tf.float32)

output = tf.nn.depthwise_conv2d(img,filt, strides=[1,1,1,1], padding='SAME')
output = output*127.5 + 127.5
output = tf.cast(output[0], tf.uint8)

with tf.Session() as sess:
    print(img_np[:2,:2,:])
    output_np = sess.run(output)
    print(output_np[:2,:2,:])
    sk.io.imsave('coucou_out.jpg',output_np)