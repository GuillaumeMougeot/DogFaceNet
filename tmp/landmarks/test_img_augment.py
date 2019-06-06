import numpy as np 
import matplotlib.pyplot as plt
import skimage as sk
import tensorflow as tf 

img_np = sk.io.imread('coucou.jpg')
img = tf.constant(img_np, dtype=tf.float32)
img = tf.expand_dims(img, 0)

a = 0
b = np.random.random()*0.5

filt = (b-a)*tf.random_uniform((3,3,3,1))+a
# filt = tf.constant(filt_np, dtype=tf.float32)

output = tf.nn.depthwise_conv2d(img,filt, strides=[1,1,1,1], padding='SAME')
output = tf.cast(output[0], tf.uint8)

with tf.Session() as sess:
    output_np = sess.run(output)
    print(output_np[:2])
    sk.io.imsave('coucou_out.jpg',output_np)