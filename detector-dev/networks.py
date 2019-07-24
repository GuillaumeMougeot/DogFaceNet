import numpy as np
import tensorflow as tf

# NOTE: Do not import any application-specific modules here!

#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: tf.cond(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False, padding='SAME'):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding=padding, data_format='NCHW')

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.0):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Fused upscale2d + conv2d.
# Faster and uses less memory than performing the operations separately.

def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Upscaling layer.

def upsample2d(x, add=2):
    with tf.variable_scope('Upsample2D'):
        s = x.get_shape().as_list()
        x = tf.transpose(x, (0,2,3,1))
        x = tf.image.resize_images(x, [s[2]+add, s[3]+add])
        x = tf.transpose(x, (0,3,1,2))
        return x

#----------------------------------------------------------------------------
# Box filter downscaling layer.

def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW') # NOTE: requires tf_config['graph_options.place_pruned_graph'] = True

#----------------------------------------------------------------------------
# Fused conv2d + downscale2d.
# Faster and uses less memory than performing the operations separately.

def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

#----------------------------------------------------------------------------
# Concatenation.

def concatenate(x, axis=1):
    with tf.variable_scope('Concat'):
        return tf.concat(x, axis=axis)

#----------------------------------------------------------------------------
# Minibatch standard deviation.

def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.

#----------------------------------------------------------------------------
# Global average pooling.

def global_avg_pool(x):
    with tf.variable_scope("GlobalAvgPool"):
        s = x.shape
        return tf.nn.avg_pool(x, ksize=[1,1,s[2],s[3]], strides=[1,1,1,1], padding='VALID', data_format='NCHW')

#----------------------------------------------------------------------------
# Dummy model for landmark detection.

def dummy(
    images_in,                          # Input: Images
    resolution      = 224,
    num_channels    = 3,
    label_size      = 6,
    dtype           = 'float32',
    use_wscale      = True,              # Enable equalized learning rate?
    **kwargs):

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    act = leaky_relu
    with tf.variable_scope("FirstLayer"):
        x = act(apply_bias(conv2d(images_in, fmaps=16, kernel=1, use_wscale=use_wscale)))

    fmaps = [16,32,64,128]

    for i in range(0,len(fmaps),1):
        # with tf.variable_scope("Conv{:d}".format(i)):
        #     x = act(apply_bias(conv2d(x, fmaps=fmaps[i], kernel=3, use_wscale=use_wscale)))
        with tf.variable_scope("Conv_down{:d}".format(i)):
            x = act(apply_bias(conv2d_downscale2d(x, fmaps=fmaps[i], kernel=3, use_wscale=use_wscale)))
            
    with tf.variable_scope("LastLayers"):
        x = global_avg_pool(x)
        with tf.variable_scope("Dense"):
            labels_out = dense(x, label_size, use_wscale=use_wscale)

    assert labels_out.dtype == tf.as_dtype(dtype)
    labels_out = tf.identity(labels_out, name='labels_out')
    return labels_out

#----------------------------------------------------------------------------
# Detector model.

def Detector(
    images_in,
    resolution      = 224,
    num_channels    = 3,
    dtype           = 'float32',
    use_wscale = True,
    **kwargs):
    
    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    act = leaky_relu
    with tf.variable_scope("FirstLayers"):
        with tf.variable_scope("Conv0"):
            x = act(apply_bias(conv2d_downscale2d(images_in, fmaps=16, kernel=3, use_wscale=use_wscale)))
        with tf.variable_scope("Conv1"):
            x = act(apply_bias(conv2d_downscale2d(x, fmaps=32, kernel=3, use_wscale=use_wscale)))
    """
    with tf.variable_scope("FeatureExtractor"):
        for i in range(7):
            with tf.variable_scope("Conv{:d}".format(i)):
                x = act(apply_bias(conv2d(x, fmaps=32, kernel=3, use_wscale=use_wscale, padding='VALID')))
            with tf.variable_scope("Conv0_{:d}".format(i)):
                if i > 0:
                    z = tf.identity(y) # Save the previous state
                y = apply_bias(conv2d(x, fmaps=1, kernel=1, use_wscale=use_wscale))
                y = tf.reshape(y, [-1, np.prod([d.value for d in y.shape[1:]])])
                if i > 0:
                    y = tf.concat([y, z], axis=-1) # Concatenate with the next one
        with tf.variable_scope("LastStage"):
            x = act(apply_bias(conv2d(global_avg_pool(x), fmaps=32, kernel=1, use_wscale=use_wscale, padding='VALID')))
            with tf.variable_scope("Conv0_Last"):
                z = tf.identity(y) # Save the previous state
                y = apply_bias(conv2d(x, fmaps=1, kernel=1, use_wscale=use_wscale))
                y = tf.reshape(y, [-1, np.prod([d.value for d in y.shape[1:]])])
                output = tf.concat([y, z], axis=-1) # Concatenate with the next one
    """
    pyramid = []
    with tf.variable_scope("FeatureExtractor"):
        for i in range(7):
            with tf.variable_scope("Conv{:d}".format(i)):
                x = act(apply_bias(conv2d(x, fmaps=64, kernel=3, use_wscale=use_wscale, padding='VALID')))
                pyramid = [x] + pyramid
        with tf.variable_scope("LastStage"):
            x = act(apply_bias(conv2d(global_avg_pool(x), fmaps=64, kernel=1, use_wscale=use_wscale, padding='VALID')))
            pyramid = [x] + pyramid

    # TODO: second try with a pyramid computation
    # with tf.variable_scope("Pyramid"):
    #     for i in range(len(pyramid)):
    #         with tf.variable_scope("Conv{:d}".format(i)):
    #             pyramid[i] = apply_bias(conv2d(pyramid[i], fmaps=1, kernel=1, use_wscale=use_wscale))
    #             pyramid[i] = tf.reshape(pyramid[i], [-1, np.prod([d.value for d in pyramid[i].shape[1:]])])

    # with tf.variable_scope("Pyramid"):  
    #     for i in range(len(pyramid)):
    #         with tf.variable_scope("Conv_before{:d}".format(i)):
    #             pyramid[i] = act(apply_bias(conv2d(pyramid[i], fmaps=64, kernel=3, use_wscale=use_wscale, padding='SAME')))
    #         if i == 0:
    #             up = upscale2d(pyramid[i])
    #         elif i < len(pyramid)-1:
    #             up = upsample2d(pyramid[i])
    #         if i < len(pyramid)-1:
    #             with tf.variable_scope("Conv_up{:d}".format(i)):
    #                 up = act(apply_bias(conv2d(up, fmaps=64, kernel=3, use_wscale=use_wscale, padding='SAME')))
    #             pyramid[i+1] = pyramid[i+1] + up
    #             with tf.variable_scope("Conv_add{:d}".format(i)):
    #                 pyramid[i+1] = act(apply_bias(conv2d(pyramid[i+1], fmaps=64, kernel=3, use_wscale=use_wscale, padding='SAME')))
    #         with tf.variable_scope("Conv{:d}".format(i)):
    #             pyramid[i] = apply_bias(conv2d(pyramid[i], fmaps=1, kernel=1, use_wscale=use_wscale))
    #             pyramid[i] = tf.reshape(pyramid[i], [-1, np.prod([d.value for d in pyramid[i].shape[1:]])])
    
    with tf.name_scope("Pyramid"):  
        for i in range(len(pyramid)):
            with tf.variable_scope("Conv_before{:d}".format(i)):
                pyramid[i] = act(apply_bias(conv2d(pyramid[i], fmaps=64, kernel=3, use_wscale=use_wscale, padding='SAME')))
            if i == 0:
                up = upscale2d(pyramid[i])
            elif i < len(pyramid)-1:
                up = upsample2d(pyramid[i])
            if i < len(pyramid)-1:
                with tf.variable_scope("Conv_up{:d}".format(i)):
                    up = act(apply_bias(conv2d(up, fmaps=64, kernel=3, use_wscale=use_wscale, padding='SAME')))
                pyramid[i+1] = pyramid[i+1] + up
                with tf.variable_scope("Conv_add{:d}".format(i)):
                    pyramid[i+1] = act(apply_bias(conv2d(pyramid[i+1], fmaps=64, kernel=3, use_wscale=use_wscale, padding='SAME')))
        # First output: dog, yes or no?
        with tf.name_scope("FirstOutput"):
            output1 = [tf.identity(pyramid[i]) for i in range(len(pyramid))]
            for i in range(len(output1)): 
                with tf.variable_scope("Conv{:d}".format(i)):
                    output1[i] = apply_bias(conv2d(output1[i], fmaps=1, kernel=1, use_wscale=use_wscale))
                    output1[i] = tf.reshape(output1[i], [-1, np.prod([d.value for d in output1[i].shape[1:]])])
        # Second output: refinement bounding box
        with tf.name_scope("SecondOutput"):
            output2 = [tf.identity(pyramid[i]) for i in range(len(pyramid))]
            for i in range(len(output2)):
                with tf.variable_scope("Conv{:d}_0".format(i)):
                    output2[i] = act(apply_bias(conv2d(output2[i], fmaps=64, kernel=3, use_wscale=use_wscale, padding='SAME')))
                with tf.variable_scope("Conv{:d}_1".format(i)):
                    output2[i] = apply_bias(conv2d(output2[i], fmaps=4, kernel=1, use_wscale=use_wscale))
                    output2[i] = tf.transpose(output2[i], (0,2,3,1)) # put the channel in the end
                    output2[i] = tf.reshape(output2[i], [-1, np.prod([d.value for d in output2[i].shape[1:3]]), 4]) #[?,HxW,4]
            
    # with tf.variable_scope("LastLayer"):
    #     output = tf.concat(pyramid, axis=-1, name='concat')
    with tf.variable_scope("LastLayer"):
        output1 = tf.concat(output1, axis=1, name='concat1')
        output2 = tf.concat(output2, axis=1, name='concat2')
    return output1, output2

#----------------------------------------------------------------------------
# Discriminator network used in the paper.

def D_paper(
    images_in,                          # Input: Images [minibatch, channel, height, width].
    num_channels        = 1,            # Number of input color channels. Overridden based on dataset.
    resolution          = 32,           # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    use_wscale          = True,         # Enable equalized learning rate?
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
    structure           = None,         # 'linear' = human-readable, 'recursive' = efficient, None = select automatically
    is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
    **kwargs):                          # Ignore unrecognized keyword args.
    
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if structure is None: structure = 'linear' if is_template_graph else 'recursive'
    act = leaky_relu

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

    # Building blocks.
    def fromrgb(x, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=use_wscale)))
    def block(x, res): # res = 2..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if res >= 3: # 8x8 and up
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                if fused_scale:
                    with tf.variable_scope('Conv1_down'):
                        x = act(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                else:
                    with tf.variable_scope('Conv1'):
                        x = act(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=use_wscale)))
                    x = downscale2d(x)
            else: # 4x4
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=use_wscale))
            return x
    
    # Linear structure: simple but inefficient.
    if structure == 'linear':
        img = images_in
        x = fromrgb(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromrgb(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = lerp_clip(x, y, lod_in - lod)
        combo_out = block(x, 2)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def grow(res, lod):
            x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
            x = block(x(), res); y = lambda: x
            if res > 2: y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
            return y()
        combo_out = grow(2, resolution_log2 - 2)

    assert combo_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(combo_out[:, :1], name='scores_out')
    labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
    return scores_out, labels_out
