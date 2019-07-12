import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import os

import tfutil
import config

#----------------------------------------------------------------------------
# Parse individual image from a tfrecords file.

def _parse_tfrecord(record):
    features = tf.parse_single_example(record, features={
        'bbox': tf.FixedLenFeature([4], tf.int64),
        'label': tf.FixedLenFeature([6], tf.int64),
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)
        })
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape']), features['label'], features['bbox']

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord,                   # A tfrecord file.
        im_shape,                   # Images input shape [channel, height, width]
        repeat          = True,     # Repeat dataset indefinitely.
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 2):       # Number of concurrent threads.

        self.tfrecord           = tfrecord
        self.shape              = im_shape
        self.label_size         = 6
        self.dtype              = 'uint8'
        self.dynamic_range      = [0, 255]
        self._tf_minibatch_in   = None
        self._tf_iterator       = None
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1

        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])
            dset = tf.data.TFRecordDataset(self.tfrecord, compression_type='', buffer_size=buffer_mb<<20)
            dset = dset.map(_parse_tfrecord)
            bytes_per_item = np.prod(self.shape) * np.dtype(self.dtype).itemsize
            if shuffle_mb > 0:
                dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
            if repeat:
                dset = dset.repeat()
            if prefetch_mb > 0:
                dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
            dset = dset.batch(self._tf_minibatch_in)
            self._tf_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
            self._tf_next_element = self._tf_iterator.get_next()
            self._tf_init_ops = self._tf_iterator.make_initializer(dset)

    def configure(self, minibatch_size):
        assert minibatch_size >= 1
        # if self._cur_minibatch != minibatch_size:
        self._tf_init_ops.run({self._tf_minibatch_in: minibatch_size})
        self._cur_minibatch = minibatch_size

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images, labels
        return self._tf_next_element

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf()
        return tfutil.run(self._tf_minibatch_np)
    
    def get_length(self):
        self._tf_init_ops.run({self._tf_minibatch_in: 1})
        length = 0
        while True:
            try:
                tfutil.run(self._tf_next_element)
                length += 1
            except tf.errors.OutOfRangeError:
                return length

    def __len__(self):
        return self.get_length()

#----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.

def load_dataset(class_name='dataset.TFRecordDataset', verbose=False, **kwargs):
    adjusted_kwargs = dict(kwargs)
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = tfutil.import_obj(class_name)(**adjusted_kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset


if __name__=='__main__':
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)

    dset = load_dataset(tfrecord='../data/landmarks/land-tfrecords/land-tfrecords.tfrecords', im_shape=(224,224,3))
    dset.configure(32)
    labs_np, imgs_np  = tfutil.run(dset.get_minibatch_tf())

    plt.imshow(imgs_np[0].transpose(1,2,0))
    plt.plot(labs_np[0][::2],labs_np[0][1::2],'o')
    plt.show()