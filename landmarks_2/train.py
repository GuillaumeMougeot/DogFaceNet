import os
import time
import numpy as np
import tensorflow as tf

import config
import tfutil
import dataset
import misc
import networks

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    return x

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        N_lrate_base            = 0.001,    # Learning rate for the network.
        tick_kimg_base          = 160,      # Default interval of progress snapshots.
        tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:20, 1024:10}): # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0

        # Minibatch size.
        self.minibatch = minibatch_base

        # Learning rate.
        self.N_lrate = N_lrate_base

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_landmark_detector(
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    total_kimg              = 1,            # Total length of the training, measured in thousands of real images.
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    snapshot_size           = 16,           # Size of the snapshot image
    snapshot_ticks          = 1000,          # Number of images before maintenance
    image_snapshot_ticks    = 1,            # How often to export image snapshots?
    network_snapshot_ticks  = 10,           # How often to export network snapshots?
    save_tf_graph           = True,         # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,        # Include weight histograms in the tfevents file?
    resume_run_id           = None,         # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,         # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0.0,          # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0):         # Assumed wallclock time at the beginning. Affects reporting.

    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(tfrecord=config.tfrecord_train, verbose=True, **config.dataset)
    testing_set = dataset.load_dataset(tfrecord=config.tfrecord_test, verbose=True, repeat=False, shuffle_mb=0, **config.dataset)
    # TODO: data augmentation
    # TODO: testing set

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None: # TODO: save methods
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            N = misc.load_pkl(network_pkl)
        else:
            print('Constructing the network...') # TODO: better network (like lod-wise network)
            N = tfutil.Network('N', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config.N)
    N.print_layers()

    print('Building TensorFlow graph...')
    # Training set up
    with tf.name_scope('Inputs'):
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        reals, labels   = training_set.get_minibatch_tf() # TODO: increase the size of the batch by several loss computation and mean
    N_opt = tfutil.Optimizer(name='TrainN', learning_rate=lrate_in, **config.N_opt)

    with tf.device('/gpu:0'):
        reals = process_reals(reals, training_set.dynamic_range, drange_net)
        labels = process_reals(labels, [0, training_set.shape[-2]], drange_net)
        with tf.name_scope('N_loss'): # TODO: loss inadapted
            N_loss = tfutil.call_func_by_name(N=N, reals=reals, labels=labels, **config.N_loss)
        
        N_opt.register_gradients(tf.reduce_mean(N_loss), N.trainables)
    N_train_op = N_opt.apply_updates()

    # Testing set up
    with tf.device('/gpu:0'), tf.name_scope('N_test_loss'):
        test_reals_tf, test_labels_tf   = testing_set.get_minibatch_tf()
        test_loss = tfutil.call_func_by_name(N=N, reals=test_reals_tf, labels=test_labels_tf, is_training=False, **config.N_loss)

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    summary_log = tf.summary.FileWriter(result_subdir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        N.setup_weight_histograms()

    test_reals, test_labels = testing_set.get_minibatch_np(snapshot_size)
    misc.save_img_coord(test_reals, test_labels, os.path.join(result_subdir, 'reals.png'), snapshot_size, adjust_range=False)

    test_reals = misc.adjust_dynamic_range(test_reals, training_set.dynamic_range, drange_net)
    test_coords = N.run(test_reals, minibatch_size=snapshot_size)
    misc.save_img_coord(test_reals, test_coords, os.path.join(result_subdir, 'fakes.png'), snapshot_size)

    print('Training...')
    tfutil.run(tf.global_variables_initializer())

    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time
    while cur_nimg < total_kimg * 1000:

        # Choose training parameters and configure training ops.
        sched = TrainingSchedule(cur_nimg, training_set, **config.sched)
        training_set.configure(sched.minibatch)

        # Run training ops.
        for _ in range(minibatch_repeats):
            tfutil.run([N_train_op], {lrate_in: sched.N_lrate, minibatch_in: sched.minibatch})
            cur_nimg += sched.minibatch

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000) or (cur_nimg % snapshot_ticks == 0)
        if done:

            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time

            testing_set.configure(sched.minibatch)
            _test_loss = 0
            for _ in range(0, testing_set.shape[0], sched.minibatch):
                _test_loss += tfutil.run(test_loss)
            _test_loss /= (testing_set.shape[0]/sched.minibatch)

            # Report progress. # TODO: improved report display
            print('tick %-5d kimg %-8.1f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f test_loss %.4f' % (
                tfutil.autosummary('Progress/tick', cur_tick),
                tfutil.autosummary('Progress/kimg', cur_nimg / 1000),
                tfutil.autosummary('Progress/minibatch', sched.minibatch),
                misc.format_time(tfutil.autosummary('Timing/total_sec', total_time)),
                tfutil.autosummary('Timing/sec_per_tick', tick_time),
                tfutil.autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                tfutil.autosummary('TrainN/test_loss', _test_loss)
                ))

            tfutil.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            tfutil.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            tfutil.save_summaries(summary_log, cur_nimg)

            if cur_tick % image_snapshot_ticks == 0 or done:
                test_coords = N.run(test_reals, minibatch_size=snapshot_size)
                misc.save_img_coord(test_reals, test_coords, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg // (10*snapshot_ticks))), snapshot_size)
            # if cur_tick % network_snapshot_ticks == 0 or done:
            #     misc.save_pkl(N, os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg // (10*snapshot_ticks))))

            # Record start time of the next tick.
            tick_start_time = time.time()

    # Write final results.
    # misc.save_pkl(N, os.path.join(result_subdir, 'network-final.pkl'))
    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    print('Running %s()...' % config.train['func'])
    tfutil.call_func_by_name(**config.train)
    print('Exiting...')

#----------------------------------------------------------------------------