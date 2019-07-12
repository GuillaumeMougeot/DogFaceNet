import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image
import skimage as sk
import pandas as pd

import tfutil
import dataset
import misc

#----------------------------------------------------------------------------

def error(msg):
    print('[{:8s}] '.format('Error') + msg)
    exit(1)

def warning(msg):
    print('[{:8s}] '.format('Warning') + msg)

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(
        self,                       
        tfrecord_dir,               # Directory where the images will be saved
        expected_images,            # Number of expected images
        tfr_prefix          = None, # Name of the saved file. (Default: Directory name)
        print_progress      = True, # Display the progress?
        progress_interval   = 10):  # When should it be displayed?

        self.tfrecord_dir       = tfrecord_dir
        if tfr_prefix == None:
            self.tfr_prefix     = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        else:
            self.tfr_prefix     = os.path.join(self.tfrecord_dir, tfr_prefix)
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.tfr_opt            = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        self.tfr_writer         = tf.python_io.TFRecordWriter(self.tfr_prefix + '.tfrecords', self.tfr_opt)
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert(os.path.isdir(self.tfrecord_dir))
        
    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        self.tfr_writer.close()
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(
                self,
                img,        # An image
                **kwargs    # Can contain the following key word: 'label'
                ):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)

        img = img.astype(np.float32)
        quant = np.rint(img).clip(0, 255).astype(np.uint8)
        if not 'label' in kwargs.keys() or not 'bbox' in kwargs.keys():
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))
                }))
        else:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=kwargs['label'])),
                'bbox'  : tf.train.Feature(int64_list=tf.train.Int64List(value=kwargs['bbox'])),
                'shape' : tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data'  : tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))
                }))
        self.tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1
            
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

#----------------------------------------------------------------------------

def create_from_images(
    tfrecord_dir,                       # The location of the tfrecord directory
    images_dir,                         # The location of the image directory
    labels_file,                        # The location of the csv file
    test_train_split    = 0.1,          # Proportion of images to keep for testing
    output_shape        = 224           # Output shape.
    ):
    assert output_shape > 0
    assert test_train_split >= 0 and test_train_split < 1, error('Test/train ratio has to be in [0,1)')

    output_shape = (output_shape, output_shape, 3)

    print('Loading Landmark images from "%s"' % images_dir)
    # Retrieve image names
    image_filenames = os.listdir(images_dir)
    assert len(image_filenames) != 0, error('No input images found.')
    
    df = pd.read_csv(labels_file)
    del df['Unnamed: 0']

    df_filenames = np.array(df['filename'])
    col = df.columns
    df_coord = np.array([[int(df[col[i]][j]) for i in range(1,len(col))] for j in range(len(df))])
    
    assert len(df_filenames)==len(df_coord), error('The CSV file is not properly formated.')

    i = 0
    while i < len(df_filenames):
        if not df_filenames[i] in image_filenames:
            warning(df_filenames[i] + ' is not in the specified folder.')
            df_filenames = np.delete(df_filenames, i)
            df_coord = np.delete(df_coord, i, axis=0)
        else:
            i += 1
    
    img = sk.io.imread(images_dir+'/'+df_filenames[0])
    channels = img.shape[2] if img.ndim == 3 else 1
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')
    
    nbof_test_fn = int(test_train_split*len(df_filenames))

    def preprocess_img_coord(filename, coord):
        img = sk.io.imread(filename)
        # Clip the image
        img_clipped, coord_clipped = misc.clipping_img_coord(img, coord)
        # Resize it
        img_resized, coord_resized = misc.resize_img_coord(img_clipped, coord_clipped, output_shape)
        assert img_resized.shape == output_shape
        img = img_resized.transpose(2, 0, 1)
        if (img < 2).all():
            img *= 255
        # Compute the bounding box
        bbox = misc.bbox_coord(img_resized,coord_resized)
        return img, coord_resized, bbox

    print('Exporting test images...')
    with TFRecordExporter(tfrecord_dir, nbof_test_fn, 'land-test') as tfr:
        for i in range(nbof_test_fn):
            img, coord, bbox = preprocess_img_coord(images_dir+'/'+df_filenames[i], df_coord[i])
            tfr.add_image(img, label=coord, bbox=bbox)

    print('Exporting train images...')
    with TFRecordExporter(tfrecord_dir, len(df_filenames) - nbof_test_fn, 'land-train') as tfr:
        for i in range(nbof_test_fn, len(df_filenames)):
            img, coord, bbox = preprocess_img_coord(images_dir+'/'+df_filenames[i], df_coord[i])
            tfr.add_image(img, label=coord, bbox=bbox)
    
    print('Done.')

#----------------------------------------------------------------------------

# def create_from_images(tfrecord_dir, image_dir, shuffle):
#     print('Loading images from "%s"' % image_dir)
#     image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
#     if len(image_filenames) == 0:
#         error('No input images found')
        
#     img = np.asarray(PIL.Image.open(image_filenames[0]))
#     resolution = img.shape[0]
#     channels = img.shape[2] if img.ndim == 3 else 1
#     if img.shape[1] != resolution:
#         error('Input images must have the same width and height')
#     if resolution != 2 ** int(np.floor(np.log2(resolution))):
#         error('Input image resolution must be a power-of-two')
#     if channels not in [1, 3]:
#         error('Input images must be stored as RGB or grayscale')
    
#     with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
#         order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
#         for idx in range(order.size):
#             img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
#             if channels == 1:
#                 img = img[np.newaxis, :, :] # HW => CHW
#             else:
#                 img = img.transpose(2, 0, 1) # HWC => CHW
#             tfr.add_image(img)

#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating and preparing Dog Face Detector datasets.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)
        
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'display',          'Display images in dataset.',
                                            'display datasets/mnist')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')
  
    p = add_command(    'extract',          'Extract images from dataset.',
                                            'extract datasets/mnist mnist-images')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')
    p.add_argument(     'output_dir',       help='Directory to extract the images into')

    p = add_command(    'compare',          'Compare two datasets.',
                                            'compare datasets/mydataset datasets/mnist')
    p.add_argument(     'tfrecord_dir_a',   help='Directory containing first dataset')
    p.add_argument(     'tfrecord_dir_b',   help='Directory containing second dataset')
    p.add_argument(     '--ignore_labels',  help='Ignore labels (default: 0)', type=int, default=0)

    p = add_command(    'create_from_images', 'Create dataset from landmark detector.',
                                            'create_landmarks datasets/landmarks ~/datasets/landmarks')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'images_dir',       help='Directory containing the images')
    p.add_argument(     'labels_file',      help='File containing the labels')
    p.add_argument(     '--output_shape',   help='Output_shape (default: 224)', type=int, default=224)

    # p = add_command(    'create_from_images', 'Create dataset from a directory full of images.',
    #                                         'create_from_images datasets/mydataset myimagedir')
    # p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    # p.add_argument(     'image_dir',        help='Directory containing the images')
    # p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
