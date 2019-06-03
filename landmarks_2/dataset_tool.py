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

#----------------------------------------------------------------------------

def error(msg):
    print('[{:8s}] '.format('Error') + msg)
    exit(1)

def warning(msg):
    print('[{:8s}] '.format('Warning') + msg)

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        tfr_opt                 = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        self.tfr_writer         = tf.python_io.TFRecordWriter(self.tfr_prefix + '.tfrecords', tfr_opt)
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
        if not 'label' in kwargs.keys():
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))
                }))
        else:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=kwargs['label'])),
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))
                }))
        self.tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1
            
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

#----------------------------------------------------------------------------

def clipping(img, coord):
    # Clips an image to a square and pays attention to let the landmarks inside the picture
    h, w, _ = img.shape
    if h < w:
        bound_min = min(coord[::2])
        bound_max = max(coord[::2])
        if bound_max - bound_min > h:
            print("Shit happens sometimes... {:d} {:d} {:d}".format(bound_max, bound_min, h))
        clip = w - h
        d = bound_min
        D = w - bound_max
        left = int(d*clip/(d+D))
        right = bound_max + D - int(D*clip/(d+D))
        
        coord_add = np.copy(coord)
        coord_add[::2] -= left
        return img[:,left:right,:], np.array(coord_add)
    elif h > w:
        new_coord = []
        for i in range(3):
            new_coord += [coord[2*i+1]] + [coord[2*i]]
            
        img_T = np.transpose(img, axes=(1,0,2))
        img_clipped, coord_add = clipping(img_T, new_coord)
        
        coord_add_T = []
        for i in range(3):
            coord_add_T += [coord_add[2*i+1]] + [coord_add[2*i]]
        
        return np.transpose(img_clipped, axes=(1,0,2)), np.array(coord_add_T)
    else:
        return img, coord

def resize(img, coord, output_shape):
    # Resize an image and its landmarks
    img_resized = sk.transform.resize(img, output_shape)
    x_ratio = output_shape[0]/img.shape[0]
    y_ratio = output_shape[1]/img.shape[1]
    
    new_coord = np.zeros(6, dtype=np.int)
    for i in range(3):
        new_coord[2*i] = int(coord[2*i]*x_ratio)
        new_coord[2*i+1] = int(coord[2*i+1]*y_ratio)
    return img_resized, new_coord

def create_landmarks(
    tfrecord_dir,                   # The location of the tfrecord directory
    images_dir,                     # The location of the image directory
    labels_file,                    # The location of the csv file
    output_shape    = (224,224,3)   # Output shape.
    ):
    print('Loading Landmark images from "%s"' % images_dir)
    # Retrieve image names
    image_filenames = os.listdir(images_dir)
    if len(image_filenames) == 0:
        error('No input images found.')
    
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
    
    with TFRecordExporter(tfrecord_dir, len(df_filenames)) as tfr:
        for i in range(len(df_filenames)):
            img = sk.io.imread(images_dir+'/'+df_filenames[i])

            img_clipped, coord_clipped = clipping(img, df_coord[i])
            img_resized, coord_resized = resize(img_clipped, coord_clipped, output_shape)

            img = img_resized.transpose(2, 0, 1)
            if (img < 2).all():
                img *= 255

            tfr.add_image(img, label=coord_resized)

#----------------------------------------------------------------------------

def create_from_images(tfrecord_dir, image_dir, shuffle):
    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')
        
    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')
    
    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :] # HW => CHW
            else:
                img = img.transpose(2, 0, 1) # HWC => CHW
            tfr.add_image(img)

#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating, extracting, and visualizing Progressive GAN datasets.',
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

    p = add_command(    'create_landmarks', 'Create dataset from landmark detector.',
                                            'create_landmarks datasets/landmarks ~/datasets/landmarks')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'images_dir',       help='Directory containing the images')
    p.add_argument(     'labels_file',      help='File containing the labels')
    p.add_argument(     '--output_shape',   help='Output_shape (default: (224,224,3))', type=int, default=(224,224,3))

    p = add_command(    'create_from_images', 'Create dataset from a directory full of images.',
                                            'create_from_images datasets/mydataset myimagedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
