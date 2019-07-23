import os
import sys
import glob
import datetime
import pickle
import re
import numpy as np
from collections import OrderedDict 
import scipy.ndimage
import PIL.Image
import skimage as sk
import matplotlib.pyplot as plt

import config
import dataset
import legacy

#----------------------------------------------------------------------------
# Convenience wrappers for pickle that are able to load data produced by
# older versions of the code.

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return legacy.LegacyUnpickler(file, encoding='latin1').load()

def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


#----------------------------------------------------------------------------
# Image utils for landmark dataset definition and saving.

def clipping_img_coord(img, coord):
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
        img_clipped, coord_add = clipping_img_coord(img_T, new_coord)
        
        coord_add_T = []
        for i in range(3):
            coord_add_T += [coord_add[2*i+1]] + [coord_add[2*i]]
        
        return np.transpose(img_clipped, axes=(1,0,2)), np.array(coord_add_T)
    else:
        return img, coord

def resize_img_coord(img, coord, output_shape):
    # Resize an image and its landmarks
    img_resized = sk.transform.resize(img, output_shape)
    x_ratio = output_shape[0]/img.shape[0]
    y_ratio = output_shape[1]/img.shape[1]
    
    new_coord = np.zeros(len(coord), dtype=np.int)
    for i in range(len(coord)//2):
        new_coord[2*i] = int(coord[2*i]*x_ratio)
        new_coord[2*i+1] = int(coord[2*i+1]*y_ratio)
    return img_resized, new_coord

def bbox_coord(img, coord):
    # From the 6 landmarks return the coordinates of the bounding box
    # (x1, y1, x2, y2) of the face

    # A and b are the results of a single layer neural network trained
    # to take as input 6 coordinates and outputs 8 parameters that
    # represent:
    #  - 4 parameters for a rotation (quaternion)
    #  - 3 parameters for a translation
    #  - 1 parameter for a resizing
    A = np.array([[-2.7170989e-01,  1.5300398e-01, -1.3729802e-01, -2.4239977e-01, \
          5.6131786e-01,  1.3565560e-02, -5.4057974e-01, -3.0438498e-01],
        [-2.4932659e-01,  6.1709815e-01,  2.0384096e-01, -5.7076162e-01, \
          2.5820252e-01,  6.4068604e-01, -3.4571856e-01, -3.8039109e-01],
        [ 3.6703181e-01, -3.5540318e-01, -2.1373703e-01, -1.7523304e-01, \
          3.3640495e-01, -1.1292211e-01,  4.0444374e-01,  3.5088646e-01],
        [ 1.2811354e-01,  3.7042162e-01, -2.5829867e-01,  7.2075284e-01, \
          2.7979067e-02,  2.6547262e-01,  3.5050452e-02, -1.8263157e-01],
        [ 5.6548056e-04,  5.8828484e-02,  3.3033276e-01,  6.2417233e-01, \
          2.7791748e-02,  5.0881155e-02, -4.3204486e-02, -4.1529749e-02],
        [ 1.3910890e-01, -1.0610757e+00,  2.7555810e-02,  3.3871261e-03, \
         -3.1195429e-01,  4.6710368e-02,  6.2814605e-01,  5.2846295e-01]], dtype=np.float32)
    b = np.array([-0.01956358,  0.05305202,  0.01559347, -0.1411001 ,  0.03508957, \
         0.0346183 ,  0.        ,  0.02556852], dtype=np.float32)
    
    coord_norm = np.sqrt(np.sum(np.square(coord),axis=-1,keepdims=True))
    coord = coord/coord_norm
    pred = (np.dot(coord,A)+b) * coord_norm
    coord = coord*coord_norm
    
    def mult_quat_np(q1, q2):
        # Quaternion multiplication.
        q3 = np.copy(q1)
        q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        q3[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        q3[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        return q3

    def rotate_quat_np(quat, vect):
        # Rotate a vector with the rotation defined by a quaternion.
        # Transfrom vect into an quaternion 
        if len(vect)==3:
            vect = np.append([0],vect)
        # Normalize it
        norm_vect = np.linalg.norm(vect)
        if norm_vect:
            vect = vect/norm_vect
        # Normalize the quaternion
        norm_quat = np.linalg.norm(quat)
        if norm_quat:
            quat = quat/norm_quat
        # Computes the conjugate of quat
        quat_ = np.append(quat[0],-quat[1:])
        # The result is given by: quat * vect * quat_
        res = mult_quat_np(quat, mult_quat_np(vect,quat_)) * norm_vect
        #res = np.dot(mat_quat_np(quat), np.dot(mat_quat_np(vect),quat_)) * norm_vect
        return res[1:]

    def transform_np(x, model_output):
        q = model_output[:4]
        t = model_output[4:7]
        s = model_output[7]

        if np.linalg.norm(x)==0:
            out = t
        else:
            out = s * rotate_quat_np(q,x) + t

        return np.dot(out,np.array([[1,0],[0,1],[0,0]]))
    
    x = [0,0.5,0.5,0.8]
    y = transform_np(x, pred)
    
    def dist(v1,v2):
        # Distance between two vectors
        return np.sqrt((v1[0]-v2[0])**2+(v1[1]-v2[1])**2)
    
    potential_h = [
        dist(coord[[0,1]],coord[[2,3]]),
        dist([(coord[0]+coord[2])/2,(coord[1]+coord[3])/2],coord[[4,5]]),
        dist(y,coord[[4,5]])
    ]
    
    h = 2.1 * max(potential_h)
    x1 = int(max(y[0] - h/2,0))
    x2 = int(min(y[0] + h/2,img.shape[1]-1))
    y1 = int(max(y[1] - h/2,0))
    y2 = int(min(y[1] + h/2,img.shape[0]-1))
    
    return (x1,y1,x2,y2)

def save_img_coord(
    images,                             # List of images to save. Values are in [-1, 1]
    coords,                             # List of coordinates for landmarks. Values are in [-1, 1]
    filename,                           # Where to save the image
    num_saved_imgs  =16,                # Number of selected image among the list. Has to be a perfect square
    adjust_range    =True,
    output_shape    =(1080,1080,3)):    # Size of the output image
    
    # Save a list of images and its corresponding landmarks
    assert len(images)==len(coords)
    assert len(images) >= num_saved_imgs
    assert (np.sqrt(num_saved_imgs)-int(np.sqrt(num_saved_imgs)))==0.0, '[{:10s}] Number of saved images should be a perfect square.'.format('Error')
    
    # The output image will be 1080x1080
    w, h, _ = output_shape
    output = np.zeros(output_shape)
    sqrt_num = int(np.sqrt(num_saved_imgs))
    sub_output_size = output.shape[0] // sqrt_num

    if images.shape[1] == 3: images = images.transpose(0,2,3,1)
    if adjust_range:
        images = adjust_dynamic_range(images, [-1, 1], [0, 1])
        coords = adjust_dynamic_range(coords, [-1, 1], [0, images.shape[-2]])
    new_coords = np.copy(coords)

    for i in range(sqrt_num):
        for j in range(sqrt_num):
            idx = i*sqrt_num+j
            output[sub_output_size*i:sub_output_size*(i+1),sub_output_size*j:sub_output_size*(j+1),:], new_coords[idx] = resize_img_coord(images[idx], coords[idx], (sub_output_size,sub_output_size,3))
    
    # Figure set up: remove the axes
    fig = plt.figure()
    fig.set_size_inches(w/h, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Display the image
    ax.imshow(output)

    # Add the labels
    for i in range(sqrt_num):
        for j in range(sqrt_num):
            idx = i*sqrt_num+j
            ax.plot(new_coords[idx][::2] + j*sub_output_size, new_coords[idx][1::2] + i*sub_output_size, 'ko', markersize=0.2)
    
    # Save the figure
    plt.savefig(filename, dpi = h)
    plt.close()

# predefined_bboxes = []
# img_shape=64
# threshold=0.6

# for window_size in range(4,17,2):
#     ratio = img_shape/window_size
#     for i in range(window_size-2):
#         for j in range(window_size-2):
#             # Computes coordinates of the region of interest in the window
#             init_coord = np.array([i,j,i+3,j+3])
#             # Transforms it into img_shape coordinates
#             trans_coord = init_coord * ratio
#             predefined_bboxes += [trans_coord]
# predefined_bboxes = np.append([[0,0,img_shape,img_shape]],predefined_bboxes,axis=0)

def convert_pred_bboxes(pred,img_shape=64,threshold=config.threshold):
    # Converts network predictions to bboxes
    bboxes = []
    for window_size in range(4,17,2):
        ratio = img_shape/window_size
        for i in range(window_size-2):
            for j in range(window_size-2):
                # Computes coordinates of the region of interest in the window
                init_coord = np.array([i,j,i+3,j+3])
                # Transforms it into img_shape coordinates
                trans_coord = init_coord * ratio
                bboxes += [trans_coord]
    bboxes = np.append([[0,0,img_shape,img_shape]],bboxes,axis=0)
    return bboxes[np.argmax(pred)] # Return only the maximum
    
    # # Non maximum-suppression:
    # # bboxes_selected = bboxes[pred > 0.2]
    # # if len(bboxes_selected)==0:
    # #     return [0,0,0,0]
    # nb = 3 # Number of selected bounding boxes
    # idx_selected = np.argsort(pred)[-nb:]
    # bboxes_selected = bboxes[idx_selected] # Select the 3 most probable bboxes
    # # Centers of the selected bboxes
    # center_x = (bboxes_selected[:,0]+bboxes_selected[:,2])/2
    # center_y = (bboxes_selected[:,1]+bboxes_selected[:,3])/2
    # # Heigths and widths of the selected bboxes
    # widths = (bboxes_selected[:,2]-bboxes_selected[:,0])
    # heights  = (bboxes_selected[:,3]-bboxes_selected[:,1])
    # # Weighted average
    # if np.linalg.norm(pred[idx_selected])>0:
    #     weights = pred[idx_selected]/np.linalg.norm(pred[idx_selected])
    # else:
    #     weights = pred[idx_selected]
    # center_x = np.average(center_x, weights=weights)
    # center_y = np.average(center_y, weights=weights)
    # width  = np.average(widths, weights=weights)
    # height = np.average(heights, weights=weights)
    
    # x1 = center_x - width/2
    # y1 = center_y - height/2
    # x2 = center_x + width/2
    # y2 = center_y + height/2
    # return [x1, y1, x2, y2]


def save_img_bboxes(
    images,                             # List of images to save. Values are in [-1, 1]
    bboxes,                             # List of coordinates for bboxes. Values are in [-1, 1]
    filename,                           # Where to save the image
    num_saved_imgs  =16,                # Number of selected image among the list. Has to be a perfect square
    adjust_range    =True,
    output_shape    =(1080,1080,3)):    # Size of the output image
    
    # Save a list of images and its corresponding landmarks
    assert len(images)==len(bboxes)
    assert len(images) >= num_saved_imgs
    assert (np.sqrt(num_saved_imgs)-int(np.sqrt(num_saved_imgs)))==0.0, '[{:10s}] Number of saved images should be a perfect square.'.format('Error')
    
    # If it is the network output, converts it to bboxes
    if bboxes.shape[1]>4: # Then it is network output
        bboxes_transformed = np.empty((len(bboxes),4))
        for i in range(len(bboxes_transformed)):
            bboxes_transformed[i] = convert_pred_bboxes(bboxes[i])
        bboxes = bboxes_transformed

    # The output image will be 1080x1080
    w, h, _ = output_shape
    output = np.zeros(output_shape)
    sqrt_num = int(np.sqrt(num_saved_imgs))
    sub_output_size = output.shape[0] // sqrt_num

    if images.shape[1] == 3: images = images.transpose(0,2,3,1)
    if adjust_range:
        images = adjust_dynamic_range(images, [-1, 1], [0, 1])
    new_bboxes = np.copy(bboxes)

    for i in range(sqrt_num):
        for j in range(sqrt_num):
            idx = i*sqrt_num+j
            output[sub_output_size*i:sub_output_size*(i+1),sub_output_size*j:sub_output_size*(j+1),:], new_bboxes[idx] = resize_img_coord(images[idx], bboxes[idx], (sub_output_size,sub_output_size,3))
    
    # Figure set up: remove the axes
    fig = plt.figure()
    fig.set_size_inches(w/h, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Display the image
    ax.imshow(output)
    # Add the labels
    for i in range(sqrt_num):
        for j in range(sqrt_num):
            idx = i*sqrt_num+j
            x1,x2 = new_bboxes[idx][::2]
            xs = np.array([x1,x2,x2,x1,x1]) + j*sub_output_size
            y1,y2 = new_bboxes[idx][1::2]
            ys = np.array([y1,y1,y2,y2,y1]) + i*sub_output_size
            ax.plot(xs, ys, 'k', lw=0.2)
    
    # Save the figure
    plt.savefig(filename, dpi = h)
    plt.close()

def test_save_img():
    import pandas as pd
    PATH = '../data/landmarks/aligned/before_4/'
    img_filenames = os.listdir(PATH)
    df = pd.read_csv('../data/landmarks/aligned/labels/reformated_8698.csv')
    del df['Unnamed: 0']
    cols = df.columns

    nbof_img = 16
    images = []
    coords = []
    idx = 0
    while len(images) < nbof_img:
        if df.filename[idx] in img_filenames:
            coord = np.array([int(df[cols[i]][idx]) for i in range(1,len(cols))])
            img = sk.io.imread(PATH+df.filename[idx])
            img, coord = clipping_img_coord(img, coord)
            img, coord = resize_img_coord(img, coord, (224,224,3))

            img = adjust_dynamic_range(img, [0, 255], [-1, 1])
            coord = adjust_dynamic_range(coord, [0, 224], [-1, 1])

            images += [img]
            coords += [coord]
        idx += 1
    save_img_coord(np.array(images), np.array(coords), 'here.png', nbof_img)

def apply_box_delta(box, delta):
    #Applies the given deltas to the given boxes.
    box = box.astype(np.float32)
    # Convert to y, x, h, w
    height = box[2] - box[0]
    width = box[3] - box[1]
    center_y = box[0] + 0.5 * height
    center_x = box[1] + 0.5 * width
    # Apply deltas
    center_y += delta[0] * height
    center_x += delta[1] * width
    height *= np.exp(delta[2])
    width *= np.exp(delta[3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return [y1, x1, y2, x2]

def convert_pred_bboxes_ref(pred,ref,img_shape=64,threshold=config.threshold):
    # Converts network predictions to bboxes
    bboxes = []
    for window_size in range(4,17,2):
        ratio = img_shape/window_size
        for i in range(window_size-2):
            for j in range(window_size-2):
                # Computes coordinates of the region of interest in the window
                init_coord = np.array([i,j,i+3,j+3])
                # Transforms it into img_shape coordinates
                trans_coord = init_coord * ratio
                bboxes += [trans_coord]
    bboxes = np.append([[0,0,img_shape,img_shape]],bboxes,axis=0)
    pred = 1. / (1+np.exp(-pred))
    if max(pred) > 0.:
        return apply_box_delta(bboxes[np.argmax(pred)],ref[np.argmax(pred)]) # Return only the maximum
    else:
        return np.array([0,0,0,0])

def save_img_bboxes_ref(
    images,                             # List of images to save. Values are in [-1, 1]
    bboxes,                             # List of coordinates for bboxes. Values are in [-1, 1]
    refinement,
    filename,                           # Where to save the image
    num_saved_imgs  =16,                # Number of selected image among the list. Has to be a perfect square
    adjust_range    =True,
    output_shape    =(1080,1080,3)):    # Size of the output image
    # Save a list of images and its corresponding landmarks
    assert len(images)==len(bboxes)
    assert len(images) >= num_saved_imgs
    assert (np.sqrt(num_saved_imgs)-int(np.sqrt(num_saved_imgs)))==0.0, '[{:10s}] Number of saved images should be a perfect square.'.format('Error')
    
    # If it is the network output, converts it to bboxes
    if bboxes.shape[1]>4: # Then it is network output
        bboxes_transformed = np.empty((len(bboxes),4))
        for i in range(len(bboxes_transformed)):
            bboxes_transformed[i] = convert_pred_bboxes_ref(bboxes[i],refinement[i])
        bboxes = bboxes_transformed
        
    save_img_bboxes(images,bboxes_transformed,filename,num_saved_imgs=num_saved_imgs,adjust_range=adjust_range,output_shape=output_shape)
    

#----------------------------------------------------------------------------
# Image utils.

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def convert_to_pil_image(image, drange=[0,1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, format)

def save_image(image, filename, drange=[0,1], quality=95):
    img = convert_to_pil_image(image, drange)
    if '.jpg' in filename:
        img.save(filename,"JPEG", quality=quality, optimize=True)
    else:
        img.save(filename)

def save_image_grid(images, filename, drange=[0,1], grid_size=None):
    convert_to_pil_image(create_image_grid(images, grid_size), drange).save(filename)

#----------------------------------------------------------------------------
# Logging of stdout and stderr to a file.

class OutputLogger(object):
    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename, mode='wt'):
        assert self.file is None
        self.file = open(filename, mode)
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()

class TeeOutputStream(object):
    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush
 
    def write(self, data):
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()

output_logger = None

def init_output_logging():
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], autoflush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], autoflush=True)

def set_output_log_file(filename, mode='wt'):
    if output_logger is not None:
        output_logger.set_log_file(filename, mode)

#----------------------------------------------------------------------------
# Reporting results.

def create_result_subdir(result_dir, desc):

    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d-%s' % (run_id, desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print("Saving results to", result_subdir)
    set_output_log_file(os.path.join(result_subdir, 'log.txt'))

    # Export config.
    try:
        with open(os.path.join(result_subdir, 'config.txt'), 'wt') as fout:
            for k, v in sorted(config.__dict__.items()):
                if not k.startswith('_'):
                    fout.write("%s = %s\n" % (k, str(v)))
    except:
        pass

    return result_subdir

def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s // 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s // (60*60), (s // 60) % 60, s % 60)
    else:              return '%dd %02dh %02dm'    % (s // (24*60*60), (s // (60*60)) % 24, (s // 60) % 60)

#----------------------------------------------------------------------------
# Locating results.

def locate_result_subdir(run_id_or_result_subdir):
    if isinstance(run_id_or_result_subdir, str) and os.path.isdir(run_id_or_result_subdir):
        return run_id_or_result_subdir

    searchdirs = []
    searchdirs += ['']
    searchdirs += ['results']
    searchdirs += ['networks']

    for searchdir in searchdirs:
        dir = config.result_dir if searchdir == '' else os.path.join(config.result_dir, searchdir)
        dir = os.path.join(dir, str(run_id_or_result_subdir))
        if os.path.isdir(dir):
            return dir
        prefix = '%03d' % run_id_or_result_subdir if isinstance(run_id_or_result_subdir, int) else str(run_id_or_result_subdir)
        dirs = sorted(glob.glob(os.path.join(config.result_dir, searchdir, prefix + '-*')))
        dirs = [dir for dir in dirs if os.path.isdir(dir)]
        if len(dirs) == 1:
            return dirs[0]
    raise IOError('Cannot locate result subdir for run', run_id_or_result_subdir)

def list_network_pkls(run_id_or_result_subdir, include_final=True):
    result_subdir = locate_result_subdir(run_id_or_result_subdir)
    pkls = sorted(glob.glob(os.path.join(result_subdir, 'network-*.pkl')))
    if len(pkls) >= 1 and os.path.basename(pkls[0]) == 'network-final.pkl':
        if include_final:
            pkls.append(pkls[0])
        del pkls[0]
    return pkls

def locate_network_pkl(run_id_or_result_subdir_or_network_pkl, snapshot=None):
    if isinstance(run_id_or_result_subdir_or_network_pkl, str) and os.path.isfile(run_id_or_result_subdir_or_network_pkl):
        return run_id_or_result_subdir_or_network_pkl

    pkls = list_network_pkls(run_id_or_result_subdir_or_network_pkl)
    if len(pkls) >= 1 and snapshot is None:
        return pkls[-1]
    for pkl in pkls:
        try:
            name = os.path.splitext(os.path.basename(pkl))[0]
            number = int(name.split('-')[-1])
            if number == snapshot:
                return pkl
        except ValueError: pass
        except IndexError: pass
    raise IOError('Cannot locate network pkl for snapshot', snapshot)

def get_id_string_for_network_pkl(network_pkl):
    p = network_pkl.replace('.pkl', '').replace('\\', '/').split('/')
    return '-'.join(p[max(len(p) - 2, 0):])

#----------------------------------------------------------------------------
# Loading and using trained networks.

def load_network_pkl(run_id_or_result_subdir_or_network_pkl, snapshot=None):
    return load_pkl(locate_network_pkl(run_id_or_result_subdir_or_network_pkl, snapshot))

def random_latents(num_latents, G, random_state=None):
    if random_state is not None:
        return random_state.randn(num_latents, *G.input_shape[1:]).astype(np.float32)
    else:
        return np.random.randn(num_latents, *G.input_shape[1:]).astype(np.float32)

def load_dataset_for_previous_run(run_id, **kwargs): # => dataset_obj, mirror_augment
    result_subdir = locate_result_subdir(run_id)

    # Parse config.txt.
    parsed_cfg = dict()
    with open(os.path.join(result_subdir, 'config.txt'), 'rt') as f:
        for line in f:
            if line.startswith('dataset =') or line.startswith('train ='):
                exec(line, parsed_cfg, parsed_cfg)
    dataset_cfg = parsed_cfg.get('dataset', dict())
    train_cfg = parsed_cfg.get('train', dict())
    mirror_augment = train_cfg.get('mirror_augment', False)

    # Handle legacy options.
    if 'h5_path' in dataset_cfg:
        dataset_cfg['tfrecord_dir'] = dataset_cfg.pop('h5_path').replace('.h5', '')
    if 'mirror_augment' in dataset_cfg:
        mirror_augment = dataset_cfg.pop('mirror_augment')
    if 'max_labels' in dataset_cfg:
        v = dataset_cfg.pop('max_labels')
        if v is None: v = 0
        if v == 'all': v = 'full'
        dataset_cfg['max_label_size'] = v
    if 'max_images' in dataset_cfg:
        dataset_cfg.pop('max_images')

    # Handle legacy dataset names.
    v = dataset_cfg['tfrecord_dir']
    v = v.replace('-32x32', '').replace('-32', '')
    v = v.replace('-128x128', '').replace('-128', '')
    v = v.replace('-256x256', '').replace('-256', '')
    v = v.replace('-1024x1024', '').replace('-1024', '')
    v = v.replace('celeba-hq', 'celebahq')
    v = v.replace('cifar-10', 'cifar10')
    v = v.replace('cifar-100', 'cifar100')
    v = v.replace('mnist-rgb', 'mnistrgb')
    v = re.sub('lsun-100k-([^-]*)', 'lsun-\\1-100k', v)
    v = re.sub('lsun-full-([^-]*)', 'lsun-\\1-full', v)
    dataset_cfg['tfrecord_dir'] = v

    # Load dataset.
    dataset_cfg.update(kwargs)
    dataset_obj = dataset.load_dataset(data_dir=config.data_dir, **dataset_cfg)
    return dataset_obj, mirror_augment

def apply_mirror_augment(minibatch):
    mask = np.random.rand(minibatch.shape[0]) < 0.5
    minibatch = np.array(minibatch)
    minibatch[mask] = minibatch[mask, :, :, ::-1]
    return minibatch

#----------------------------------------------------------------------------
# Text labels.

_text_label_cache = OrderedDict()

def draw_text_label(img, text, x, y, alignx=0.5, aligny=0.5, color=255, opacity=1.0, glow_opacity=1.0, **kwargs):
    color = np.array(color).flatten().astype(np.float32)
    assert img.ndim == 3 and img.shape[2] == color.size or color.size == 1
    alpha, glow = setup_text_label(text, **kwargs)
    xx, yy = int(np.rint(x - alpha.shape[1] * alignx)), int(np.rint(y - alpha.shape[0] * aligny))
    xb, yb = max(-xx, 0), max(-yy, 0)
    xe, ye = min(alpha.shape[1], img.shape[1] - xx), min(alpha.shape[0], img.shape[0] - yy)
    img = np.array(img)
    slice = img[yy+yb : yy+ye, xx+xb : xx+xe, :]
    slice[:] = slice * (1.0 - (1.0 - (1.0 - alpha[yb:ye, xb:xe]) * (1.0 - glow[yb:ye, xb:xe] * glow_opacity)) * opacity)[:, :, np.newaxis]
    slice[:] = slice + alpha[yb:ye, xb:xe, np.newaxis] * (color * opacity)[np.newaxis, np.newaxis, :]
    return img

def setup_text_label(text, font='Calibri', fontsize=32, padding=6, glow_size=2.0, glow_coef=3.0, glow_exp=2.0, cache_size=100): # => (alpha, glow)
    # Lookup from cache.
    key = (text, font, fontsize, padding, glow_size, glow_coef, glow_exp)
    if key in _text_label_cache:
        value = _text_label_cache[key]
        del _text_label_cache[key] # LRU policy
        _text_label_cache[key] = value
        return value

    # Limit cache size.
    while len(_text_label_cache) >= cache_size:
        _text_label_cache.popitem(last=False)

    # Render text.
    import moviepy.editor # pip install moviepy
    alpha = moviepy.editor.TextClip(text, font=font, fontsize=fontsize).mask.make_frame(0)
    alpha = np.pad(alpha, padding, mode='constant', constant_values=0.0)
    glow = scipy.ndimage.gaussian_filter(alpha, glow_size)
    glow = 1.0 - np.maximum(1.0 - glow * glow_coef, 0.0) ** glow_exp

    # Add to cache.
    value = (alpha, glow)
    _text_label_cache[key] = value
    return value

#----------------------------------------------------------------------------
# Test

if __name__=='__main__':
    test_save_img()
