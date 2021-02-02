"""
DogFaceNet
The main DogFaceNet implementation
This file contains:
 - Data loading
 - Model definition
 - Model training

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import skimage as sk
from skimage import io
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from online_training import *
from tensorboard.plugins import projector
from datetime import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, DepthwiseConv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda, BatchNormalization
#----------------------------------------------------------------------------
# Config.

PATH        = '../data/dogfacenet/aligned/after_4_bis/' # Path to the directory of the saved dataset
PATH_SAVE   = '../output/history/'                      # Path to the directory where the history will be stored
PATH_MODEL  = '../output/model/2019.07.29/'             # Path to the directory where the model will be stored
SIZE        = (224,224,3)                               # Size of the input images
TEST_SPLIT  = 0.1                                       # Train/test ratio

already_step = 1
LOAD_NET    = True                                     # Load a network from a saved model? If True NET_NAME and START_EPOCH have to be precised
NET_NAME    = 'model'                                   # Network saved name
START_EPOCH = 0                                         # Start the training at a specified epoch
NBOF_EPOCHS = 60                                       # Number of epoch to train the network
HIGH_LEVEL  = True                                      # Use high level training ('fit' keras method)
STEPS_PER_EPOCH = 300                                   # Number of steps per epoch
VALIDATION_STEPS = 30                                   # Number of steps per validation
checkpoint_path= './pointfile'
checkpoint_filepath = './pointfile/checkpoint'
#----------------------------------------------------------------------------
# Import the dataset.

assert os.path.isdir(PATH), '[Error] Provided PATH for dataset does not exist.'

print('Loading the dataset...')

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH):
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames,files)
        labels = np.append(labels,np.ones(len(files))*idx)
        idx += 1

print("--------keep_test------:")
print(filenames)
print("--------keep_train------:")
print(labels)
assert len(labels)!=0, '[Error] No data provided.'

print('Done.')

print('Total number of imported pictures: {:d}'.format(len(labels)))

nbof_classes = len(np.unique(labels))
print('Total number of classes: {:d}'.format(nbof_classes))

#----------------------------------------------------------------------------
# Split the dataset.

nbof_test = int(TEST_SPLIT*nbof_classes)

keep_test = np.less(labels,nbof_test)
keep_train = np.logical_not(keep_test)
# print("--------keep_test------:")
# print(keep_test)
# print(type(keep_test))
# print("--------keep_train------:")
# print(keep_train)
filenames_test = filenames[keep_test]
labels_test = labels[keep_test]

filenames_train = filenames[keep_train]
labels_train = labels[keep_train]
# print("--------labels_train------:")
# print(labels_train)
# print(type(labels_train))
filename_pridict = np.empty(0)
filename_pridict=np.append(filename_pridict,'../data/dogfacenet/aligned/after_4_bis/19/19.0.jpg')
filename_pridict=np.append(filename_pridict, '../data/dogfacenet/aligned/after_4_bis/19/19.1.jpg')
filename_pridict=np.append(filename_pridict, '../data/dogfacenet/aligned/after_4_bis/19/19.2.jpg')
labels_pridict= np.empty(0)
labels_pridict=np.append(labels_pridict,9)
labels_pridict=np.append(labels_pridict,9)
labels_pridict=np.append(labels_pridict,9)
# print("--------filename_pridict------:")
# print(filename_pridict)
# print(type(filename_pridict))

print("Number of training data: " + str(len(filenames_train)))
print("Number of training classes: " + str(nbof_classes-nbof_test))
print("Number of testing data: " + str(len(filenames_test)))
print("Number of testing classes: " + str(nbof_test))

#----------------------------------------------------------------------------
# Loss definition.

alpha = 0.3
def triplet(y_true,y_pred):
    
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)

    return K.sum(tf.nn.relu(ap - an + alpha))

def triplet_acc(y_true,y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)
    
    return K.less(ap+alpha,an)

#----------------------------------------------------------------------------
# Model definition.

if LOAD_NET:
    print('Loading model ')
    model = keras.models.load_model(NET_NAME,custom_objects={'triplet': triplet,'triplet_acc':triplet_acc})
    print('Loading weights to model 1')
    if os.path.isdir(checkpoint_path):
        print('Loading weights to model 2')
        model.load_weights(checkpoint_filepath)

    # print('Loading model from {:s}{:s}.{:d}.h5 ...'.format(PATH_MODEL,NET_NAME,START_EPOCH))
    #
    # model = tf.keras.models.load_model(
    #     '{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,START_EPOCH),
    #     custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})
    print('Done.')
else:


    """
    Model number 12: Paper version: a modified ResNet with Dropout layers and without bottleneck layers
    """

    print('Defining model {:s} ...'.format(NET_NAME))

    # 32 这里定义了embedding 的 大小，初始为32 是否我现在定义为200 他的结果会符合我的需求？
    emb_size = 32

    inputs = Input(shape=SIZE)


    #activate funtion relu Rectified Linear Units  激活函数，使得输出不为零， padding，卷积
    #conv2D 第一个参数filters，设置输出大小 第二个kernel_size 卷积核 第三个strides，滑动步长
    x = Conv2D(16, (7, 7), (2, 2), use_bias=False, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3,3))(x)

    #重复了5次，
    for layer in [16,32,64,128,512]:

        x = Conv2D(layer, (3, 3), strides=(2,2), use_bias=False, activation='relu', padding='same')(x)
        r = BatchNormalization()(x)
        
        x = Conv2D(layer, (3, 3), use_bias=False, activation='relu', padding='same')(r)
        x = BatchNormalization()(x)
        r = Add()([r,x])
        
        x = Conv2D(layer, (3, 3), use_bias=False, activation='relu', padding='same')(r)
        x = BatchNormalization()(x)
        x = Add()([r,x])
        

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(emb_size, use_bias=False)(x)
    outputs = Lambda(lambda x: tf.nn.l2_normalize(x,axis=-1))(x)

    model = tf.keras.Model(inputs,outputs)

    model.compile(loss=triplet,
                optimizer='adam',
                metrics=[triplet_acc])


    print(model.summary())
    model.save(NET_NAME)
    print('Model created.')

#----------------------------------------------------------------------------
# Model training.

if HIGH_LEVEL:
    """
    Hard training: high level of implementation
    """

    histories = []
    crt_loss = 0.6
    crt_acc = 0
    batch_size = 3*10
    nbof_subclasses = 40

    # Create saving folders
    if not os.path.isdir(PATH_MODEL):
        os.makedirs(PATH_MODEL)
    if not os.path.isdir(PATH_SAVE):
        os.makedirs(PATH_SAVE)

    # Bug fixed: keras models are to be initialized by a training on a single batch
    for images_batch,labels_batch in online_adaptive_hard_image_generator(
        filenames_train,
        labels_train,
        model,
        crt_acc,
        batch_size,
        nbof_subclasses=nbof_subclasses):
        h = model.train_on_batch(images_batch,labels_batch)
        break

    logdir = "log/test/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_triplet_acc',
        mode='max',
        save_best_only=False)

    model.fit(
        online_adaptive_hard_image_generator(filenames_train, labels_train, model, crt_loss, batch_size,
                                             nbof_subclasses=nbof_subclasses),
        initial_epoch=already_step,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=NBOF_EPOCHS,
        callbacks=[tensorboard_callback,model_checkpoint_callback],
        validation_data=image_generator(filenames_test, labels_test, batch_size, use_aug=False),
        validation_steps=VALIDATION_STEPS)

    # for i in range(START_EPOCH,START_EPOCH+NBOF_EPOCHS):
    #     print("Beginning epoch number: "+str(i))
    #
    #     hard_triplet_ratio = np.exp(-crt_loss * 10 / batch_size)
    #     nbof_hard_triplets = int(batch_size//3 * hard_triplet_ratio)

        # print("Current hard triplet ratio: " + str(hard_triplet_ratio))
        # logdir = "log/test/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        # model.fit(
        #     online_adaptive_hard_image_generator(filenames_train,labels_train,model,crt_loss,batch_size,nbof_subclasses=nbof_subclasses),
        #     steps_per_epoch=STEPS_PER_EPOCH,
        #     epochs=1,
        #     callbacks=[tensorboard_callback],
        #     validation_data=image_generator(filenames_test,labels_test,batch_size,use_aug=False),
        #     validation_steps=VALIDATION_STEPS)


        #history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
        #这里epochs 用了1 而不是NBOF_EPOCHS（现在设置为60）要求的值，而通过上面的for循环60次，使得实际上是调用了60次model.fit，而不是进行了60个epochs
        #
        # histories += [model.fit(
        #     online_adaptive_hard_image_generator(filenames_train,labels_train,model,crt_loss,batch_size,nbof_subclasses=nbof_subclasses),
        #     steps_per_epoch=STEPS_PER_EPOCH,
        #     epochs=10,
        #     callbacks=[tensorboard_callback],
        #     validation_data=image_generator(filenames_test,labels_test,batch_size,use_aug=False),
        #     validation_steps=VALIDATION_STEPS)]
        #
        # crt_loss = histories[-1].history['loss'][0]
        # crt_acc = histories[-1].history['triplet_acc'][0]


        #预测功能
        # print("-----predict model now1-----")
        # h, w, c = SIZE
        # print("-----predict model now2-----")
        # images = np.empty((len(filename_pridict), h, w, c))
        # print("-----predict model now3-----")
        # for i, f in enumerate(filename_pridict):
        #     print(i)
        #     print(f)
        #     images[i] = sk.io.imread(f) / 255.0
        # print("-----predict model now4-----")
        # predictions = model.predict(images)
        # print("-----predict model now5-----")
        # print(predictions)
        # print("-----predict model now6-----")
        # print(np.argmax(predictions, axis=1))




        # Save model
        # print(model.summary())
        # print("model name--------------------")
        # print('{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,i))
        # model.save
        # model.save('{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,i))



        # Save history
        # loss = np.empty(0)
        # val_loss = np.empty(0)
        # acc = np.empty(0)
        # val_acc = np.empty(0)
        #
        # for history in histories:
        #     loss = np.append(loss,history.history['loss'])
        #     val_loss = np.append(val_loss,history.history['val_loss'])
        #     acc = np.append(acc,history.history['triplet_acc'])
        #     val_acc = np.append(val_acc,history.history['val_triplet_acc'])
        #
        # history_ = np.array([loss,val_loss,acc,val_acc])
        # np.save('{:s}{:s}.{:d}.npy'.format(PATH_SAVE,NET_NAME,i),history_)

    print(model.summary())
    model.save(NET_NAME)

else:
    print("需要反注释以下内容，并再次开发")
#     """
#     Training: lower level of implementation
#     """
#
#     max_epoch = NBOF_EPOCHS + START_EPOCH
#
#     max_step = 300
#     max_step_test = 30
#     batch_size = 3*10
#
#
#     tot_loss_test = 0
#     mean_loss_test = 0
#
#     tot_acc_test = 0
#     mean_acc_test = 0
#
#     # Save
#     loss = []
#     val_loss = []
#     acc = []
#     val_acc = []
#
#     for epoch in range(START_EPOCH,max_epoch):
#
#         step = 1
#
#         tot_loss = 0
#         mean_loss = 0
#
#         tot_acc = 0
#         mean_acc = 0
#
#         # Training
#         for images_batch,labels_batch in online_adaptive_hard_image_generator(
#             filenames_train,
#             labels_train,
#             model,
#             mean_acc,
#             batch_size,
#             nbof_subclasses=10
#             ):
#
#
#             h = model.train_on_batch(images_batch,labels_batch)
#             tot_loss += h[0]
#             mean_loss = tot_loss/step
#             tot_acc += h[1]
#             mean_acc = tot_acc/step
#             #clear_output()
#
#             # hard_triplet_ratio = np.exp(-mean_loss * 10 / batch_size)
#             hard_triplet_ratio = max(0,1.2/(1+np.exp(-10*mean_acc+5.3))-0.19)
#
#             print(
#                 "Epoch: " + str(epoch) + "/" + str(max_epoch) +
#                 ", step1234: " + str(step) + "/" + str(max_step) +
#                 ", loss: " + str(mean_loss) +
#                 ", acc: " + str(mean_acc) +
#                 ", hard_ratio: " + str(hard_triplet_ratio)
#             )
#             print(
#                 "Test loss: " + str(mean_loss_test) +
#                 ", test acc: " + str(mean_acc_test)
#             )
#
#             if step == max_step:
#                 break
#             step+=1
#
#         loss += [mean_loss]
#         acc += [mean_acc]
#
#         print("we are test now")
#         # Testing
#         step = 1
#
#         tot_loss_test = 0
#         mean_loss_test = 0
#
#         tot_acc_test = 0
#         mean_acc_test = 0
#
#         for images_batch,labels_batch in image_generator(filenames_test,labels_test,batch_size,use_aug=False):
#             h = model.test_on_batch(images_batch,labels_batch)
#
#             tot_loss_test += h[0]
#             mean_loss_test = tot_loss_test/step
#             tot_acc_test += h[1]
#             mean_acc_test = tot_acc_test/step
#
#             if step == max_step_test:
#                 break
#             step+=1
#
#         val_loss += [mean_loss_test]
#         val_acc += [mean_acc_test]
#
#         # Save
#         model.save('{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,epoch))
#         history_ = np.array([loss,val_loss,acc,val_acc])
#         np.save('{:s}{:s}.{:d}.npy'.format(PATH_SAVE,NET_NAME,epoch),history_)