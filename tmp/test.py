import os
import numpy as np


PATH        = '../data/dogfacenet/aligned/after_4_bis/' # Path to the directory of the saved dataset
PATH_SAVE   = '../output/history/'                      # Path to the directory where the history will be stored
PATH_MODEL  = '../output/model/2019.07.29/'             # Path to the directory where the model will be stored
SIZE        = (224,224,3)                               # Size of the input images
TEST_SPLIT  = 0.1                                       # Train/test ratio

LOAD_NET    = False                                     # Load a network from a saved model? If True NET_NAME and START_EPOCH have to be precised
NET_NAME    = '2019.07.29.dogfacenet'                   # Network saved name
START_EPOCH = 0                                         # Start the training at a specified epoch
NBOF_EPOCHS = 250                                       # Number of epoch to train the network
HIGH_LEVEL  = True                                      # Use high level training ('fit' keras method)
STEPS_PER_EPOCH = 300                                   # Number of steps per epoch
VALIDATION_STEPS = 30                                   # Number of steps per validation

#----------------------------------------------------------------------------
# Import the dataset.

assert os.path.isdir(PATH), '[Error] Provided PATH for dataset does not exist.'

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
directs = np.empty(0)
for root,dirs,files in os.walk(PATH):
    if len(files)>1:
        directs = np.append(directs,root[2:])
        for i in range(len(files)):
            files[i] = str.split(root[2:],'/')[-1]
        filenames = np.append(filenames,files)
        labels = np.append(labels,np.ones(len(files))*idx)
        idx += 1
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

filenames_test = filenames[keep_test]
labels_test = labels[keep_test]

filenames_train = filenames[keep_train]
labels_train = labels[keep_train]

print("Number of training data: " + str(len(filenames_train)))
print("Number of training classes: " + str(nbof_classes-nbof_test))
print("Number of testing data: " + str(len(filenames_test)))
print("Number of testing classes: " + str(nbof_test))

np.savetxt('classes_train.txt',filenames_train,fmt='%s')
np.savetxt('classes_test.txt',filenames_test,fmt='%s')