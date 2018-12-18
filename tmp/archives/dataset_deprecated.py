
import numpy as np
import os

def get_dataset(PATH_BG, PATH_DOG1, TRAIN_SPLIT):
    # Retrieve filenames
    filenames_bg = []
    for file in os.listdir(PATH_BG):
            if ".jpg" in file:
                    filenames_bg += [file]

    filenames_dog1 = []
    for file in os.listdir(PATH_DOG1):
            if ".jpg" in file:
                    filenames_dog1 += [file]

    # Splits the dataset

    split_dog1 = int(TRAIN_SPLIT * len(filenames_dog1))
    split_bg = int(TRAIN_SPLIT * len(filenames_bg))

    ## Training set

    filenames_dog1_train = filenames_dog1[:split_dog1]
    filenames_bg_train = filenames_bg[:split_bg]

    filenames_train = np.append(
            [PATH_DOG1 + filenames_dog1_train[i] for i in range(len(filenames_dog1_train))],
            [PATH_BG + filenames_bg_train[i] for i in range(len(filenames_bg_train))],
            axis=0
            )
    # labels_train = [1,1,1,...,1,2,3,...,len(filenames_bg_train)] <-- there are len(filenames_dog1_train) ones
    labels_train = np.append(np.ones(len(filenames_dog1_train)), np.arange(2,2+len(filenames_bg_train)))

    assert len(filenames_train)==len(labels_train)

    ## Validation set
    filenames_dog1_valid = filenames_dog1[split_dog1:]
    filenames_bg_valid = filenames_bg[split_bg:]

    filenames_valid = np.append(
            [PATH_DOG1 + filenames_dog1_valid[i] for i in range(len(filenames_dog1_valid))],
            [PATH_BG + filenames_bg_valid[i] for i in range(len(filenames_bg_valid))],
            axis=0
            )

    # labels_valid = [1,1,1,...,1,labels_train[-1]+1,labels_train[-1]+2,...,len(filenames_bg_valid)+labels_train[-1]+1]
    # <-- there are len(filenames_dog1_valid) ones
    labels_valid = np.append(np.ones(len(filenames_dog1_valid)), np.arange(labels_train[-1]+1,labels_train[-1]+1+len(filenames_bg_valid)))
    return filenames_train, labels_train, filenames_valid, labels_valid