"""
DogFaceNet
Dataset retrieving for landmarks detection

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import numpy as np
import os
import skimage as sk
import pandas as pd
from ast import literal_eval # string to dict
import matplotlib.pyplot as plt


############################################################
#  Data pre-processing for landmarks detection
############################################################


def get_landmarks_dataset(path='../data/landmarks/', split=0.8):
    csv_path = path
    for file in os.listdir(path):
        if '.csv' in file:
            csv_path += file
    df = pd.read_csv(csv_path)

    index = df.index
    filenames = df.loc[:,'filename']
    region_id = df.loc[:,'region_id']
    dictionary = [literal_eval(df.loc[:,'region_shape_attributes'][i]) for i in range(len(index))]

    images = np.empty(())