import numpy as np 
import os
import skimage as sk


PATH = '../data/dogfacenet/aligned/after_4_bis/'
PATH_2 = '../data/dogfacenet/aligned/after_4_resized_2/'
PATH_SAVE = '../output/images/dcgan/dogs/'
PATH_MODEL = '../output/model/gan/'
id = 0
for root,_,files in os.walk(PATH):
    if len(files)>1:
        for i in range(len(files)):
            tmp = sk.io.imread(root + '/' + files[i])
            tmp_resized = sk.transform.resize(tmp,(28,28,3))
            print(np.max(tmp))
            id+=1
            print(id)
            break
        break
