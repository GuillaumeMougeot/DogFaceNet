import os
from tqdm import tqdm
from shutil import copyfile

PATH = '../../data/dogfacenet/images/'
PATH_dest = '../../data/dogfacenet/aligned/'

count_classes = 0
mean = 0.

for root,dirs,files in tqdm(os.walk(PATH)):
    if len(files) > 6:
        idx = -1
        for i in range(-1,-len(root),-1):
            if root[i] == '/':
                idx = i+1
                break
        for i in range(len(files)):
            copyfile(root+'/'+files[i],PATH_dest+root[idx:]+'.'+files[i])

        count_classes += 1
        mean += len(files)

print("Number of classes: " + str(count_classes))
print("Mean number of images per classes: " + str(mean/count_classes))