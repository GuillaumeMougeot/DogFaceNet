# DogFaceNet

This code is an implementation of a deep learning method for dog identification. It relies on the triplet loss defined in FaceNet paper and on novel deep learning techniques as ResNet networks.

Dog faces pictures were retrieved from the web and aligned using three handmade labels. We used VIA tool to label the images. The dataset is available here: [Zenodo](https://zenodo.org/records/12578449) and the old dataset here: [Releases Page](https://github.com/GuillaumeMougeot/DogFaceNet/releases/).

This code contains also an automatic face alignement tool and several implementation of GANs (Generative Adverserial Networks) onto the dataset.

### Citation

If ever this project (code or dataset) is used for a publication, please cite: [A Deep Learning Approach for Dog Face Verification and Recognition](https://link.springer.com/chapter/10.1007/978-3-030-29894-4_34)
```
@InProceedings{10.1007/978-3-030-29894-4_34,
author="Mougeot, Guillaume and Li, Dewei and Jia, Shuai",
editor="Nayak, Abhaya C. and Sharma, Alok",
title="A Deep Learning Approach for Dog Face Verification and Recognition",
booktitle="PRICAI 2019: Trends in Artificial Intelligence",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="418--430",
isbn="978-3-030-29894-4"
}
```

### Dataset

**[NEWS 2024/06/28] The complete dataset is now available on [Zenodo](https://zenodo.org/records/12578449).**

The old dataset is still available for download on the releases page: [Releases Page](https://github.com/GuillaumeMougeot/DogFaceNet/releases/)

### Run the recognition algorithm

To run the code you will need:
* python >= 3.6.4
* tensorflow == 1.12.0 (this constraint will be improved)
* numpy >= 1.14.0
* matplotlib >= 2.1.2
* scikit-image >= 0.13.1
* jupyter >= 1.0.0 (optional: only for dev)
* tqdm >= 4.23.4 (optional: only for dev)

Then run the following command from the root directory of the project:

    python dogfacenet/dogfacenet.py

To run properly the dataset has to be located in a data/dogfacenet folder or you will have to edit the config part of the dogfacenet.py file.

The above command will train a model and save it into output/model directory. It will also save its history in output/history.

### Content

As previously described, the stable version is in dogfacenet/dogfacenet.py. It contains:

* the online and offline training modules
* the model definition and training
* the model evaluation (still in development)

The dogfacenet-dev folder contains the developer version of the code. Model evaluation (verification, recognition, clustering, ROC curve, observation on the heatmap, ...) is in developer folder. It will be transfer in stable folder soon. The main dev version is in dogfacenet-dev/dogfacenet_v12-dev.ipynb jupyter notebook file.

The rest of the project contains:

* (data: the images of the project) not available right now...
* dogfacenet: stable version of the DogFaceNet project.
    * dogfacenet: dataset loading, model definiton and training
    * offline/online_training: function for triplet generation
* dogfacenet-dev: the main part, it contains the code on dog face verification and on face alignment (in dogfacenet/labelizer).
    * labelizer: contains the data-preprocessing function after labeling the images using VIA
        * copy_images: copies the images from the output folder of VIA to the input folder of DogFaceNet
        * transform_csv_to_clean_format: edits the output csv file from VIA to a adapted format
        * align_face: aligns copied faces using the edited csv file
    * dogfacenet-*dataset*: different tries on different dataset 
    * dogfacenet_v*version_number*: the different version of the code on dog pictures
    * dataset: deprecated
    * losses: deprecated
    * models: deprecated
    * triplet_loss: deprecated
    * triplet_preprocessing: triplets linked functions
        * triplets definition
        * triplets augmentation
        * hard triplets definition
* GAN: a dog face generator in developement...
* landmarks: a try on automatic facial landmarks detection, still in developement...
* output:
    * (model: the trained models not available right now...)
    * history: the convergence curves
* tmp: archive of old codes and tests

### Results on face verification

UPDATE: the now available dataset is bigger (around 8600 pictures) than the one presented in the article (around 1400 pictures). The following results (and the one presented in the paper) were found using the small dataset. The results on the big dataset are unfortunately worse than the one on the small dataset (86% accuracy on the open-set): this can be explained knowing that there are much more dogs per breed in the test set, it thus makes the problem harder (two different dogs of the same breed are very similar).

The current version of the code reaches 92% accuracy on an open-set (48 unknown dogs) of pairs of dogs pictures. That is to say that for a pair of pictures representing either the same dog or two different dogs, the current code could tell if it is the same dog or not with an accuracy of 92%.

Here is the corresponding ROC curve:

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/roc.png)

Here follows some false accepted examples and false rejected ones. The model mistakes are mainly due to light exposure, dogs' posture and occlusions.

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/fa_fr.png)

### Results on face clustering

The obtained code presents great results on face clustering (even for dog faces that the code hasn't seen before).
Here follows is an example of two of these clusters: the left one shows a correct example and the right one shows a mistake.

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/clustering.png)

### The face detector

This part of the code is still in development. The current detector can for now detect dog faces on 64x64 images. Here follows a set of examples given by the current algorithm on a testing set.

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/detector.png)

As the dataset contains only 3 keypoints (the two eyes and the nose) we had to modify the dataset to extract a bounding box. We created a small piece of code that automatically applies a 3D mask on the dog faces using the landmarks. We then considered the middle of the dog brain as the middle of the bounding box. Here follows some examples of computed 3D masks:

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/mask.png)

### GAN

The Generative Adverserial Network created by NVIDIA in https://github.com/tkarras/progressive_growing_of_gans gives the best results on our dataset. Here follows the results obtained using a single GPU (a GTX1080) during a day of training on the dataset:

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/gan.png)
