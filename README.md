# DogFaceNet

This code is an implementation of a deep learning method for dog identification. It relies on the triplet loss defined in FaceNet paper and on novel deep learning techniques as ResNet networks.

Dog faces pictures were retrieved from the web and aligned using three handmade labels. We used VIA tool to label the images.

### Run this code

To run the code you will need:
* python >= 3.6.4
* tensorflow >= 1.12.0
* numpy >= 1.14.0
* matplotlib >= 2.1.2
* scikit-image >= 0.13.1
* jupyter >= 1.0.0
* tqdm >= 4.23.4

### Content


The stable version is in dogfacenet/dogfacenet_v10-stable.ipynb. You can run it into jupyter notebook. It contains:

* the data-preprocessing after alignment
* the model definition and training
* the model evaluation on verification, recognition and clustering

The dataset is not available for now, but coming soon...
The rest of the project contains:

* (data: the images of the project) not available right now...
* landmarks: a try on automatic facial landmarks detection, still in developpement...
* dogfacenet: the main part, it contains the code on dog face verification and on face alignment (in dogfacenet/labelizer).
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

* output contains:
    * (model: the trained models not available right now...)
    * history: the convergence curves
* tmp: archive of old codes and tests


### Results on face verification

The current version of the code reaches 92% accuracy on an open-set (48 unknown dogs) of pairs of dogs pictures. That is to say that for a pair of pictures representing either the same dog or two different dogs, the current code could tell if it is the same dog or not with an accuracy of 92%.

Here is the corresponding ROC curve:

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/roc.png)

Here follows some false accepted examples and false rejected ones. The model mistakes are mainly due to light exposure, dogs' posture and occlusions.

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/fa_fr.png)


### Results on face clustering

The obtained code presents great results on face clustering (even for dog faces that the code hasn't seen before).
Here follows is an example of two of these clusters: the left one shows a correct example and the right one shows a mistake.

![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/clustering.png)
