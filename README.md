# DogFaceNet
FaceNet implementation for dog identification.

The stable version is in dogfacenet/dogfacenet_v8.ipynb. It contains
* the model definition and training
* the model evaluation on clustering and one-shot learning

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

* (output: contains the trained models and the convergence curves) not available right now...
* tmp: archive of old codes and tests

### Results on face verification
The current version of the code reaches 97% accuracy on an open-set (48 unknown dogs) of pairs of dogs pictures. That is to say that for a pair of pictures representing either the same dog or two different dogs, the current code could tell if it is the same dog or not with an accuracy of 97%.

Here is the corresponding ROC curve:
![picture alt](https://github.com/GuillaumeMougeot/DogFaceNet/blob/master/images/roc.png)