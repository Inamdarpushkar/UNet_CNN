# UNet CNN
# UNet model to segment building coverage in Boston using Remote sensing data

I implemented the UNet Convolutional Neural Network architecture for segmenting the building
footprints from the image. Here I aimed at classifying each pixel representing building footprints.
It is important to note that I specifically focused on the buildings and did not consider any other
features like roads, bridges, etc. I used high resolution aerial image provided by the USGS. 

<img width="630" alt="Screen Shot 2019-11-04 at 2 54 43 PM" src="https://user-images.githubusercontent.com/28696943/68165501-73e00200-ff14-11e9-9753-b993d5303be5.png">

# Data preparation and methods

I used a cloud-free 2D- high resolution latest available aerial image to create the training dataset.
One advantage of using 2D Aerial data over google images was in generating a binary building
mask layer. The image was georeferenced with a constant scale so it would also help in precisely
estimating the building coverage in metric units.
To create a binary raster (mask), I used reference vector building layer (ground truth) available on
the ArcGIS online platform. Then I adjusted the image and vector layers to the same coordinate
systems and converted original RGB layer to corresponding binary (Building:1, Other:0) data. I
clipped the RGB and Masks images iteratively with (128,128) image dimensions. I used a
threshold of 20% on the building coverage in the mask layer and considered only images with
sufficient building coverage. After applying the threshold value, size of the data reduced to 792. I
have split this data into a train (80%), test (10%) and validation(10%) sets. I then normalized, and
shuffled the dataset.
This processing ended up in a very small dataset. I thought of using a substantial augmentation on
this training dataset to maintain reasonable amount of images during training phase. Unfortunately,
the training on augmented data did not progress well. So I ended up implementing the UNet model
on this small dataset.
I trained the UNet model from scratch. I changed the dimensions of the input images to 128,128.
As used in a similar study by Chorr et al, I modified the padding to the ‘same’, to avoid shrinking
when doing convolutions. Also, I have added the batch normalization after each ReLU activation
and used Adam Optimizer for faster convergence during training phase. I did training on smallbatches of 8 images with 50 epochs. I used Keras callback ReduceLROnPlateau to reduce the
learning rate by a factor of 0.1. 
