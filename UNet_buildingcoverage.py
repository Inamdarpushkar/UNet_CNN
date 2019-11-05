
# Commented out IPython magic to ensure Python compatibility.
## Imports
import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt
# %matplotlib inline

## Seeding 
seed = 9
random.seed = seed
np.random.seed = seed
tf.seed = seed

"""# Image Data Generator"""

class Image_DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path: images and masks folder
        image_path = os.path.join(self.path, id_name, "images", id_name) + ".png"
        mask_path = os.path.join(self.path, id_name, "masks", id_name)+ ".png"
        
        
        ## Reading Images and associated masks
        image = cv2.imread(image_path, 1)
        mask=cv2.imread(mask_path,1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = np.zeros((self.image_size, self.image_size, 1))
        
        ## Reading building footprint Masks
        mask_image = cv2.imread(mask_path)
        mask_image= cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        mask_image = cv2.resize(mask_image, (self.image_size, self.image_size)) #128x128
        mask_image = np.expand_dims(mask_image, axis=-1)
        mask = np.maximum(mask, mask_image)
            
        ## Normalizaing image (RGB) and masks:
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

"""# Hyperparameters"""

##inputs
image_size = 128
train_path = "/Users/Pushkar/Desktop/xy/sat/training_data/"
epochs = 50
batch_size = 8

##Ids
data_ids = next(os.walk(train_path))[1]

## Split dataset ids to 80%/10%/10% tran/val/test ratio
train_ids, valid_ids, test_ids = np.split(data_ids, [int(.8 * len(data_ids)), int(.9 * len(data_ids))])

"""* Plot RGB color image and associated ground truth building footprints"""

image_gen = Image_DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
x, y = image_gen.__getitem__(1)
print(x.shape, y.shape)

r = random.randint(0, len(x)-1)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(x[r])
ax.set_title('Original_RGB')

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(y[r], (image_size, image_size)), cmap="gray")
ax.set_title('Building_mask')

"""# UNET Architecture 
## i. Down sampling
## ii. Bottleneck
## iii. Upsampling
"""

#Convolutional layers in UNnet 

def Unet_down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = BatchNormalization(axis=3)(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def Unet_up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = BatchNormalization(axis=3)(x)
    us = keras.layers.UpSampling2D((2, 2))(c)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def Unet_bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = BatchNormalization(axis=3)(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


# UNet model
def UNet_model():
    size = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = Unet_down_block(p0, size[0]) #128 > 64
    c2, p2 = Unet_down_block(p1, size[1]) #64 > 32
    c3, p3 = Unet_down_block(p2, size[2]) #32 > 16
    c4, p4 = Unet_down_block(p3, size[3]) #16 > 8
    
    
    bn = Unet_bottleneck(p4, size[4])
    
   
    u1 = Unet_up_block(bn, c4, size[3]) #8 > 16
    u2 = Unet_up_block(u1, c3, size[2]) #16 > 32
    u3 = Unet_up_block(u2, c2, size[1]) #32 > 64
    u4 = Unet_up_block(u3, c1, size[0]) #64 > 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

"""# Metrics"""

#f1, recall, precision metics 

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

model = UNet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc",f1_m,recall_m,precision_m])
model.summary()

"""# Callbacks"""

#Save the best model
def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save, reduce_lr_loss]
name_weights = "Final_best_boston" + "_weights.h5"
callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)

"""# Model Training"""

#generating train, test, validation datasets
train_gen = Image_DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = Image_DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)
test_gen = Image_DataGen(test_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

history=model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=epochs,callbacks=callbacks,shuffle=True)

# Load the best model
model.load_weights('Final_best_boston_weights.h5')

## Dataset for prediction
x, y = valid_gen.__getitem__(1)
result = model.predict(x)
result = result > 0.5

#test data for evaluation
g=Image_DataGen(test_ids, train_path, batch_size=batch_size, image_size=image_size)
x=[]
y=[]
for i in test_ids:
    x_test,y_test=g.__load__(i)
    x.append(x_test)
    y.append(y_test)
    
x=np.asarray(x)
y=np.asarray(y)

"""# Testing the model"""

#comparison of true color image, ground truth masks and unet predicted output on validation data
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x, y = valid_gen.__getitem__(8)
result = model.predict(x)

result = result > 0.5

ax = fig.add_subplot(2, 3, 1)
ax.imshow(x[0])
ax.set_title('True Color image')


ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('Building Footprints')

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('U-net Predicted')

fig.savefig('valid1.png')

#comparison of true color image, ground truth masks and unet predicted output on a validation data
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x, y = valid_gen.__getitem__(1)
result = model.predict(x)


result = result > 0.5

ax = fig.add_subplot(2, 3, 1)
ax.imshow(x[0])
ax.set_title('True Color image')


ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('Building Footprints')

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('U-net Predicted')

fig.savefig('valid1.png')
x.shape,y.shape,result[1].shape
fig.savefig('valid2.png')

#comparison of true color image, ground truth masks and unet predicted output on an unseen test data
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x, y = test_gen.__getitem__(1)
result = model.predict(x)
result = result > 0.5

ax = fig.add_subplot(2, 3, 1)
ax.imshow(x[0])
ax.set_title('True Color image')


ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('Building Footprints')

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('U-net Predicted')

fig.savefig('valid1.png')
x.shape,y.shape,result[1].shape

x.shape,y.shape,result[1].shape
fig.savefig('test1.png')

#comparison of true color image, ground truth masks and unet predicted output on an unseen test data
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x, y = test_gen.__getitem__(2)
result = model.predict(x)
result = result > 0.5

ax = fig.add_subplot(2, 3, 1)
ax.imshow(x[0])
ax.set_title('True Color image')


ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('Building Footprints')

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('U-net Predicted')

fig.savefig('valid1.png')
x.shape,y.shape,result[1].shape

x.shape,y.shape,result[1].shape
fig.savefig('test2.png')

#comparison of true color image, ground truth masks and unet predicted output on an unseen test data
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x, y = test_gen.__getitem__(3)
result = model.predict(x)
result = result > 0.5

ax = fig.add_subplot(2, 3, 1)
ax.imshow(x[0])
ax.set_title('True Color image')


ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('Building Footprints')

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('U-net Predicted')

fig.savefig('valid1.png')
x.shape,y.shape,result[1].shape

x.shape,y.shape,result[1].shape
fig.savefig('test3.png')

#comparison of true color image, ground truth masks and unet predicted output on an unseen test data
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x, y = test_gen.__getitem__(4)
result = model.predict(x)
result = result > 0.5

ax = fig.add_subplot(2, 3, 1)
ax.imshow(x[0])
ax.set_title('True Color image')


ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('Building Footprints')

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('U-net Predicted')

fig.savefig('valid1.png')
x.shape,y.shape,result[1].shape

x.shape,y.shape,result[1].shape
fig.savefig('test4.png')

#comparison of true color image, ground truth masks and unet predicted output on an unseen test data
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x, y = test_gen.__getitem__(5)
result = model.predict(x)
result = result > 0.5

ax = fig.add_subplot(2, 3, 1)
ax.imshow(x[0])
ax.set_title('True Color image')


ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('Building Footprints')

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('U-net Predicted')

fig.savefig('valid1.png')
x.shape,y.shape,result[1].shape

x.shape,y.shape,result[1].shape
fig.savefig('test5.png')

#comparison of true color image, ground truth masks and unet predicted output on an unseen test data
fig = plt.figure()
fig.subplots_adjust(hspace=0.3, wspace=0.3)

x, y = test_gen.__getitem__(6)
result = model.predict(x)
result = result > 0.5

ax = fig.add_subplot(2, 3, 1)
ax.imshow(x[0])
ax.set_title('True Color image')


ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('Building Footprints')

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")
ax.set_title('U-net Predicted')

fig.savefig('valid1.png')
x.shape,y.shape,result[1].shape

x.shape,y.shape,result[1].shape
fig.savefig('test6.png')

#model evaluation
hist=history
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
train_f1m=hist.history['f1_m']
val_f1m=hist.history['val_f1_m']
xc=range(epochs)

#Plot train,val loss
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
plt.savefig('loss.png')

#Plot train,val accuracy
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.show()
plt.savefig('Acc.png')

#Plot train,val dice coefficients
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_f1m)
plt.plot(xc,val_f1m)
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.grid(True)
plt.legend(['train','val'],loc='lower right')
plt.style.use(['classic'])
plt.show()
plt.savefig('dice.png')

#All test data for evaluation
g=Image_DataGen(test_ids, train_path, batch_size=len(test_ids), image_size=image_size)
x=[]
y=[]
for i in test_ids:
    x_test,y_test=g.__load__(i)
    x.append(x_test)
    y.append(y_test)
    
x=np.asarray(x)
y=np.asarray(y) 
#model evaluation
model.evaluate(x,y,verbose=1),print(model.metrics_names)

