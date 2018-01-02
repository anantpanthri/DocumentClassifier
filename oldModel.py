from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Conv2D, Dense
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
# installed TensorFLow
# installed Theano
# install Keras

from datetime import datetime
print("time now is:",str(datetime.now()))


# initialise CNN
classifier = Sequential()

# convolution Layer/convolutonal Kernel number of rows
# starting with 32 then adding more layers , working on CPU
# we created 32 feature maps of 3 by 3 dimensions
# using RGB images. telling fomrats of images
# input shape coz we are using tensor flow backend
# classifying image is a non linear problem thus we will be using RELU activation function in our NN
# Removing the linearity by means negative values.

classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# applying MaxPooling to reduce the feature map again
# to reduce the number of nodes in the flattenig steps
# flatenning the steps making the model less computive intensive
# wihtout loosing thr pefomance of the model
# dont wanna loose the spatial structure and time complexity
# by taking 2x2 recommended reduce the size of the feature map and reduce the complexity w/o reducing complexity of the model
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# flatennign for ANN
# keras automaticaly understands it
classifier.add(Flatten())

# Creating a ANN using this that as input layer
# Full connection step
# ANN is a great classifier of a NON Linear problem
# Dense function is used to add a Fully connected layer
# output param is the number of nodes in the hidden layer
# 32 features map so many nodes neither a too small or not large thus output param is 128 can change due to experimentation
# rectifier activation function we neend
# softmax AF if more than two output that is sigmoid activation function
classifier.add(Dense(units=128 , activation="relu"))
# we need just a predicted probabilty of one class thus 1
classifier.add(Dense( units=1 , activation="sigmoid"))

# compile CNN
# loss=for yes sale_title or advance chargers
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# image preprocesing , image augmentation to prevent outfitting if not done will result into great result in training set
# but poor result in test sets
# https://keras.io/preprocessing/image/
# https://keras.io/preprocessing/text/
# to do rnd about text preprocessing

# we need to find some patterns in pixels or more images. In data augemntation it will create many batches like rotating shifting etc
# thus a lot or materials to train
# these are random samples
# image augmentation will enrich our model with small amount of images
from keras.preprocessing.image import ImageDataGenerator

# rescaling all pixels value to 0 and 1
# genration enough transfomation to not find same changes

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
# targate size smae as our classifier
training_set = train_datagen.flow_from_directory('images/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('images/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                          steps_per_epoch=8001/32,
                          epochs=25,
                          validation_data=test_set,
                          validation_steps=2002/32)
#import numpy as np
#from keras.preprocessing import image

#to predict images
#image to be imported as the same dimension as the dimension of the test image
test_image = image.load_img('images/prediction/test.jpg' , target_size=(64,64))
#Input layer is 64x64x3 the 3 is for RGB and our test image is only 2 d.  this will change it to 3Dimension
test_image=image.img_to_array(test_image)
test_image=test_image/255.0
#we need 4D so addinng an extra dimension to this. the inputs must be a batch whether one or several.
#axis is the index of the new dimension we are adding.
test_image=np.expand_dims(test_image, axis=0)
result=classifier.predict(test_image)
print(result)
print(training_set.class_indices)

# #serialising the model to  be used later on as a JSON
classifier_json=classifier.to_json()
with open('model.json',"w") as json_file:
     json_file.write(classifier_json)

# serialise weights to HDF5
classifier.save_weights("model.h5")
print('save model to disk')

print("total time for it to run is:",str(datetime.now()))

