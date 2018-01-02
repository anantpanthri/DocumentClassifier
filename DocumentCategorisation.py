from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

# Image dimensions
img_width, img_height = 150, 150

"""
    Creates a CNN model
    p: Dropout rate
    input_shape: Shape of input 
"""


def create_model(p, input_shape=(32, 32, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p / 2))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the CNN
    optimizer = Adam(lr=1e-3)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model


"""
    Fitting the CNN to the images.
"""


def run_training(bs=32, epochs=25):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('images/training_set',
                                                     target_size=(img_width, img_height),
                                                     batch_size=bs,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('images/test_set',
                                                target_size=(img_width, img_height),
                                                batch_size=bs,
                                                class_mode='binary')

    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))
    model.fit_generator(training_set,
                        steps_per_epoch=8000 / bs,
                        epochs=epochs,
                        validation_data=test_set,
                        validation_steps=2000 / bs)

    # #serialising the model to  be used later on as a JSON
    classifier_json = model.to_json()
    with open('model_dropout.json', "w") as json_file:
        json_file.write(classifier_json)

    # serialise weights to HDF5
    model.save_weights("model_dropout.h5")
    print('save model to disk')

    # test_image = image.load_img('images/prediction/adv.png', target_size=(64, 64))
    # # Input layer is 64x64x3 the 3 is for RGB and our test image is only 2 d.  this will change it to 3Dimension
    # test_image = image.img_to_array(test_image)
    # test_image = test_image / 255.0
    # # we need 4D so addinng an extra dimension to this. the inputs must be a batch whether one or several.
    # # axis is the index of the new dimension we are adding.
    # test_image = np.expand_dims(test_image, axis=0)
    # result = model.predict(test_image)
    # print(result)
    # print(training_set.class_indices)



def main():
    run_training(bs=32, epochs=25)


""" Main """
if __name__ == "__main__":
    main()