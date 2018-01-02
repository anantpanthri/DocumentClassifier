from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Conv2D, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

json_file=open('model_dropout.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("model_dropout.h5")
print('model successfuly loaded')

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1. / 255)
# training_set = train_datagen.flow_from_directory('images/training_set',
#                                                  target_size=(64, 64),
#                                                  batch_size=32,
#                                                  class_mode='binary')


test_image = image.load_img('images/prediction/adv.png' , target_size=(150,150))
test_image=image.img_to_array(test_image)
test_image=test_image/255.0
test_image=np.expand_dims(test_image, axis=0)
result=loaded_model.predict(test_image)
# print(training_set.class_indices)

print(result)
if np.round(result, 0) == 1:
    print("The doc is of a saletittle")
else:
    print("The doc is of a advancecharge")


import matplotlib.pyplot as plt

labels = 'AdvanceCharges','Saletitle'
sizes = [result,1-result[0] ]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()