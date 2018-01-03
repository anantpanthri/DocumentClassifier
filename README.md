# DocumentClassifier
Used Convoluted neural networks to classify images/documents.
Prereq-> Keras lib, python 3.5+
This project requiers a sample excel file containing the list of all the images to be downloaded then it will download the images into the respective folders(testing and training directories)
Based on it the classification of images begins.
Runs great if you have a gpu processor.
The model trains itself based on weights and is later saved in the system as model.h5 and model.json
The "to be predicted" images is stored in the predicted directory and is classified upto the approximation of 85%.
