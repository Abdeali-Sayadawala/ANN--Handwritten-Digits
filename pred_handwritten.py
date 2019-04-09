import keras
from keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf

# Loading the Model that we created by training
Network = load_model('my_model.h5')

# Import our image to make prediction
img = cv2.imread('cus_eg/4.jpg', 0)
gray = cv2.resize(255-img, (28, 28))

# Converting our image to array to pass it in out network
im = np.array(gray)
im = im.astype('float32')
imr = im.reshape(1, 784)
imr /= 255

# Predicting class of our image from our model
y_pred = Network.predict_classes(imr)
print(y_pred)
