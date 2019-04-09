import keras
from keras_preprocessing import image
from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import cv2
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


Network = load_model('my_model.h5')

img = cv2.imread('cus_eg/4.jpg', 0)
gray = cv2.resize(255-img, (28, 28))

im = np.array(gray)
im = im.astype('float32')
imr = im.reshape(1, 784)
imr /= 255
y_pred = Network.predict_classes(imr)
print(y_pred)
