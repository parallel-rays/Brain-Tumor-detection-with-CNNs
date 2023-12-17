import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir
import os

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s,1)}"
    
def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()
    return new_image
    
    
    
    
    
    
    
    
model = load_model('best_models/cnn-parameters-improvement-3.model')


def load_test_images(directory = None , IMG_WIDTH=300, IMG_HEIGHT=350):
    test_images = []
    image_width = IMG_WIDTH 
    image_height = IMG_HEIGHT
    directory = 'test'
    for filename in os.listdir(directory):
        image = cv2.imread(directory + '/' + filename)
        image = crop_brain_contour(image, plot=False)
        image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
        image = image / 255.
        test_images.append(image)
    test_images = np.array(test_images)
    return test_images


def execute(a=240, b=240):
    test_images = load_test_images(None, a, b)
    shape = tf.shape(test_images)
    print(shape)
    predictions = model.predict(test_images)
    num_true = 0
    num_false = 0
    for prediction in predictions:
        if prediction < 0.5:
            num_true+=1
            print(f"No tumor detected Confidence: {(1-prediction)*100}")
        else:
            num_false+=1
            print(f"Tumor Detected  Confidence: {prediction*100}")
    print(f"Negative Cases: {num_true}")
    print(f"Positive Cases: {num_false}")
    
    

while True:
    print("1. Test Images")
    print("2. Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        execute()
    elif choice == "2":
        break
    else:
        print("Invalid choice. Try again.")
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
