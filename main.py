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
AUG_DIR = 'augmented/'
ORIGINAL_DIR = 'original/'

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



def load_data(directories = None , IMG_WIDTH=280, IMG_HEIGHT=280):
    X = []
    y = []
    image_width = IMG_WIDTH
    image_height = IMG_HEIGHT
    #aug_yes = AUG_DIR + 'yes'
    #aug_no = AUG_DIR + 'no'
    aug_yes = 'aug/yes'
    aug_no = 'aug/no'
    directories = [aug_yes, aug_no]
    for directory in directories:
        print("In directory __ in directories")
        i = 0
        for filename in os.listdir(directory):
            if i>= 300:
                print(f"Done with {i} images")
                break
            image = cv2.imread(directory + '/' + filename)
            image = image / 255.
            X.append(image)
            i+=1
            if directory[-2:] == 'no':
                y.append([0])
            else:
                y.append([1])
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    return X, y


def load_image(image, IMG_WIDTH=300, IMG_HEIGHT=350):
    image_width = IMG_WIDTH 
    image_height = IMG_HEIGHT
    image = crop_brain_contour(image, plot=False)
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    image = image / 255.
    print_image(image)
    print(f'Added Image')
    return image

def print_image(image):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False,labelleft=False,   labelright=False)
    plt.title('Cropped image')
    plt.show()



X = []
y = []
X, y = load_data()

def split_data(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)
X = []
y = []
def data_details():
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of development examples = " + str(X_val.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    print ("X_val (dev) shape: " + str(X_val.shape))
    print ("Y_val (dev) shape: " + str(y_val.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(y_test.shape))

data_details()

#arr = y_train or y_test or y_val	
arr = y_train
num_zeros = np.count_nonzero(arr == 0)
num_ones = np.count_nonzero(arr == 1)

print(f"The number of zeroes in the array is {num_zeros}.")
print(f"The number of ones in the array is {num_ones}.")


####################################################################################
inp_shp = (350, 300, 3)
pool_shape = (4,4)2
model = Sequential()
model.add(Conv2D (32, (7,7),padding='same', activation='relu',input_shape = inp_shp))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_shape))
model.add(Conv2D (64, (3,3),padding='same', activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_shape))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
############################################################################################
inp_shp = (280, 280, 3)
pool_shape = (4,4)
model = None
model = Sequential()
model.add(Conv2D (32, (7,7),strides = (1,1), activation='relu', input_shape = inp_shp)))
model.add(BatchNormalization(axis=3))
model.add(MaxPooling2D(pool_shape))
model.add(MaxPooling2D(pool_shape))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
#from tensorflow.keras.layers import  Dropout
#model.add(Dropout(0.25)
############################################################################################
def build_model():
    input_shape = (240, 240, 3)
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) # shape=(?, 238, 238, 32)
    X = MaxPooling2D((4, 4), name='max_pool0')(X) # shape=(?, 59, 59, 32)
    X = MaxPooling2D((4, 4), name='max_pool1')(X) # shape=(?, 14, 14, 32)
    X = Flatten()(X) # shape=(?, 6272)
    X = Dense(1, activation='sigmoid', name='fc')(X) # shape=(?, 1)
    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')
    return model
model = build_model()
##########################################################################################






##############  CORRECT START  #######################################
log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

filepath="cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}"

checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])


end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")


##############  CORRECT END ##########################################



############## Load trained model and test image ###################
from keras.models import load_model


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

def execute(a, b):
    test_images = load_test_images(None, a, b)
    shape = tf.shape(test_images)
    print(shape)
    predictions = model.predict(test_images)
    num_true = 0
    num_false = 0
    for prediction in predictions:
        if prediction < 0.5:
            num_true+=1
            print("No tumor detected")
            print(f"Confidence: {(1-prediction)*100}")
        else:
            num_false+=1
        print(f"Tumor Detected  Confidence: {prediction*100}")
    accuracy = 100*num_true/(num_true+num_false)
    print(f"Positive Cases: {num_true}") 
    print(f"Negative Cases: {num_false}")
    print(f"Accuracy: {accuracy}")     
##################################################################### 



















history = model.history.history

for key in history.keys():
    print(key)

def plot_metrics(history):
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['acc']
    val_acc = history['val_acc']
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
















