# Import libraries
import os, sys, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl 
import scipy.io
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, MaxPool1D, GaussianNoise, GlobalMaxPooling1D, Conv1D, GlobalAveragePooling1D, Input, concatenate
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import BatchNormalization
import tensorflow
from tensorflow.keras.utils import plot_model


# Function for N X N patches  
def GetPatches_nxn(data, N) :
    
    n = math.floor(N/2)
    n = int(n)
    row, col, ch = data.shape
    num_ele = row * col
    # print("Original ", data.shape)
    padding_top = data[:n, :, :] * 0
    padding_bottom = data[row-n:, :, :] * 0
    data = np.concatenate((padding_top, data, padding_bottom), axis= 0)

    row, col, ch = data.shape
    padding_left = data[:, :n, :] * 0
    padding_right = data[:, col-n:, :] * 0
    data = np.concatenate((padding_left, data, padding_right), axis= 1)
    # print("Final ", data.shape)
    patches = np.zeros((num_ele, N, N, 2)) + np.nan
    num_nans = np.count_nonzero(np.isnan(patches))
    # print(num_nans)
    row, col, ch = data.shape
    count = 0
    for i in range(n, row-n) :
        for j in range(n, col-n) :
            
            # print(i,j)
            # print(i-n, i+n+1)
            # print(j-n, j+n+1)
            # print("i ", i, " j ", j, "row ", row, " col ", col, " top ", i-n, " bottom ", i+n+1, " left ", j-n, " right ", j+n+1)

            assert i-n >= 0
            assert j-n >= 0
            assert i+n+1 <= row
            assert j+n+1 <= col
            patch = data[i-n:i+n+1, j-n:j+n+1, :]
            assert patch.shape == (N,N,2)
            patches[count] = patch
            count += 1

    patches = np.array(patches)

    return patches



# Get path to your current directory
basedir = os.getcwd()

# Path to your dataset
filename = basedir + "/hyperSpec.mat"

# Open .mat file with scipy
hyper = scipy.io.loadmat(filename)

# Hyperspectral data
hyper = hyper['hyper']

# print(hyper.shape)

# Path to truth dataset
truth_file = basedir + "/muufl_gulfport_campus_1_hsi_220_label.mat"
mat = scipy.io.loadmat(truth_file)
hsi = ((mat['hsi'])[0])[0]

# LiDAR
lidar = ((((hsi[-4])[0])[0])[0])[0]

# x, y, z. z contains Height and Intensity
x, y, z, info = lidar[0], lidar[1], lidar[2], lidar[3]

# Ground truth
truth = ((hsi[-2])[0])[-1]
truth = truth[-1]

# print(truth.shape)

patches = GetPatches_nxn(z, 11)
# print(patches.shape)

# Ground truth contains label -1, 1, 2, ..., 11. Label '-1' is unlabelled data. So, we subtract 1 from ground truth.
# Now, the ground truth becomes -2, 0, 1, ..., 10. And we take ground truth >=0.
truth = truth.flatten()
truth = truth - 1
indx, = np.where(truth >= 0)
patches = patches[indx]
truth = truth[indx]

hyper = hyper.reshape(325*220, 64)
hyper = hyper[indx]

# print(patches.shape)
# print(truth.shape)
# print(hyper.shape)

# sys.exit(0)
# Split data into train and test
H_train, H_test, L_train, L_test, y_train, y_test = train_test_split(hyper, patches, truth, test_size= 0.6, random_state = int(time.time()), shuffle = True)

np.savez_compressed(basedir +"/train_test_split_fusion6.npz", H_train = H_train, H_test = H_test, L_train = L_train, L_test = L_test, y_train = y_train, y_test = y_test)

file = np.load(basedir+"/train_test_split_fusion6.npz")
H_train = file['H_train']
H_test = file['H_test']
L_train = file['L_train']
L_test = file['L_test']
y_train = file['y_train']
y_test = file['y_test']

# print(H_train.shape)
# print(L_train.shape)

# One-hot encoding of labels
y_train = to_categorical(y_train, num_classes = 11, dtype ="int32")
y_test = to_categorical(y_test, num_classes = 11, dtype ="int32")

# Lidar Normalization
ch1 = L_train[:, :, :, 0]
pmin = np.amin(ch1)
pmax = np.amax(ch1)
ch1 = (ch1-pmin) / (pmax- pmin) 


ch2 = L_train[:, :, :, 1]
pmin1 = np.amin(ch2)
pmax1 = np.amax(ch2)
ch2 = (ch2-pmin1) / (pmax1- pmin1) 
L_train[:, :, :, 0] = ch1
L_train[:, :, :, 1] = ch2

ch3 = L_test[:, :, :, 0]
ch3 = (ch3-pmin) / (pmax- pmin) 

ch4 = L_test[:, :, :, 1]
ch4 = (ch4-pmin1) / (pmax1- pmin1) 

L_test[:, :, :, 0] = ch3
L_test[:, :, :, 1] = ch4

# HSI 
H_train[H_train > 1] = 1
H_train[H_train < 0] = 0
H_test[H_test > 1] = 1
H_test[H_test < 0] = 0


def get_model1():

    # LiDAR

    input_layer1 = Input((11,11,2))
    out1 = Conv2D(16, (3,3), activation = 'tanh', padding = 'same')(input_layer1)
    out1 = MaxPooling2D((2,2))(out1)
    out1 = BatchNormalization()(out1)
    out1 = Conv2D(32, (3,3), activation = 'tanh', padding = 'same')(out1)
    out1 = MaxPooling2D((2,2))(out1)
    out1 = BatchNormalization()(out1)
    out1 = Conv2D(64, (3,3), activation = 'tanh', padding = 'same')(out1)
    out1 = MaxPooling2D((2,2))(out1)
    out1 = BatchNormalization()(out1)
    out1 = Flatten()(out1)
    out1 = Dense(20, activation='tanh')(out1)
    out1 = BatchNormalization()(out1)

    # HSI

    input_layer2 = Input(64)
    out2 = Dense(32, activation='tanh')(input_layer2)
    out2 = BatchNormalization()(out2)
    out2 = Dense(20, activation='tanh')(out2)
    out2 = BatchNormalization()(out2)

    # Merge

    concat = concatenate([out1, out2])

    # Classifier
    out = Dense(16, activation= 'tanh')(concat)
    out = BatchNormalization()(out)
    out = Dense(11, activation= 'softmax')(out)

    model = Model(inputs= [input_layer1, input_layer2], outputs= out)

    model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy', metrics=['categorical_accuracy'])

    model.summary()

    return model



model = get_model1()
model.fit([L_train, H_train], y_train, epochs=40, batch_size=128, verbose= 1)
model.save(os.getcwd()+ "/fusion6.h5")
model = load_model(os.getcwd()+"/fusion6.h5")

y_pred = model.predict([L_test, H_test])
y_pred = np.argmax(y_pred, axis= -1)
y_test = np.argmax(y_test, axis= -1)

correct = len(y_pred) - np.count_nonzero(y_pred - y_test)
acc = correct/ len(y_pred)
acc = np.round(acc, 4) * 100
   
print("Accuracy: ", acc)

class_names = ['Trees', 'Mostly grass', 'Mixed ground', 'Dirt and sand', 'road', 'water', 'building shadow', 'building', 'sidewalk', 'yellow curb', 'cloth panels']

cm = confusion_matrix(y_test, y_pred, normalize= 'true')
cm = np.round(cm, 3)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot()
plt.show()
