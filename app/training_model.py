import cv2 as cv
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit


data = []

labels = []

base_folder_images = "app/db_letters"

images = paths.list_images(base_folder_images)

for file in images:
    label = file.split(os.path.sep)[-2]
    image = cv.imread(file)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # standardize the image in 20x20
    image = resize_to_fit(image, 20, 20)

    # add a dimension so Keras can read the image
    image = np.expand_dims(image, axis=2)

    # add to data lists and labels
    labels.append(label)
    data.append(image)

data = np.array(data, dtype="float") / 255
labels = np.array(labels)

# separation into training date (75%) and test date (25%)
(X_train, X_test, Y_train, Y_test) = train_test_split(
    data, labels, test_size=0.25, random_state=0
)

# Convert with one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# save label_binarizer to a file with pickle
with open("app/labels_model.dat", "wb") as pickle_file:
    pickle.dump(lb, pickle_file)

# create and train artificial intelligence
modelo = Sequential()

# create the neural network layers
modelo.add(
    Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu")
)
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# create a second layer
modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# add one more layer
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))
# output layer
modelo.add(Dense(26, activation="softmax"))
# compile all layers
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# train IA
modelo.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    batch_size=26,
    epochs=10,
    verbose=1,
)

# save the model to a file
modelo.save("app/trained_model.hdf5")
