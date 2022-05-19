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


# dados
data = []
# rotúlos
labels = []
# imagem base
base_folder_images = "app/base_letters"

images = paths.list_images(base_folder_images)

for file in images:
    label = file.split(os.path.sep)[-2]
    image = cv.imread(file)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # padronizar a imagem em 20x20
    image = resize_to_fit(image, 20, 20)

    # adicionar uma dimensão para o Keras poder ler a imagem
    image = np.expand_dims(image, axis=2)

    # adicionar às listas de dados e rótulos
    labels.append(label)
    data.append(image)

data = np.array(data, dtype="float") / 255
labels = np.array(labels)

# separação em data de treino (75%) e data de teste (25%)
(X_train, X_test, Y_train, Y_test) = train_test_split(
    data, labels, test_size=0.25, random_state=0
)

# Converter com one-hot encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salvar o labelbinarizer em um file com o pickle
with open("app/labels_model.dat", "wb") as pickle_file:
    pickle.dump(lb, pickle_file)

# criar e treinar a inteligência artificial
modelo = Sequential()

# criar as camadas da rede neural
modelo.add(
    Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu")
)
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# criar a 2ª camada
modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# mais uma camada
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))
# camada de saída
modelo.add(Dense(26, activation="softmax"))
# compilar todas as camadas
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# treinar a inteligência artificial
modelo.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    batch_size=26,
    epochs=10,
    verbose=1,
)

# salvar o modelo em um arquivo
modelo.save("app/trained_model.hdf5")
