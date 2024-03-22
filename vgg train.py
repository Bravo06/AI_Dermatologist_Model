import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, Sequential


def load_data():
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")

    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# preprocessing
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

base_model = VGG16(weights='imagenet', include_top=False)
base_model.trainable = False


model = Sequential([
    base_model, 
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)



# evaluate model
print(f"Accuracy: {model.evaluate(x_test, y_test)[1] * 100}%")


model.save("vgg16_melanoma_cnn.keras")