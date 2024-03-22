import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import os

def load_data():
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")

    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# preprocessing
x_train = preprocess_input(x_train)


base_model = ResNet50(weights='imagenet', include_top = False)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15)


# evaluate model
print(f"Accuracy: {model.evaluate(x_test, y_test)[1] * 100}%")


model.save("resnet_melanoma_cnn.keras")