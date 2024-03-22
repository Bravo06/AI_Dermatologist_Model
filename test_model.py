import tensorflow as tf
import cv2, os
import numpy as np
import matplotlib.pyplot as plt


model = tf.keras.models.load_model("melanoma_cnn.keras")


image_number = 1
print(str(image_number).zfill(7))
while image_number <= 60:
    try:
        img = cv2.imread(f"C:/Users/Adit/Desktop/Minor Project/skin_cancer_dataset/Test/melanoma/ISIC_{str(image_number).zfill(7)}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        x_test = np.array([img])
        x_test_norm = x_test.astype('float32')
        x_test_norm = x_test_norm / 255.0

        print(f"Prediction: {model.predict(x_test_norm)}")

        plt.imshow(img, cmap='gray')
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1

'''

image1 = cv2.imread("C:/Users/Adit/Desktop/Minor Project/test1.jpg")
image2 = cv2.imread("C:/Users/Adit/Desktop/Minor Project/melanoma_cancer_dataset/test/malignant/melanoma_10117.jpg")

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

image1 = cv2.resize(image1, (256, 256))
image2 = cv2.resize(image2, (256, 256))

test_data = np.array([image1, image2])
test_data = test_data.astype('float32') / 255.0

prediction = model.predict(test_data)
print(prediction)

print("\nPredictions:")
print("\nImage1:")
if prediction[0][0] >= 0.5:
    print("Malignant")
else:
    print("Benign")

print("\nImage2:")
if prediction[1][0] >= 0.5:
    print("Malignant")
else:
    print("Benign")

print()


plt.figure(figsize=(10,10))

plt.subplot(121)
plt.title("Image1 - No cancer (benign)")
plt.imshow(image1)

plt.subplot(122)
plt.title("Image2 - Cancerous (malignant)")
plt.imshow(image2)

plt.show()
'''