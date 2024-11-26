import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from google.colab import drive
from PIL import Image, ImageFilter
from numpy import asarray
import os

# path to weight file, image file
weights_path_yolo = '/weight/best_one_number_region.pt'
weights_path_keras = '/weight/letters_model2.h5'
image_path = '/img_test/test1.png'
emnist_labels = [
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57
]
# load yolo model
model_yolo = YOLO(weights_path_yolo)

# read image
image = cv2.imread(image_path)
results = model_yolo(image, conf=0.2)
boxes = results[0].boxes.xyxy  # get bounding boxes
sorted_boxes = sorted(boxes, key=lambda box: box[0].item())  # sort boxes from left to right

# draw bounding boxes
def draw_bounding_boxes(image, boxes):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4].tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Vẽ khung đỏ
    plt.imshow(image)
    plt.axis('off')
    plt.show()

draw_bounding_boxes(image, sorted_boxes)

# preprocess
def square_image(img, size_x, size_y, interpolation=cv2.INTER_AREA):
    height, width = img.shape
    add_x = (max(height, width) - width) // 2
    add_y = (max(height, width) - height) // 2
    padded_img = cv2.copyMakeBorder(img, add_y, add_y, add_x, add_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(padded_img, (size_x, size_y), interpolation)
def emnist_predict_img(model, img):
    img_arr = img
    img_arr[0] = np.rot90(img_arr[0], k=1, axes=(1, 0))
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))
    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(emnist_labels[result[0]])

def get_letters(img, model):
    letters = ''
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sizeH, sizeW = gray.shape
    ret,thresh1 = cv2.threshold(gray ,120,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)
    for box in sorted_boxes:
        x1, y1, x2, y2 = map(int, box[:4].tolist())
        roi = gray[y1:y2, x1:x2]
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = square_image(thresh, 28, 28, cv2.INTER_AREA)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,28,28,1)
        ypred = emnist_predict_img(model, thresh)
        print(ypred)
        letters += ypred
    return letters, image
def create_emnist_model():
    model = Sequential([
        Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        MaxPooling2D((2, 2)),
        Convolution2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Convolution2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(emnist_labels), activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# load model emnist for handwritten digits
emnist_model = create_emnist_model()
emnist_model.load_weights(weights_path_keras)

# predict
predicted_letters,image_result = get_letters(image_path, emnist_model)
print("Predicted Letters:", predicted_letters)
