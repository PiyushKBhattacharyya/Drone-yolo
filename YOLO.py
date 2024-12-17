import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, LeakyReLU, Flatten, Dense
from tensorflow.keras import Model
from math import pow
from PIL import Image, ImageDraw


class Box:
    def __init__(self, classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = 0
        self.probs = np.zeros((classes, 1))


def SimpleNet(yoloNet):
    model = Sequential()

    # First Layer: Convolution + LeakyReLU + Max Pooling
    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(448, 448, 3)))  # TensorFlow uses NHWC format
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), weights=[yoloNet.layers[1].weights, yoloNet.layers[1].biases]))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add layers dynamically from yoloNet
    for i in range(3, yoloNet.layer_number):
        l = yoloNet.layers[i]
        if l.type == "CONVOLUTIONAL":
            model.add(ZeroPadding2D(padding=(l.size // 2, l.size // 2)))
            model.add(Conv2D(l.n, (l.size, l.size), padding='valid', strides=(1, 1), weights=[l.weights, l.biases]))
            model.add(LeakyReLU(alpha=0.1))
        elif l.type == "MAXPOOL":
            model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        elif l.type == "FLATTEN":
            model.add(Flatten())
        elif l.type == "CONNECTED":
            model.add(Dense(l.output_size, weights=[l.weights, l.biases]))
        elif l.type == "LEAKY":
            model.add(LeakyReLU(alpha=0.1))
        elif l.type == "DROPOUT":
            pass
        else:
            print("Error: Unknown Layer Type", l.type)

    return model


def get_activations(model, layer, X_batch):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[layer].output)
    activations = intermediate_layer_model.predict(X_batch)
    return activations


def convert_yolo_detections(predictions, classes=20, num=2, square=True, side=7, w=1, h=1, threshold=0.2, only_objectness=0):
    boxes = []
    probs = np.zeros((side * side * num, classes))
    for i in range(side * side):
        row = i // side
        col = i % side
        for n in range(num):
            index = i * num + n
            p_index = side * side * classes + i * num + n
            scale = predictions[p_index]
            box_index = side * side * (classes + num) + (i * num + n) * 4

            new_box = Box(classes)
            new_box.x = (predictions[box_index + 0] + col) / side * w
            new_box.y = (predictions[box_index + 1] + row) / side * h
            new_box.h = pow(predictions[box_index + 2], 2) * w
            new_box.w = pow(predictions[box_index + 3], 2) * h

            for j in range(classes):
                class_index = i * classes
                prob = scale * predictions[class_index + j]
                if prob > threshold:
                    new_box.probs[j] = prob
                else:
                    new_box.probs[j] = 0
            if only_objectness:
                new_box.probs[0] = scale

            boxes.append(new_box)
    return boxes


def overlap(x1, w1, x2, w2):
    left = max(x1 - w1 / 2, x2 - w2 / 2)
    right = min(x1 + w1 / 2, x2 + w2 / 2)
    return max(0, right - left)


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    return w * h if w > 0 and h > 0 else 0


def box_union(a, b):
    i = box_intersection(a, b)
    return a.w * a.h + b.w * b.h - i


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


def draw_detections(impath, num, thresh, boxes, classes, labels, save_name):
    img = Image.open(impath)
    drawable = ImageDraw.Draw(img)
    ImageSize = img.size
    for i in range(num):
        # For each box, find the class with maximum prob
        max_class = np.argmax(boxes[i].probs)
        prob = boxes[i].probs[max_class]
        if prob > thresh and labels[max_class] == "person":
            b = boxes[i]

            temp = b.w
            b.w = b.h
            b.h = temp

            left = (b.x - b.w / 2.) * ImageSize[0]
            right = (b.x + b.w / 2.) * ImageSize[0]
            top = (b.y - b.h / 2.) * ImageSize[1]
            bot = (b.y + b.h / 2.) * ImageSize[1]

            left = max(0, left)
            right = min(ImageSize[0] - 1, right)
            top = max(0, top)
            bot = min(ImageSize[1] - 1, bot)

            drawable.rectangle([left, top, right, bot], outline="red")
            img.save("results/" + save_name)
