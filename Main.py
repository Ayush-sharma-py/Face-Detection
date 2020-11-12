import os
import cv2
import tkinter
import matplotlib.image as img
import numpy
from tensorflow import keras
from scipy import misc
from PIL import Image,ImageDraw

main_path = "C:\\Users\\Ayush Sharma\\Desktop\\Programs\\Face-Detection"
training_directory = main_path + "\\training"
resized_training_directory = main_path + "\\resized"
names = []
labels = []

try:
    os.mkdir(training_directory)
    os.mkdir(resized_training_directory)
except:
    pass

for name in os.listdir(training_directory):
    names.append(name)

def add_training_class(name:str):
    try:
        os.mkdir(training_directory + f"\\{name}")
        names.append(name)
        print("Done")
    except:
        print("Already Made")

training_class_directories = os.listdir(training_directory)

for i in range(0,len(training_class_directories)):
    for k in os.listdir(training_directory + "\\" + training_class_directories[i]):
        ig = Image.open(training_directory + "\\" + training_class_directories[i] + "\\" + k)
        ig = ig.resize((100,100))
        ig.save(resized_training_directory + "\\" + k)
        labels.append(i)

processed_images = []
for i in os.listdir(resized_training_directory):
    n = img.imread(resized_training_directory + "\\" + i)
    processed_images.append(n)

training_set = numpy.array(processed_images)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(training_set.shape[1], training_set.shape[2], training_set.shape[3])),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(len(names))
])

for i in range(0,len(labels)):
    labels[i] = float(labels[i])

print(labels)





