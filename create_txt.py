import numpy as np
import os
 
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR + '/Data/train'))
DATA_DIR_TEST = os.path.abspath(os.path.join(CURRENT_DIR + '/Data/test'))
TXT_DIR = os.path.abspath(os.path.join(CURRENT_DIR + '/Data'))
 


human_images = [image for image in os.listdir(os.path.abspath(os.path.join(DATA_DIR + '/human')))]
car_images = [image for image in os.listdir(os.path.join(DATA_DIR + "/car"))]
noise_images = [image for image in os.listdir(os.path.join(DATA_DIR + "/noise"))]
 
human_images_test = [image for image in os.listdir(os.path.join(DATA_DIR_TEST + "/human"))]
car_images_test = [image for image in os.listdir(os.path.join(DATA_DIR_TEST + "/car"))]
noise_images_test = [image for image in os.listdir(os.path.join(DATA_DIR_TEST + "/noise"))]
 

human_train = human_images[:int(len(human_images))]
human_test = human_images_test[:int(len(human_images_test))]

car_train = car_images[:int(len(car_images))]
car_test = car_images_test[:int(len(car_images_test))]

noise_train = noise_images[:int(len(noise_images))]
noise_test = noise_images_test[:int(len(noise_images_test))]
 
 
with open('{}/train.txt'.format(TXT_DIR), 'w') as f:
    for image in human_train:
        f.write('Data/train/human/{} 0\n'.format(image))
    for image in car_train:
        f.write('Data/train/car/{} 1\n'.format(image))
    for image in noise_train:
        f.write('Data/train/noise/{} 2\n'.format(image))
 
with open('{}/text.txt'.format(TXT_DIR), 'w') as f:
    for image in human_test:
        f.write('Data/test/human/{} 0\n'.format(image))
    for image in car_test:
        f.write('Data/test/car/{} 1\n'.format(image))
    for image in noise_test:
        f.write('Data/test/noise/{} 2\n'.format(image))
