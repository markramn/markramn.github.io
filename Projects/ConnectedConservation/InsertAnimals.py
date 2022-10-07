# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 07:04:08 2022

@author: nicholas.markram
"""

import cv2
import numpy as np
import random
from PIL import Image

def insert(image, dimming_factor, patch_number, patch_size, samples, for_training):
    
    canvas = []
    #print("Stitching in animals samples now")
    training_data = []
    
    if for_training == True:
        animals = samples
        for n in range(0, 500):
            sel = random.randint(0, len(animals)-1)
            #sel = n
            tag = animals[sel][0]
            animal = animals[sel][1]
        
            #dimming_factor = 0.25 * (1 + (random.randint(-25, 25) / 100))
            
            x_1 = int((image.shape[0] - animal.shape[0]) / 2)
            y_1 = int((image.shape[1] - animal.shape[1]) / 2)
            start_loc = (y_1, x_1)
            #print(start_loc)
            canvas = 255 * np.ones((18, 18, 3), np.uint8)
            
            for i in range(0, animal.shape[0]):
                for j in range(0, animal.shape[1]):
                    if np.any(animal[i, j] <= (15, 15, 15), axis=-1):
                        image[i + x_1][j + y_1] = (255, 255, 255)
                        canvas[i + x_1][j + y_1] = (255, 255, 255)
                    else:
                        color = animal[i, j]
                        image[i + x_1][j + y_1] = color
                        canvas[i + x_1][j + y_1] = color
                        
            sample_number = n
            sample_location = (start_loc[0] + animal.shape[0], start_loc[1] + animal.shape[1])
            a = tag
            b = sample_number
            c = sample_location[0]
            d = sample_location[1]
            training_data.append((a, b, c, d))
            
        return image, training_data, canvas

#samples = fetchSamplesFull()    
def insertPlain(samples):
    animals = samples
    for n in range(0, len(animals)):
        
        tag = animals[n][0]
        print(tag)
        animal = animals[n][1]
        canvas = 255 * np.ones((18, 18, 3), np.uint8)
        x_1 = int((canvas.shape[0] - animal.shape[0]) / 2)
        y_1 = int((canvas.shape[1] - animal.shape[1]) / 2)
        start_loc = (y_1, x_1)
        
        for i in range(0, animal.shape[0]):
            for j in range(0, animal.shape[1]):
                if np.any(animal[i, j] <= (15, 15, 15), axis=-1):
                    canvas[i + x_1][j + y_1] = (255, 255, 255)
                else:
                    color = animal[i, j]
                    canvas[i + x_1][j + y_1] = color
                    
        save_string = 'C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF_2022/Samples/TrainingData/' + str(tag) + '/' + str(
                tag) + '-1' + str(n) + '.jpg'
        canvas = Image.fromarray(canvas)
        print('saving')
        canvas.save(save_string)
        
#insertPlain(samples)


def insertIntoSatellite(image, num_animals):
    training_data = []
    animals = fetchSamplesFull()   
    for n in range(0, num_animals):
        sel = random.randint(0, len(animals))
        tag = animals[sel][0]
        animal = animals[sel][1]

        if tag == 'Elephant' or tag == 'Rhino' or tag == 'RhinoC' or tag == 'ElephantC' or tag == 'Giraffe':
            scale_perc = random.randint(-25, 0)
            width = int(animal.shape[1] * (1 + (scale_perc / 100)))
            height = int(animal.shape[1] * (1 + (scale_perc / 100)))
            dim = (width, height)
            animal = cv2.resize(animal, dim, interpolation=cv2.INTER_AREA)
        else:
            pass

        x_1 = random.randint(0, image.shape[0])
        y_1 = random.randint(0, image.shape[1])
        start_loc = (y_1, x_1)
        #dimming_factor = 0.25 * (1 + (random.randint(-25, 25) / 100))
        try:
            for i in range(0, animal.shape[0]):
                for j in range(0, animal.shape[1]):
                    if np.all(animal[i, j] <= (15, 15, 15), axis=-1):
                        image[i + x_1][j + y_1] = image[i + x_1][j + y_1]
                        bottom_right = (j + y_1, i + x_1)
                    else:
                        color = animal[i, j]
                        image[i + x_1][j + y_1] = color
                        bottom_right = (j + y_1, i + x_1)

            sample_number = n
            sample_location = (start_loc[0] + animal.shape[0], start_loc[1] + animal.shape[1])
            a = tag
            b = sample_number
            c = int(x_1 + (animal.shape[0] / 2))
            d = int(y_1 + (animal.shape[1] / 2))
            training_data.append((a, b, c, d))
            image = np.ascontiguousarray(image, dtype=np.uint8)
            #image = np.float32(image)
            #image = (image * 256).astype('uint8')
            #unmarked = image
            #final = cv2.rectangle(image, start_loc, bottom_right, color=(0, 0, 255), thickness=2)


        except IndexError as e:
            sample_number = n
            sample_location = (start_loc[0] + animal.shape[0], start_loc[1] + animal.shape[1])
            nullTag = "Null"
            training_data.append((nullTag, sample_number, start_loc[0] + animal.shape[0], start_loc[1] + animal.shape[1]))
            print("Attempted to place sample out of bounds, passing...")
                
    return image, training_data