# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 07:08:29 2022

@author: nicholas.markram
"""

import cmath
import math
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from skimage import img_as_ubyte, measure, io
from skimage.color import label2rgb, rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import area_opening, medial_axis
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import area_opening, medial_axis
from skimage.exposure import histogram
from skimage.filters import threshold_otsu


from Insertion import *
from Preprocess import *
from Patch import *
from Display import display, showTraining
#from rasterio.plot import show, adjust_band, reshape_as_image, reshape_as_raster
import cv2
from tensorflow import keras
from CCF_Model import make_prediction as make_pred

start_time = time.time()

segment = "R2C2"
sample1 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/AnimalSample1.jpg"

sample2 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/MicrosoftTeams-image (2).png"
sample3 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/MicrosoftTeams-image (3).png"
sample4 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/Animals_L3.jpg"
sample6 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/sample6.jpg"
sample7 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/sample7.jpg"
sample8 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/S1_WhiteFigures.tif"
sample9 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/S2_PossibleElephants.tif"
sample10 = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/Samples/S3_Rhinos.tif"
im_dir = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/IMG/5901886101/IMG_PHR1A_PMS-N_001/IMG_PHR1A_PMS-N_202108280829181_ORT_5901886101_" + segment + ".JP2"
animal_sample = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF/ImagesPadded32x32/Elephant/elephant_-1.jpg"
meta_dir = 'IMG/5901886101/IMG_PHR1A_PMS-N_001/DIM_PHR1A_PMS-N_202108280829181_ORT_5901886101.XML'

# model = train_model()
# model.save('model')
model = keras.models.load_model('model')
# testModel(model)

"""
Read in sample image
"""
image = cv2.imread(sample9) #BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sample = image.copy()

plt.imshow(sample)
plt.show()

# image = ~image
# image = equalize(image)

patch_size = 32
overlap = 0.75
window_size = int(patch_size * overlap)
margin = patch_size - window_size
results = split(img=image, window_size=window_size, margin=margin)

image_dataset = results[0]
location_dataset = results[1]
count = 0

for i in range(0, len(image_dataset)):

    image = image_dataset[i]
    print(image)
    location = location_dataset[i]

    #plt.imshow(image)
    #plt.show()

    """
    image = closing(image)

    original = image.copy()

    # Denoising Image
    image = denoise(image)

    # Color Segmentation by k-means
    k = 8
    image = FindClusters(img=image, k=k)

    # Image thresholding accounting for initial scale
    scale = 3
    factor = scale
    threshold = (threshold_otsu(image) / factor) + 5
    print(threshold)

    ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    """

    """
    cv2
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    original = image.copy()

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)

    gray = cv2.erode(image, kernel=kernel, iterations=4)

    gray = cv2.dilate(gray, kernel=kernel, iterations=2)

    # Image thresholding accounting for initial scale

    gray = cv2.GaussianBlur(gray, (1, 1), 0)

    gray = cv2.filter2D(src=gray, ddepth=-3, kernel=kernel)

    _, gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    results = detectBlobs(gray)
    patch = results[0]
    keypoints = results[1]
    print(keypoints)

    img_with_blobs = cv2.drawKeypoints(original, keypoints, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Pass patch for prediction if keypoint present
    if len(keypoints) > 0:
        print('Object has been detected!')
        for j in range(0, len(keypoints)):

            #plt.imshow(img_with_blobs)
            #plt.show()


            def masked_image(image, mask):
                r = image[:, :, 0] * mask
                g = image[:, :, 1] * mask
                b = image[:, :, 2] * mask
                return np.dstack([r, g, b])


            x = round(keypoints[j].pt[0])
            y = round(keypoints[j].pt[1])
            kp_x = round((location[0]) + int(x)) - int(patch_size * (1 - overlap))
            kp_y = round((location[1]) + int(y)) - int(patch_size * (1 - overlap))

            src = image_dataset[i]
            # convert img to grayscale
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            kernel = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], np.uint8)

            kernelE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            gray = cv2.erode(gray, kernel=kernelE, iterations=6)
            #plt.imshow(gray)
            #plt.show()

            gray = cv2.dilate(gray, kernel=kernel, iterations=3)
            #plt.imshow(gray)
            #plt.show()

            # Image thresholding accounting for initial scale

            gray = cv2.GaussianBlur(gray, (1, 1), 0)

            gray = cv2.filter2D(src=gray, ddepth=-3, kernel=kernel)
            #plt.imshow(gray)
            #plt.show()

            _, gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            #plt.imshow(gray)
            #plt.title('For classification')
            #plt.show()

            centroid = gray.copy()
            centroid = centroid[y - 6: y + 6, x - 6:x + 6]
            #plt.imshow(centroid)
            #plt.show()

            cent = src.copy()
            cent = cent[y - 6: y + 6, x - 6:x + 6, :]

            if centroid.shape[0] < 7 or centroid.shape[1] < 7:
                pass
            else:
                mask = centerImage(centroid)

                cent = centerImageColor(cent)

                #plt.imshow(mask, cmap='gray')
                #plt.title('mask')
                #plt.show()

                #plt.imshow(cent)
                #plt.show()

                for m in range(0, mask.shape[0]):
                    for n in range(0, mask.shape[1]):
                        if np.any(mask[m, n] == (179, 179, 179), axis=-1):
                            cent[m, n] = cent[m, n]
                        elif np.any(mask[m, n] == (255, 255, 255), axis=-1):
                            cent[m, n] = (255, 255, 255)

                for_pred = cent

                #plt.imshow(for_pred)
                #plt.title('for pred')
                #plt.show()

                results = make_pred(for_pred, model=model)
                classification = results[0]
                softmax = results[1]

                # final = marked.copy()
                final = cv2.rectangle(sample, (kp_x - 5, kp_y - 5), (kp_x + 5, kp_y + 5), color=(255, 0, 0),
                                      thickness=1)
                final = cv2.putText(final, classification, (kp_x - 5, kp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 0, 0), 1)
                print(classification, softmax)

                #plt.imshow(final)
                #plt.show()

plt.imshow(final)
plt.show()