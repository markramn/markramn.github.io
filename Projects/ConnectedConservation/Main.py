# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 06:54:25 2022

@author: nicholas.markram
"""

"""
Load dependencies
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from skimage.filters import threshold_otsu
from Methods import process, manipulation
from Model import predict, train_model
from InsertAnimals import insertIntoSatellite


"""
Define directories
"""
sample_dir = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF_2022/SpeciesDetection/Images/IMG_PNEO4_2022030208_2.tif"
ElephantsAtWater = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF_2022/SpeciesDetection/Images/Madikwe dam- sighting 2.tif"
Herd = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/CCF_2022/SpeciesDetection/Images/extract_herd_large.tif"
elephants = "C:/Users/nicholas.markram/OneDrive - NTT/Desktop/elephants.tif"


"""
Load in satellite image
"""
original = cv2.imread(elephants)  # BGR
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
image = original.copy()

#image = results[0]
sample = image.copy()
results_dataset = []


"""
Stitch in samples if needed
"""
#results = insertIntoSatellite(image=image, num_animals=750)

"""
Load in prediction model
"""
model = keras.models.load_model('CCF_Model')


"""
Split satellite image
"""
patch_size = 300  # Patches of 128x128 pixels
overlap = 0.9 #0.75  # 25% overlap on images
print('Splitting images into patches...')
window_size = int(patch_size * overlap)
margin = patch_size - window_size

results = process.split(img=image, window_size=window_size, margin=margin)

image_dataset = results[0]
location_dataset = results[1]


"""
Detection pipeline
"""
for i in range(0, len(image_dataset)):
    image = image_dataset[i]
    #image = manipulation.FindClusters(image, 5)
    original_patch = image.copy()
    location = location_dataset[i]

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    
    

    # Method One: Image Opening
    """
    image = manipulation.denoise(image) # Denoising Image
    
    # Image thresholding accounting for initial scale
    factor = (3 - 1)
    threshold = (threshold_otsu(image) / factor)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #plt.hist(image.flat, bins=100, range=(0, 255))
    
    _, gray = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.erode(gray, kernel=kernel, iterations=4)
    gray = cv2.dilate(gray, kernel=kernel, iterations=2)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    gray = cv2.filter2D(src=gray, ddepth=-3, kernel=kernel)
    _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    processed = gray.copy()


    """
    # Method Two: Clustering
    image = manipulation.denoise(image) # Denoising Image
    plt.imshow(image)
    plt.show()
    
    image = manipulation.FindClusters(image, 3)
    plt.imshow(image)
    plt.show()
    
    # Image thresholding accounting for initial scale
    #factor = (scale - 1)
    #threshold = (threshold_otsu(image) / factor)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.hist(image.flat, bins=100, range=(0, 255))
    plt.show()
    
    threshold = threshold_otsu(gray)
    print(threshold)
    
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    processed = thresh.copy()
    """

    
    """
    Everything up until this point isolates potential features for
    binary large object detection. If an object is detected, 
    pass the object for detection
    """
    results = manipulation.detectBlobs(processed)
    patch = results[0]
    keypoints = results[1]

    img_with_blobs = cv2.drawKeypoints(original_patch, keypoints, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    

    # Pass patch for prediction if keypoint present
    if len(keypoints) > 0:
        print('Object has been detected!')

        for j in range(0, len(keypoints)):

            x = round(keypoints[j].pt[0])
            y = round(keypoints[j].pt[1])
            kp_x = round((location[0]) + int(x)) - \
                int(patch_size * (1 - overlap))
            kp_y = round((location[1]) + int(y)) - \
                int(patch_size * (1 - overlap))

            src = image_dataset[i]
            #plt.imshow(src)

            centroid = processed.copy()
            centroid = centroid[y - 6: y + 6, x - 6:x + 6]

            cent = src.copy()
            cent = cent[y - 6: y + 6, x - 6:x + 6, :]

            if centroid.shape[0] < 7 or centroid.shape[1] < 7:
                pass
            else:
                mask = manipulation.centerImage(centroid)

                cent = manipulation.centerImageColor(cent)

                for m in range(0, mask.shape[0]):
                    for n in range(0, mask.shape[1]):
                        if np.any(mask[m, n] == (179, 179, 179), axis=-1):
                            cent[m, n] = cent[m, n]
                        elif np.any(mask[m, n] == (255, 255, 255), axis=-1):
                            cent[m, n] = (255, 255, 255)

                for_pred = cent
                #plt.imshow(for_pred)

                results = predict(for_pred, model=model)
                classification = results[0]
                softmax = results[1]
                
                if classification == 'Hyena':
                    classification = 'Small Animal'

                coords = (kp_x, kp_y)
                results_dataset.append((classification, softmax, coords))

                output = cv2.rectangle(
                    sample, (kp_x - 5, kp_y - 5), (kp_x + 5, kp_y + 5), color=(255, 0, 0), thickness=1)
                #output = cv2.putText(output, classification + " - " + str(softmax), (kp_x - 5, kp_y - 5),
                                     #cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                #print(classification, softmax)


"""
Visualize output after species detection
"""
plt.imshow(output)


"""
Save detection information
"""