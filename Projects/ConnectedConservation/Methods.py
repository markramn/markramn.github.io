# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:32:58 2022

@author: nicholas.markram
"""

import cv2
import numpy as np
from tensorflow import keras
import os
import random
import bm3d
from skimage import exposure
from PIL import Image
import rasterio
from rasterio import adjust_band

"""
The process class has useful image preprocesing functions
"""
class process:
    
    Image.MAX_IMAGE_PIXELS = None  # disables the warning
    
    def split(img, window_size, margin):
        sh = list(img.shape)
        sh[0], sh[1] = int(sh[0] + margin * 2), int(sh[1] + margin * 2)
        
        img_ = 255 * np.zeros(shape=sh, dtype=np.uint8)
        img_[margin:-margin, margin:-margin] = img

        stride = window_size
        step = window_size + 1 * margin

        nrows, ncols = img.shape[0] // window_size, img.shape[1] // window_size
        image_dataset = []
        location_dataset = []
        
        for i in range(nrows):
            for j in range(ncols):
                x = j*stride
                x_2 = j*stride + step
                y = i*stride
                y_2 = i*stride + step
        
                cropped = img_[y:y_2, x:x_2]
                image_dataset.append(cropped)
                location_dataset.append((x, y))

        return image_dataset, location_dataset
    
    
    def patch(img, patch_size, overlap):
        image = img
    
        SIZE_X = (image.shape[1] // patch_size) * patch_size
        SIZE_Y = (image.shape[0] // patch_size) * patch_size
    
        image = image[0:SIZE_Y, 0:SIZE_X, :]
    
        ncols = int(image.shape[0] / patch_size)
        nrows = int(image.shape[1] / patch_size)
    
        image_dataset = []
        location_dataset = []
    
        for i in range(ncols):
            for j in range(nrows):
   
                y_1 = int(patch_size*i)
                y_2 = int(patch_size*(i+1))
                x_1 = int(patch_size*j)
                x_2 = int(patch_size*(j+1))
    
                pt = image[y_1:y_2, x_1:x_2, :]
    
                row = int(y_1 / patch_size)
                col = int(x_1 / patch_size)
    
                image_dataset.append(pt)
                location_dataset.append((y_1, y_2, x_1, x_2, row, col))
    
        return image_dataset, location_dataset
    

"""
This class provides a number of functions geared 
towards manipulation of images and/or image patches
"""
class manipulation:
    
    def centerImage(image):
        canvas = 255 * np.ones((18, 18, 3), np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
        
        x_1 = int((canvas.shape[0] - image.shape[0]) / 2)
        y_1 = int((canvas.shape[1] - image.shape[1]) / 2)
    
        canvas[x_1:x_1 + image.shape[0],
               y_1:y_1 + image.shape[1]] = image
    
        return canvas
    
    
    def centerImageColor(image):
        canvas = 255 * np.ones((18, 18, 3), np.uint8)
        
        x_1 = int((canvas.shape[0] - image.shape[0]) / 2)
        y_1 = int((canvas.shape[1] - image.shape[1]) / 2)
    
        canvas[x_1:x_1 + image.shape[0],
               y_1:y_1 + image.shape[1], :] = image
    
        return canvas
        
    
    def denoise(image):
        denoised = bm3d.bm3d(image, sigma_psd=0.05, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        denoised = (denoised).astype('uint8')
        
        return denoised
    
    
    def equalize(image):
        equalized = exposure.equalize_adapthist(image)
        equalized = (equalized * 256).astype('uint8')
        
        return equalized
    
    
    def hist_equalize(image):
        img = exposure.equalize_hist(image)
        
        return img
    
    
    def stretchContrast(image):
        p2, p98 = np.percentile(image, (92, 98))
        img = exposure.rescale_intensity(image, in_range=(p2, p98))
        #img = (img * 256).astype('uint8')
        
        return img
    
    
    def UnetSegmentation(image):
        model = keras.models.load_model('UnetSegmentation')
        image = model.predict(image)
        image = (image * 256).astype('uint8')
        
        return image
    
    
    def Canny(image):
        image = cv2.Canny(image, 0, 300)
        image = (image * 256).astype('uint8')
        
        return image
    
    
    def closing(image):
        kernel = np.ones((2, 2), np.uint8)
        erosion = cv2.erode(image, kernel)
        closed = cv2.dilate(erosion, kernel)
        
        return closed
    
    
    def FindClusters(img, k):
        img2 = img.reshape(-1, 3)
        img2 = np.float32(img2)
    
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(
            img2, k, None, criteria, 10, flags)
    
        center = np.uint8(centers)
        res = center[labels.flatten()]
        im = res.reshape(img.shape)
    
        return im
    
    
    def detectBlobs(image):
        #image = np.float32(image)
    
        params = cv2.SimpleBlobDetector_Params()
    
        # Define thresholds
        # Can define thresholdStep. See documentation.
        params.minThreshold = 0
        params.maxThreshold = 255
    
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2 #2px = 1m
        params.maxArea = 500 #30px = 15m
    
        # Filter by Color (black=0)
        params.filterByColor = False
        params.blobColor = 0
    
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0
        params.maxCircularity = 1
    
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0
        params.maxConvexity = 1
    
        # Filter by InertiaRatio
        params.filterByInertia = True
        params.minInertiaRatio = 0
        params.maxInertiaRatio = 1
    
        # Distance Between Blobs
        params.minDistBetweenBlobs = 0
    
        # Setup the detector with parameters
        detector = cv2.SimpleBlobDetector_create(params)
    
        # Detect blobs
        keypoints = detector.detect(image)
    
        #print("Number of blobs detected are : ", len(keypoints))
    
        # Draw blobs
        img_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
        return img_with_blobs, keypoints
    

"""

"""
class Import:

    
    """
    Import by window is useful for importing very large images
    """
    def import_by_window(file):
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1

        def window_from_extent(xmin, xmax, ymin, ymax, aff):
            col_start, row_start = ~aff * (xmin, ymax)
            col_stop, row_stop = ~aff * (xmax, ymin)
            return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

        with rasterio.open(file) as src:
            aff = src.transform
            meta = src.meta.copy()
            window = window_from_extent(xmin, xmax, ymin, ymax, aff)

            img = src.read(1, window=window)

            meta.pop('transform', None)

        return img
    
    
    def import_raster(file):
    
        img = rasterio.open(file)

        imgdata = np.array([adjust_band(img.read(i)) for i in (3, 2, 1)])
        #imgdata = np.array([adjust_band(img.read(i)) for i in (1, 2, 3)])[:, 60:160, 10100:10200]

        
        return imgdata
    
    
    def get_coords(px, py, raster_dir):

        raster = rasterio.open(raster_dir)
        coords = raster.transform * (px, py)

        return coords
    
    
    """
    The stitch function was originally created to loop through each patched-square 
    of an image and manually insert known training data such as elephant samples into
    known locations.
    """
    def stitch(image, dimming_factor, patch_number, patch_size, samples):

        canvas = []
        print("Stitching in animals samples now")
        training_data = []
    
        if patch_size > 50:
            """
            Import Animal Samples
            """
            animals = samples
            for n in range(0, len(animals)):
                tag = animals[n][0]
                animal = animals[n][1]
    
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
                dimming_factor = 0.25 * (1 + (random.randint(-25, 25) / 100))
                try:
                    for i in range(0, animal.shape[0]):
                        for j in range(0, animal.shape[1]):
                            if np.all(animal[i, j] <= (10, 10, 10), axis=-1):
                                image[i + x_1][j + y_1] = image[i + x_1][j + y_1]
                            else:
                                color = animal[i, j]
                                image[i + x_1][j + y_1] = color * dimming_factor
    
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
                    num = patch_number
                    training_data.append((nullTag, sample_number, start_loc[0] + animal.shape[0], start_loc[1] + animal.shape[1], num))
                    #print("Attempted to place sample out of bounds, passing...")
    
        else:
            animals = samples
            n = random.randint(1, len(animals) - 1)
            tag = animals[n][0]
            animal = animals[n][1]
    
            if tag == 'Elephant' or tag == 'Rhino' or tag == 'RhinoC' or tag == 'ElephantC' or tag == 'Giraffe':
                scale_perc = random.randint(-25, 0)
                width = int(animal.shape[1] * (1 + (scale_perc / 100)))
                height = int(animal.shape[1] * (1 + (scale_perc / 100)))
                dim = (width, height)
                animal = cv2.resize(animal, dim, interpolation=cv2.INTER_AREA)
            else:
                pass
    
            dimming_factor = 0.25 * (1 + (random.randint(-25, 25) / 100))
    
            x_1 = 0
            y_1 = 0
            start_loc = (y_1, x_1)
    
            try:
                # Create masks
                canvas = 255 * np.zeros((32, 32, 3), np.uint8)
    
                for i in range(0, animal.shape[0]):
                    for j in range(0, animal.shape[1]):
                        if np.all(animal[i, j] <= (10, 10, 10), axis=-1):
                            image[i + x_1][j + y_1] = image[i + x_1][j + y_1]
                            canvas[i + x_1][j + y_1] = (0, 0, 0)
                        else:
                            color = animal[i, j]
                            image[i + x_1][j + y_1] = color * dimming_factor
                            canvas[i + x_1][j + y_1] = (255, 255, 255)
    
                sample_number = n
                sample_location = (start_loc[0] + animal.shape[0], start_loc[1] + animal.shape[1])
                a = tag
                b = sample_number
                c = sample_location[0]
                d = sample_location[1]
                training_data.append((a, b, c, d))
                #final = cv2.rectangle(image, start_loc, bottom_right, color = (255, 0, 0), thickness = 2)
    
    
            except IndexError as e:
                sample_number = n
                sample_location = (start_loc[0] + animal.shape[0], start_loc[1] + animal.shape[1])
                nullTag = "Null"
                training_data.append(
                    (nullTag, sample_number, start_loc[0] + animal.shape[0], start_loc[1] + animal.shape[1]))
                print("Attempted to place sample out of bounds, passing...")
    
        return image, training_data, canvas
    
    

"""
get_data loads the animal samples from file into an image array
"""
def get_data(data_dir):
    data = []
    labels = ['Elephant', 'ElephantC', 'Giraffe', "Hyena", 'Hippo', 'Wildebeest', 'HyenaC', 'Rhino', 'RhinoC']  
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format

                #img_arr = img_arr.astype(np.float64)
                # resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([img_arr, class_num])
            except Exception as e:
                print(e)
                
    return np.array(data, dtype=object)



"""
This class handles all functions relating to passing the image for prediction
"""
class predict:
    def predict_paths(image):
        model = keras.models.load_model('model')

        SIZE_Y, SIZE_X = 128, 128

        test_img = cv2.resize(image, (SIZE_Y, SIZE_X))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        test_img = np.expand_dims(test_img, axis=0)
        prediction = model.predict(test_img)

        prediction = prediction.reshape((128, 128))
        prediction = (prediction*256).astype('uint8')

        return prediction

    
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    