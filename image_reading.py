import os
import numpy as np
import warnings
#import SimpleITK as sitk
import cv2
from scipy import misc
from scipy import ndimage


def load_image_from_folder(folder_path, new_size, HE=False, Truc=False, Aug=False):
    """loads images in the folder_path and returns a ndarray and threshold the label image"""

    image_list = []
    label_list = []
    #counter = 0
    for image_name in os.listdir(folder_path):
        image_original = np.load(folder_path + image_name)
        image_original = image_original['a']
        #counter = counter + 1
        #print image_name, counter
        image_ph = image_original[:, 0:len(image_original)]
        image_sc = image_original[:,len(image_original):len(image_original)*2]
        label = image_original[:,len(image_original)*2:len(image_original)*3]

        image_all = np.concatenate((image_ph, image_sc), axis=1)
        image_list.append(image_all)
        label_list.append(label)

    image_array = np.asarray(image_list)
    label_array = np.asarray(label_list)

    return image_array, label_array