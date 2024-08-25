import os
import numpy as np
from utils import IMGSorter


'''Compute araays of test image paths'''
'''------------------------------------------------------------------------------------------------------'''
img_folder = 'database/blender_data/Train_img_2/y-axis_FL35_r50/renders'

# sort files in the folders in ascending order
img_paths_sorted = IMGSorter(img_folder).get_sorted_image_paths(format='_0.png')
all_image_paths = []
for image_path in img_paths_sorted:
    file_name = os.path.basename(image_path)
    all_image_paths.append(image_path)

for i in all_image_paths:
    print(i)
np.save('database/arrays/test1Paths.npy', all_image_paths)