#!usr/bin/env python3

import numpy as np
import cv2 as cv
from utils import IMGSorter
import os
import OpenEXR
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

'''Method to normalize depth mats'''
'''------------------------------------------------------------------------------------------------------'''
def convert_dmap(kf_depth):
    kf_depth = kf_depth[:,:,0]
    dmap_min_val,dmap_max_val,dmap_min_indx,dmap_max_indx=cv.minMaxLoc(kf_depth)
    om = np.floor(np.log10(dmap_max_val))
    dmap_norm = np.ones(kf_depth.shape[0:2]).astype('float32') * -1
    for i in range(kf_depth.shape[0]):
        for j in range(kf_depth.shape[1]):
            if abs(np.floor(np.log10(kf_depth[i][j]))-om) < 2:
                continue
            else:
                dmap_norm[i][j] = kf_depth[i][j]
    return dmap_norm
 

if __name__ == '__main__':

    # ranges of the camera
    FL = 15
    ranges = [15] 

    for rn in ranges:
        # directory
        dmap_folder = f'database/y-axis_FL{FL}_r{rn}/dmaps/'
        dmap_paths_sorted = IMGSorter(dmap_folder).get_sorted_image_paths(format='.exr')

        for path in dmap_paths_sorted:
            # get dmap
            get_file_id = os.path.splitext(os.path.basename(path))[0]
            print(path)

            dmap = cv.imread(path, cv.IMREAD_UNCHANGED)
            print(dmap)
            # make dirs to save normalized dmap
            save_path = f'database/y-axis_FL{FL}_r{rn}/dmaps_norm/'
            os.makedirs(save_path, exist_ok=True)
            # normalize dmap
            dmap_norm = convert_dmap(dmap)
            # write dmap to created directory
            cv.imwrite('{}/{}.exr'.format(save_path, get_file_id),  dmap_norm)

