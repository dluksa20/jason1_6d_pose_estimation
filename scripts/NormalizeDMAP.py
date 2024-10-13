#!./venv/bin/python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import cv2 as cv
from utils import IMGSorter
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

'''Method to normalize depth mats'''
'''------------------------------------------------------------------------------------------------------
Run the script in the terminal:
    ./scripts/normalizeDMAP.py database 15 10 20 30 40 50
    database - folder/directory where depth maps residing
    agrv1(15) - camera focal length
    argv2(10-50) - select space separated ranges
'''


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
    FL = sys.argv[2]
    ranges = sys.argv[3:]

    for rn in ranges:
        # directory
        dmap_folder = f'{sys.argv[1]}/y-axis_FL{FL}_r{rn}/dmaps/'
        dmap_paths_sorted = IMGSorter(dmap_folder).get_sorted_image_paths(format='.exr')

        for path in dmap_paths_sorted:
            start_timer = time.time()
            # get dmap
            get_file_id = os.path.splitext(os.path.basename(path))[0]
            dmap = cv.imread(path, cv.IMREAD_UNCHANGED)
            # make dirs to save normalized dmap
            save_path = f'{sys.argv[1]}/y-axis_FL{FL}_r{rn}/dmaps_norm/'
            os.makedirs(save_path, exist_ok=True)
            # normalize dmap
            dmap_norm = convert_dmap(dmap)
            # write dmap to created directory
            cv.imwrite('{}/{}.exr'.format(save_path, get_file_id),  dmap_norm)
            end_time = time.time()

            print(f'{'>'*80}\nSucces!\nNormalized depth map saved to: {save_path}{get_file_id}.exr\nTime elapsed: {end_time-start_timer:.2f}s')

