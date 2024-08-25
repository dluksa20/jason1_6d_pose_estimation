import numpy as np
import cv2 as cv
import os
from utils import IMGSorter
from utils import data2sort
from utils import helper
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from utils import NaturalFeature


# ranges of the camera
ranges = [20,30,40] 
FL=15
# database image folder
folder_id = ''
# save folder for keypoints, decriptors and 3d points
folder_json = 'json_data'

# detectors amd descriptors main
detector1 = cv.BRISK_create()
descriptor1 = cv.xfeatures2d.FREAK_create()

# initialize main loop for features
for rn in ranges:
    # get folder paths
    img_folder = 'database{}/y-axis_FL{}_r{}/renders'.format(folder_id, FL, rn)
    k_folder = 'database{}/y-axis_FL{}_r{}/k'.format(folder_id, FL, rn)
    rt_folder = 'database{}/y-axis_FL{}_r{}/rt'.format(folder_id,FL, rn)
    dmap_folder = 'database{}/y-axis_FL{}_r{}/dmaps_norm'.format(folder_id,FL, rn)


    # sort files in the folders in ascending order
    img_paths_sorted = IMGSorter(img_folder).get_sorted_image_paths(format='.png')
    k_paths_sorted = IMGSorter(k_folder).get_sorted_image_paths(format='.txt')
    rt_paths_sorted = IMGSorter(rt_folder).get_sorted_image_paths(format='.txt')
    dmap_paths_sorted = IMGSorter(dmap_folder).get_sorted_image_paths(format='.exr')

    # get k matrix from txt
    get_file_id = os.path.splitext(os.path.basename(k_paths_sorted[0]))[0]
    kMat = data2sort('{}'.format(get_file_id), focal_length=FL).get_k_matrix(k_paths_sorted[0])# 0 - cause k -matrix at all instances is same


    # init feature detectors and descriptors
    feature1 = NaturalFeature(detector=detector1, descriptor=descriptor1, border=0)

    # init sub loop
    for i in range(len(img_paths_sorted)): 

        get_file_id = os.path.splitext(os.path.basename(img_paths_sorted[i]))[0]

        # Get corresponding rotation matrices
        rtMat = data2sort('{}'.format(get_file_id),focal_length=FL).get_k_matrix(rt_paths_sorted[i])# step - cause rt -matrix values changes at every step
        rtMat = np.array(rtMat)

        # get corresponding depth map
        dmap = cv.imread(dmap_paths_sorted[i], cv.IMREAD_ANYDEPTH)
        # read the image
        image = cv.imread(img_paths_sorted[i], 0)
        image = cv.equalizeHist(image)


        feature1.Generate_KpsDes(image)
        feature1.Get_KpEhd_3dPoints(dmap, kMat, rtMat)

            # save keypoints to json
        data2sort(get_file_id, focal_length=FL).keyPoint_2_json(feature1.vis_kps, folder='kps_json_brisk', folder_id=folder_json)
        # save descriptors to json 
        data2sort(get_file_id, focal_length=FL).descriptor_2_json(feature1.vis_des, folder='des_json_brisk', folder_id=folder_json)
        # save 3d points to json
        data2sort(get_file_id, focal_length=FL).pts3D_2_json(feature1.vis_3d, 'pts_3d_brisk', folder_id=folder_json)

        print(get_file_id)
