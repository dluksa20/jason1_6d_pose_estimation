import cv2
import numpy as np
import utils_extra as miscelaneous
from utils import data2sort, helper
from PosePnP import PosePnP
import time
from utils_extra import getError, rt_transform, quaternion_to_rotation_matrix
import pandas as pd
import os

# ---------------------------------------------------------------------------------------------------------------------#
#                                                       MAIN EXECUTION                                                 #
# ---------------------------------------------------------------------------------------------------------------------#

'''''
paths to the database and test images
kps_path - reference keyframe keypoints in JSON format
path_rt - corresponding keyframe groundtruth
path_test - test images from ZED camera

'''''
kps_path = 'database/json_data/kps_json_brisk/y-axis_FL35_r50'
path_rt = 'database/blender_data/GroundTruth/y-axis_FL35_r50/rt'
# paths_test = miscelaneous.sort_images_in_folder('/Users/dovydasluksa/Documents/Documents/Project_MSc/Test_img/Test_images_3/images_9_28')
paths_test = miscelaneous.sort_images_in_folder('database/Test_rot/img_rot_2.0m_seg/')
# paths_test = paths_test[0:200]

'''''
- paths to sythetic test images from blender
paths_test = miscelaneous.sort_images_in_folder('database/blender_data/Train_img_3/y-axis_FL35_r40/renders')
paths_test = np.load('database/arrays/test1Paths.npy')
'''''


# Create objects for reference and test frame 
'''''
available detectors - ORB,BRISK,SIFT
available descriptors - ORB, BRISK, FREAK, SIFT

Note: To use different detector or descriptor type database have to be generated accordingly

'''''
obj = PosePnP(detector_type='BRISK', descriptor_type='FREAK')
obj1 = PosePnP(detector_type='BRISK', descriptor_type='FREAK')

'''BF- Brute force, BFKnn - BRute force with knnMatch, FLANN - flann based matcher
nndr - nerest neighbour distance ratio BF - set None, FLANN, BFKnn '''
obj.setMatcher('BF', nndr=None)
obj1.setMatcher('BF', nndr=None)

#Camera params 
dist_coeffs = np.array([-0.1744, 0.0273, 0.0007, -0.0003, 0], dtype=np.float32)# ZED camera distortion coefficients
#dist_coeffs = None ## Blender camera

kMat_test = np.array([[1400.41, 0.000000, 1100.29 ], 
            [0.000000, 1400.41, 638.258 ], 
            [0.000000, 0.000000, 1.000000 ]]) # ZED camera
            

# kMat_test = np.array([[1991.111084, 0.000000, 320 ], 
#             [0.000000, 1991.111084, 320 ], 
#             [0.000000, 0.000000, 1.000000 ]]) ## Belnder camera



idx = 0 # set to 0 to initialize initial reference keyframe search
data_type = 'brisk' # keypoint detector of the database, Note: for current databases ORB and BRISK a FREAK descriptor used

'''Keyframe update method:
    - relativeRotation - reference keyframe updated based on relative rotation of the test image with respect to 
      selected keyframe
    - variance - reference keyframe updated by conducting search of nearest keyframes within pre-defined radius
      (refer to posePnP class computeNeighbours method), new keyframe selected with most inlying keypoints'''

mode_update = 'variance' 

# timeout exit loop if reference keyframe not found within defined time 
 

predictions = []

# ---------------------------------------------------------------------------------------------------------------------#
#                                                       Estimation loop                                                #
# ---------------------------------------------------------------------------------------------------------------------#


while True:
    # initial variables for storing data
    relative_rotation_reference = 0
    reset_kf_counter = 0
    variance = 0
    reset_counter = 0

    start_time = time.time()  # Record the current time when entering the for loop
    timeout = 10
    # initialize main loop 
    for path_query in paths_test:

        # idx  = 0, loops searches for initial keyframe, initialized only once, unless further in the script it is reset to 0
        if idx == 0:

            # pre-process test image as needed 
            img_query_ = path_query
            frame_query = cv2.imread(img_query_, 0) 
            frame_query = cv2.equalizeHist(frame_query)
            # frame_query  = cv2.GaussianBlur(frame_query, (7,7,),7)
            # frame_query = cv2.bilateralFilter(frame_query, d=4, sigmaColor=45, sigmaSpace=45)
            ''' compute keypoints and descriptors of test frame''' 
            obj.detectAndCompute(frame_query)

            ''' match with database keypoints, thresh - min inlying keypoints after RANSAC (refer to posePnP class for more details)
            timeout error diplayed if reference keyframe not found 10s, inliers threshold needs to be reduced'''
            id, kps_reference, des_reference = obj.selectKeyframe(kps_path, thresh=40, data_type=data_type)

            # idx = 1, this loop wont be executed, proceeding to the main pose estimation loop
            idx =1       

        # If reference frame found idx = 1 condition met and pose estimation loop initiated 
        if idx != 0:

            # read query test(query) image 
            frame_query = cv2.imread(path_query, 0)
            # preprocess the image 
            frame_query = cv2.equalizeHist(frame_query)
            frame_query  = cv2.GaussianBlur(frame_query, (7,7,),7)
            frame_query = cv2.bilateralFilter(frame_query, d=2, sigmaColor=95, sigmaSpace=95)
            # generate keypoints and descriptos for test image (obj1)
            obj1.detectAndCompute(frame_query)
            # match with reference descriptor (selected keyframe)
            good_matches = obj1.matchFeatures(des_reference)
            good_matches_ransac, mask = obj1.RANSAC(kps_reference, good_matches)# reduce outliers using RANSAC

            # get 3d points from a selected reference frame
            pts = data2sort(id, focal_length=35).pts3dFromJson('pts_3d_{}'.format(data_type), folder_id='json_data')
            # get only inlying keypoints of the test image and its corresponding 3D points
            pts3d = np.array([pts[match.queryIdx] for match in good_matches_ransac])
            pts2dimg = np.array([obj1.kps[match.trainIdx].pt for match in good_matches_ransac])

            # estimate the pose initialise PosePnP with RANSAC
            success, rotation_vec_, translation_vec_,_= cv2.solvePnPRansac(pts3d, pts2dimg , kMat_test, dist_coeffs, cv2.SOLVEPNP_EPNP )
            print(success)
            if success:# if pose estimate c

                #get ground truth for synthetic dataset
                path_rt_ = 'database/blender_data/GroundTruth/y-axis_FL35_r50/rt'
                path_rt_ = path_rt+'/'+id+'.txt'
                # print(path_rt)
                rtmat = data2sort(id, focal_length=35).get_rt_matrix(path_rt_)
                rtmat = np.array([rtmat[0][0:3], rtmat[1][0:3], rtmat[2][0:3] ])
                q = miscelaneous.rotation_vector_to_quaternion(rotation_vec_)
                euler = miscelaneous.quaternion_to_euler(q)
        
                # get estimated rotation matrix from solvePnP 
                rotation_matrix = cv2.Rodrigues(rotation_vec_)[0]

                # Calculate relative pose(rotation)
                path_rt_ = 'database/blender_data/GroundTruth/y-axis_FL35_r50/rt'
                relative_rotation_euler = miscelaneous.getRelativeRotAngle(path_rt_, id, rotation_vec_)

                # calculate variance of the angles between two consecutive frames
                if reset_kf_counter > 0:
                    variance = miscelaneous.euclideanVariance(relative_rotation_reference, relative_rotation_euler)
                relative_rotation_reference=relative_rotation_euler
                print(variance)
                reset_kf_counter+=1


                if mode_update == 'relativeRotation':

                    # print(variance)
                    if variance < 1 and reset_kf_counter>0: # if anles variance does not excced threshold do not update keyframe
                        print(translation_vec_,relative_rotation_euler, id, path_query)
                        continue

                    elif 1 < variance < 15: # if variance more than 2 the folowing keyframe selected based on relative rotation
                        new_id,_,rounded_y,_ = miscelaneous.updateKeyframeID(id, relative_rotation_euler)
                        if rounded_y > 360:# frames which at angles 360 or 180 dropped from database to avoid ambiguities 
                            idx = 0
                            continue

                        elif rounded_y != 180:
                            obj1.updateKeyframe(new_id, data_type=data_type)
                            kps_reference = obj1.kps
                            des_reference= obj1.des
                            id = new_id
                            reset_kf_counter += 1
                            print('Keyframe updated!')
                            print(translation_vec_,relative_rotation_euler, id, path_query)
                    if variance >15:# if variance exceed threshold re-initialize new reference keyframe
                        idx = 0
                        print('Frame dropped, reseting initial keyframe...') # searching for new reference keyframe in the database
                        continue


                elif mode_update == 'variance':# keyframe update mode - variance 


                    if variance < 2 and reset_kf_counter>0:# keyframe not updated if variance does not exceed threshold
                        # print(variance)
                        xyz = id.split('_')
                        x = float(xyz[3]); y = float(xyz[2]); z = float(xyz[1])
                        print('Estimated attitude:','x:', relative_rotation_euler[0]+x, \
                              'y:', relative_rotation_euler[1]+y, 'z:', relative_rotation_euler[2]+z )
                        print('Estimated pose:', 'x:' ,translation_vec_[0]*0.1,'y:' \
                              ,translation_vec_[1]*0.1,'z:' ,translation_vec_[2]*0.1 )
                        print(translation_vec_,relative_rotation_euler, id, path_query)
                        continue

                    elif 15 > variance > 2:# if threshold exceeded update keyframe 

                        new_id = obj1.computeNeighbours(id) # generate a list of possible new keyframes 
                        best_id,_ = obj1.getBestKeyframe(new_id, data_type=data_type) # take the keyframe with most inliers 
                        # obj1.updateKeyframe(best_id[0], data_type=data_type)
                        kps_reference = best_id[1]
                        des_reference = best_id[2]
                        id = best_id[0]
                        ## reset variables
                        # reset_kf_counter = 0
                        relative_rotation_reference = 0 
                        variance = 0
                        print('Reference frame updated')
                        xyz = id.split('_')
                        x = float(xyz[3]); y = float(xyz[2]); z = float(xyz[1])
                        print('Estimated attitude:','x:', relative_rotation_euler[0]+x, \
                              'y:', relative_rotation_euler[1]+y, 'z:', relative_rotation_euler[2]+z )
                        print('Estimated pose:', 'x:' ,translation_vec_[0]*0.1,'y:' \
                              ,translation_vec_[1]*0.1,'z:' ,translation_vec_[2]*0.1 )
                        print(translation_vec_,relative_rotation_euler, id, path_query)
                        print(translation_vec_,relative_rotation_euler, id, path_query)

                    if variance >16:
                        idx = 0
                        reset_kf_counter = 0
                        variance = 0
                        print('Frame dropped, reseting initial keyframe...')
                        continue

                else:# if pose not could not be estimated 
                    raise ValueError('Pose estimate not computed!')
                
    break






                        
 
        






















