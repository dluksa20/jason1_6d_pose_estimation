import cv2
import numpy as np
from utils import data2sort
from utils import NaturalFeature
import utils_extra as miscelaneous


'''Note:
    - this script provides more visual pose estimation process 
    - this sript is just for testing purposes we assume that we know preliminar pose of 
    the object in the test image and select the database keyframe manually

'''

FL =35 # camera focal length
id = '50_0_270_-10' # selected keyframe id 
r = [50] # selected keyframe range
idxs = 0 


'''Initialize detector and descriptor and brute force matcher'''
'''------------------------------------------------------------------------------------------------------'''
detector1 = cv2.BRISK_create()
descriptor1 = cv2.xfeatures2d.FREAK_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

'''Create object instances for database keyframe and test image keyframe'''
'''------------------------------------------------------------------------------------------------------'''
feature1 = NaturalFeature(detector=detector1, descriptor=descriptor1, border=0)
feature2 = NaturalFeature(detector=detector1, descriptor=descriptor1, border=0)

'''Set up the directories'''
'''------------------------------------------------------------------------------------------------------'''
# img_folder_test = 'database/Test_images_2'
# img_paths_test = miscelaneous.sort_images_in_folder(img_folder_test)

img_test = cv2.imread('database/Test_rot/img_rot_2.0m_seg/frame_10.png',0)
img_database = cv2.imread('database/blender_data/Train_img/y-axis_FL{}_r{}/renders/{}.png'.format(FL,r[idxs], id),0)

# pre-process image
img_database = cv2.equalizeHist(img_database)
img_test = cv2.equalizeHist(img_test)
# img_test = cv2.bilateralFilter(img_test, d=4, sigmaColor=95, sigmaSpace=95)
# img_test  = cv2.GaussianBlur(img_test, (7,7,),7)

'''Get ground truth rotation matrix and database frame camera matrix (k)'''
'''------------------------------------------------------------------------------------------------------'''
dmap = cv2.imread('database/blender_data/Train_img/y-axis_FL{}_r{}/dmaps_norm/{}.exr'.format(FL,r[idxs], id), cv2.IMREAD_ANYDEPTH)
id_ = id.split('.'); id_ = id[0]
kMat = data2sort(id_,FL).get_k_matrix('database/blender_data/Train_img/y-axis_FL{}_r{}/k/{}.txt'.format(FL,r[idxs],id))# 0 - cause k -matrix at all instances is same
rtMat = data2sort(id_,FL).get_rt_matrix('database/blender_data/Train_img/y-axis_FL{}_r{}/rt/{}.txt'.format(FL,r[idxs],id))# 0 - cause k -matrix at all instances is same
kMat = np.array(kMat)
rtMat = np.array(rtMat)

# query(test image) camera matrix 
kMat_test = np.array([[1991.111084, 0.000000, 320.00 ], 
                    [0.000000, 1493.333374, 240.00 ], 
                    [0.000000, 0.000000, 1.000000 ]])

'''Compute keypoints, descriptors and corresponding 3D points'''
'''------------------------------------------------------------------------------------------------------'''
feature1.Generate_KpsDes(img_test)
feature2.Generate_KpsDes(img_database)
feature2.Get_KpEhd_3dPoints(dmap, kMat, rtMat)


'''Match features'''
'''------------------------------------------------------------------------------------------------------'''
good_matches = bf.match(feature1.vis_des, feature2.vis_des)
image_with_keypoints = cv2.drawKeypoints(img_test, feature1.vis_kps, None, color=(0, 255, 0))
cv2.imshow('', image_with_keypoints)
cv2.waitKey(0)



'''Aplly ransac to remove outliers'''
'''------------------------------------------------------------------------------------------------------'''
src_pts = np.float32([feature1.vis_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([feature2.vis_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()
good_matches_ = [m for i, m in enumerate(good_matches) if mask[i] == 1]
draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
matched_image = cv2.drawMatches(img_test,feature1.vis_kps,img_database,feature2.vis_kps,good_matches,None,**draw_params)
'''-------------------------------------------------------------------------------------------------------'''
# matched_image = cv2.drawMatches(img1, feature1.vis_kps, img2, feature2.vis_kps, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

'''distortion coefficients of test(ZED) camera'''
dist_coeffs = np.array([-0.1744, 0.0273, 0.0007, -0.0003, 0], dtype=np.float32)

'''current test frame 2d keypoints and corresponding 3D points'''
pts2dimg = np.array([feature1.vis_kps[match.queryIdx].pt for match in good_matches_])
pts3d = np.array([feature2.vis_3d[match.trainIdx] for match in good_matches_])



'''Estimate the pose'''
'''------------------------------------------------------------------------------------------------------'''
if len(good_matches_) >= 4:

    success, rotation_vec_, translation_vec_,_= cv2.solvePnPRansac(pts3d, pts2dimg, kMat_test, dist_coeffs, cv2.SOLVEPNP_EPNP )
    print(success)
    R, _ = cv2.Rodrigues(rotation_vec_)

    '''attitude ground truth matrix'''
    rtmat = np.array([rtMat[0][0:3], rtMat[1][0:3], rtMat[2][0:3] ])
    q = miscelaneous.rotation_vector_to_quaternion(rotation_vec_)
    euler = miscelaneous.quaternion_to_euler(q)
    '''estimated rotation matrix'''
    rotation_matrix = cv2.Rodrigues(rotation_vec_)[0]
    

    '''Compute relative rotation in degrees'''
    relative_rotation_matrix = np.dot(rotation_matrix, np.linalg.inv(rtmat))
    relative_rotation_euler = cv2.RQDecomp3x3(relative_rotation_matrix)[0]
    print('Pose(m) ZYX: ', translation_vec_.flatten())
    print('Relative pose(deg) ZYX: ',relative_rotation_euler)
    print(len(good_matches_))

    cv2.imshow('inliers', matched_image)
    cv2.waitKey(0)



