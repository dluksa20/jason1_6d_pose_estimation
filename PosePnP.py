import cv2
import numpy as np
import os
from utils import data2sort
import time

'''Main pose estimation class'''
'''------------------------------------------------------------------------------------------------------'''
class PosePnP:
    # define parameter to initialize 
    def __init__(self, detector_type, descriptor_type):

        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.detector = None
        self.descriptor = None
        self.feature = None
        self.kps = None
        self.des = None
        self.matcher = None
        self.matcher_type = None
        self.matches = None
        self.neighbours = []
        self.nndr = None
    # set detector
    def initializeDetectors(self):
        if self.detector_type == 'ORB':
            self.detector = cv2.ORB_create()
        elif self.detector_type == 'BRISK':
            self.detector = cv2.BRISK_create()
        elif self.detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
        else:
            raise ValueError("Invalid detector type. Supported detectors: 'ORB', 'BRISK', 'SIFT'")
        
    # set descriptor
    def initializeDescriptors(self):
        if self.descriptor_type == 'ORB':
            self.descriptor = cv2.ORB_create()
        elif self.descriptor_type == 'BRISK':
            self.descriptor = cv2.BRISK_create()
        elif self.descriptor_type == 'FREAK':
            self.descriptor = cv2.xfeatures2d.FREAK_create()
        elif self.descriptor_type == 'SIFT':
            self.descriptor = cv2.SIFT_create()
        else:
            raise ValueError("Invalid descriptor type. Supported descriptors: 'ORB', 'BRISK', 'FREAK', 'SIFT'")
    # computer descriptors and keypoints
    def detectAndCompute(self, image):
        self.initializeDetectors()
        self.initializeDescriptors()
        kps = self.detector.detect(image, None)

        if len(kps) == 0:
            raise ValueError("No keypoints were detected.")
        
        kps, des = self.descriptor.compute(image, kps)

        self.des = des
        self.kps = kps

    # set feature matcher
    def setMatcher(self, matcher_type, nndr):
        if matcher_type == 'BF':
            self.matcher_type = 'BF'

            if self.descriptor_type == 'SIFT':
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

        elif matcher_type == 'BFKnn':
            self.matcher_type = 'BFKnn'
            self.matcher = cv2.BFMatcher()

        elif matcher_type == 'FLANN':
            self.matcher_type = 'FLANN'
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
            search_params = dict(checks=20)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError("Invalid matcher type. Supported matchers: 'BF', 'FLANN'")
        
        self.nndr = nndr
        return None
    # match features 
    def matchFeatures(self, query_des):

        if self.matcher_type is None:
            raise ValueError("Matcher is not set. Please call setMatcher before matching features.")
        
        if self.nndr is None and self.matcher_type == 'BF':

            matches = self.matcher.match(query_des, self.des)
            # matches = sorted(matches, key=lambda x: x.distance)
            return matches
        
        elif self.matcher_type == 'BFKnn' and self.nndr is None:
        
            matches = self.matcher.knnMatch(query_des, self.des, k=2)
            matches = [match for pair in matches for match in pair]
            # print(matches)
            return matches
        
        elif self.matcher_type == 'BFKnn' and self.nndr is not None:
            matches = self.matcher.knnMatch(query_des, self.des, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < self.nndr * n.distance:
                    good_matches.append(m)
            return good_matches

        elif self.matcher_type == 'FLANN':
            if self.nndr is None:
                query_des = np.array(query_des).astype(np.float32)
                self.des = np.array(self.des).astype(np.float32)
                matches = self.matcher.match(query_des, self.des)
                return matches
            else:
                query_des = np.array(query_des).astype(np.float32)
                self.des = np.array(self.des).astype(np.float32)
                matches = self.matcher.knnMatch(query_des, self.des, k=2)
                good_matches = []
                for i, (m, n) in enumerate(matches):
                    if m.distance < self.nndr * n.distance:
                        good_matches.append(m)
                return good_matches
            

    # remove outliers
    def RANSAC(self, kps2, matches):
        src_pts = np.float32([self.kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps2[m.queryIdx] for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        good_matches = [m for i, m in enumerate(matches) if mask[i] == 1]
        return good_matches, matchesMask
    

    # select initial keyframe
    def selectKeyframe(self, folder, thresh, data_type):

        start_time = time.time() #timeout error if frame not found in specific time
        timeout = 10  

        for filename in os.listdir(folder):
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise TimeoutError("The function timed out! Try to reduce reference keypoint threshold!")

            if filename.endswith('.json'):
                reference_kps_path = os.path.join(folder, filename)
                id = os.path.basename(reference_kps_path).split('.')[0]
                kps_reference = data2sort(id, focal_length=35).kpsFromJson('kps_json_{}'.format(data_type), folder_id='json_data')
                des_reference = data2sort(id, focal_length=35).desFromJson('des_json_{}'.format(data_type), folder_id='json_data')
                des_reference = np.array(des_reference).astype(np.uint8)
                
                good_matches = self.matchFeatures(des_reference)
                if len(good_matches) > 4 and good_matches != None :
                    good_matches_ransac, mask = self.RANSAC(kps_reference, good_matches)
                    if len(good_matches_ransac)> thresh:
                        return id, kps_reference, des_reference  
                elif len(good_matches)<4:
                    raise ValueError('Not Enough matches found, Increase the threshold!')     
                 
    # get nearest keyframes from database
    def computeNeighbours(self, current_frame_id):
        # Given ID values
        given_id = current_frame_id
        given_values = [int(val) for val in given_id.split('_')]

        # Vary ranges and steps around the given values
        vary_params = { # ranges to compute neighbours 
            'r': {'range': (0, 0), 'step': 10},
            'z': {'range': (-5, 5), 'step': 5},
            'y': {'range': (-6, 6), 'step': 2},
            'x': {'range': (-5, 5), 'step': 5}
        }

        thresholds = { # bounds of the dataset
            'r': {'min': 50, 'max': 100},
            'z': {'min': 80, 'max': 100},
            'y': {'min': 0, 'max': 360},
            'x': {'min': -10, 'max': 10}
        }

        new_ids = []
        # compute neighbours
        for vary_r in range(vary_params['r']['range'][0], vary_params['r']['range'][1] + 1, vary_params['r']['step']):
            for vary_z in range(vary_params['z']['range'][0], vary_params['z']['range'][1] + 1, vary_params['z']['step']):
                for vary_y in range(vary_params['y']['range'][0], vary_params['y']['range'][1] + 1, vary_params['y']['step']):
                    for vary_x in range(vary_params['x']['range'][0], vary_params['x']['range'][1] + 1, vary_params['x']['step']):
                        new_r = given_values[0] + vary_r
                        new_z = given_values[1] + vary_z
                        new_y = (given_values[2] + vary_y) % 360  # Handle the wrap-around
                        new_x = given_values[3] + vary_x

                        # Check thresholds
                        if thresholds['r']['min'] <= new_r <= thresholds['r']['max'] and \
                        thresholds['z']['min'] <= new_z <= thresholds['z']['max'] and \
                        thresholds['y']['min'] <= new_y <= thresholds['y']['max'] and \
                        thresholds['x']['min'] <= new_x <= thresholds['x']['max'] and new_y != 180 and new_y != 0: # 180 and 360 poses removed for this axis from database 
                            new_values = [str(new_r), str(new_z), str(new_y), str(new_x)]
                            new_id = "_".join(new_values)
                            new_ids.append(new_id)

        self.neighbours = new_ids
        return self.neighbours
    
    # update object keypoints and descriptors 
    def updateKeyframe(self,id, data_type):

        kps_reference = data2sort(id, focal_length=35).kpsFromJson('kps_json_{}'.format(data_type), folder_id='json_data')
        des_reference = data2sort(id, focal_length=35).desFromJson('des_json_{}'.format(data_type), folder_id='json_data')
        des_reference = np.array(des_reference).astype(np.uint8)

        self.des = des_reference
        self.kps = kps_reference

        return None
    # get keyframe with most inliers from computed neighbourhood
    def getBestKeyframe(self, ids, data_type):
        inliers = []
        ind = []
        for id  in ids:
            kps_reference = data2sort(id, focal_length=35).kpsFromJson('kps_json_{}'.format(data_type), folder_id='json_data')
            des_reference = data2sort(id, focal_length=35).desFromJson('des_json_{}'.format(data_type), folder_id='json_data')
            des_reference = np.array(des_reference).astype(np.uint8)

            good_matches = self.matchFeatures(des_reference)
            if len(good_matches) > 4 and good_matches != None :
                good_matches_ransac, mask = self.RANSAC(kps_reference, good_matches)
                inliers.append([id, kps_reference, des_reference, len(good_matches_ransac)])
                ind.append(len(good_matches_ransac))
        best_keyFrame = sorted(inliers, key=lambda x: x[3])
                
        return best_keyFrame[-1], ind
