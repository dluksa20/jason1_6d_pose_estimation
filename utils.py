import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import math
import json
import cv2



'''Sort the images class'''
'''------------------------------------------------------------------------------------------------------'''
class IMGSorter:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_number_before_zero(self, image_path):
        filename = os.path.basename(image_path)
        parts = filename.split('.')
        parts_ = parts[0].split('_')
        x = int(parts_[1])
        z = int(parts_[2])
        y = int(parts_[3])
        return  x, z, y

    def sort_images_by_number_before_zero(self, image_paths):
        return sorted(image_paths, key=self.get_number_before_zero)

    def get_sorted_image_paths(self, format):# format - variable to senter file format has the format of '0.exr'/ '0.png' / '0.txt' and so on
        image_paths = os.listdir(self.base_path)
        image_paths = [os.path.join(self.base_path, filename) for filename in os.listdir(self.base_path) if filename.endswith(str(format)) ]
        return self.sort_images_by_number_before_zero(image_paths)
    
    def get_image_path(self, filename):
        # Join the directory and filename to create the image path
        image_path = os.path.join(self.base_path, filename)
        return image_path


'''3D point extraction and data manipulators class'''
'''------------------------------------------------------------------------------------------------------'''
class helper():
    def __init__(self) -> None:
        pass
    @staticmethod
    def register_kps_with_dmaps(kp, des, dmaps, kmat, rt):
        pt_3d_list = []
        kp_registed = []
        des_registed = []
        dmaps_size = dmaps.shape[0]
        zed = np.array([0,0,1])
        for i in range(len(kp)):        
            x = np.array(kp[i].pt).astype('int')[1]
            y = np.array(kp[i].pt).astype('int')[0]
            # obtain depth value from depth map
            if  x < dmaps_size and y < dmaps_size: 
                zdval = dmaps[x][y]
                if zdval >= 0:
                    orig, my_dir, _, _ = helper.backproject(kp[i].pt, kmat, rt)  # backproject the keypoint
                    b_pt = np.linalg.inv(np.mat(kmat)) *  np.mat(helper.pt_2_ptvec(kp[i].pt)).T
                    b_pt = b_pt/b_pt[2]
                    theta = math.acos(np.dot(b_pt.T,zed) / np.linalg.norm(b_pt))
                    dval = zdval *1.0/math.cos(theta)
                    pt_3d = orig + dval * my_dir # 3d point with correct depth
                    pt_3d_list.append(pt_3d.T)
                    kp_registed.append(kp[i])
                    des_registed.append(des[i])
        des_registed = np.array(des_registed).astype(np.uint8) 
        return kp_registed, des_registed, pt_3d_list
    @staticmethod
    def backproject(pt,kf_k, kf_rt):
        # pt = pt_.copy()
        # my_lambda = 8.0
        point2d_vec = helper.pt_2_ptvec(pt)  #*my_lambda
        
        # point in camera coordinates
        X_c = np.linalg.inv(np.mat(kf_k)) * np.mat(point2d_vec).T  #3*1
        
        # point in world coordinates
        mat = kf_rt[:,0:3].copy()
        t_vec = np.mat(kf_rt[:,3]).T
        q_inv = helper.su2_inv(helper.so3_to_su2(mat))   
        X_w = helper.su2_mtp_point3f(q_inv.copy() ,(X_c - t_vec)) # 3*1
        
        # centre of projection
        orig = helper.su2_mtp_point3f(q_inv.copy() , (-1*t_vec)) #3*1
        
        # ray direction vector
        my_dir = X_w - orig                #3*1
        my_dir = my_dir / np.linalg.norm(my_dir) #3*1
        
        return orig, my_dir, X_c, X_w
    @staticmethod
    def pt_2_ptvec(pt):
        point2d_list = list(pt)
        point2d_list.append(1.0)
        point2d_vec = np.array(point2d_list)
        return point2d_vec
    @staticmethod
    def su2_inv(su2):  
        su2_copy = su2.copy()
        su2_copy[0:3] = su2_copy[0:3]*-1
        return su2_copy
    @staticmethod
    def so3_to_su2(mat):
        r11 = mat[0][0]
        r12 = mat[0][1]
        r13 = mat[0][2]
        r21 = mat[1][0]
        r22 = mat[1][1]
        r23 = mat[1][2]
        r31 = mat[2][0]
        r32 = mat[2][1]
        r33 = mat[2][2] 
        
        sqrt_arg = np.array([1 + r11 + r22 + r33,
                            1 + r11 - r22 - r33,
                            1 - r11 + r22 - r33,
                            1 - r11 - r22 + r33])
        idx_max = sqrt_arg.argmax()
        arg = np.max(sqrt_arg)
        
        if idx_max == 0:
            q4 = 0.5*math.sqrt(arg)
            q1 = 1/(4*q4)*(r23 - r32)
            q2 = 1/(4*q4)*(r31 - r13)
            q3 = 1/(4*q4)*(r12 - r21)
        elif idx_max == 1:
            q1 = 0.5*math.sqrt(arg)
            q2 = 1/(4*q1)*(r12 + r21)
            q3 = 1/(4*q1)*(r13 + r31)
            q4 = 1/(4*q1)*(r23 - r32)
        elif idx_max == 2:
            q2 = 0.5*math.sqrt(arg)
            q1 = 1/(4*q2)*(r21 + r12)
            q3 = 1/(4*q2)*(r23 + r32)
            q4 = 1/(4*q2)*(r31 - r13)
        elif idx_max == 3:
            q3 = 0.5*math.sqrt(arg)
            q1 = 1/(4*q3)*(r31 + r13)
            q2 = 1/(4*q3)*(r32 + r23)
            q4 = 1/(4*q3)*(r12 - r21)   

        norm = math.sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4)
        su2 = np.array([q1/norm, q2/norm, q3/norm, q4/norm])
        return su2
    @staticmethod
    def su2_mtp_point3f(lhs,rhs):
        rhs_list = list(np.array(rhs).squeeze())
        rhs_list.append(0.0)
        rhs_su2 = np.array(rhs_list)
        
        su2_temp = helper.su2_mtp_su2(lhs,rhs_su2)
        su2_comp = helper.su2_mtp_su2(su2_temp, helper.su2_inv(lhs))
        su2_vec = np.mat(su2_comp[0:3]).T
        return su2_vec
    @staticmethod
    def su2_mtp_su2(lhs, rhs):
        lhs_e_0 = lhs[0]
        lhs_e_1 = lhs[1]
        lhs_e_2 = lhs[2]
        lhs_q = lhs[3]
        
        rhs_e_0 = rhs[0]
        rhs_e_1 = rhs[1]
        rhs_e_2 = rhs[2]
        rhs_q = rhs[3]
        
        e_0 = lhs_e_0*rhs_q - lhs_e_1*rhs_e_2 + lhs_e_2*rhs_e_1 + lhs_q*rhs_e_0
        e_1 = lhs_e_0*rhs_e_2 - lhs_e_2*rhs_e_0 + lhs_e_1*rhs_q + lhs_q*rhs_e_1
        e_2 =-lhs_e_0*rhs_e_1 + lhs_e_1*rhs_e_0 + lhs_e_2*rhs_q + lhs_q*rhs_e_2
        q = -lhs_e_0*rhs_e_0 - lhs_e_1*rhs_e_1 - lhs_e_2*rhs_e_2 + lhs_q*rhs_q
        
        su2_out = np.array([e_0, e_1, e_2, q])   # norm(su2_out) may not equal to 1 due to computer data formate 
        # norm = math.sqrt(e_0*e_0 + e_1*e_1 + e_2*e_2 + q*q)
        # su2_out = np.array([e_0/norm, e_1/norm, e_2/norm, q/norm])
        # if q > 1.0:
        #     su2_out = np.array([0.0, 0.0, 0.0, 1.0])
        return su2_out
    @staticmethod
    def su2_error(lhs, rhs):
        dq = helper.su2_mtp_su2(lhs, helper.su2_inv(rhs))
        if dq[3] < 1.0:
            dphi = 2.0*math.acos(dq[3])
        else:
            dphi = 2.0*math.acos(1.0)
        if dphi > math.pi:
            dphi = 2*math.pi - dphi   
        return dphi


'''Data pre-processing class'''
'''------------------------------------------------------------------------------------------------------'''
class data2sort:
    def __init__(self, id, focal_length):
            self.id = id
            self.keypoints_data = {}
            self.descriptor_data = {}
            self.first_number = []
            self.first_number = int(self.id.split('_')[:2][0])#Note file encoding format - '90_240_0_0' varies depending on range and angle
            self.focal_length =focal_length
    def keyPoint_2_json(self, arr, folder, folder_id):
        # Create a folder for keypoints (if it doesn't exist) 
        # keypoints_json/y-axis_FL75_r{}'first_number'} - default directory folder 
        # first_number - variable represents range
        keypoints_folder = 'database/{}/{}/y-axis_FL{}_r{}'.format(folder_id, folder, self.focal_length, self.first_number)
        os.makedirs(keypoints_folder, exist_ok=True)

        data = {}
        counter = 0
        for i in arr:
            data['KeyPoint_%d' % counter] = []
            data['KeyPoint_%d' % counter].append({'x': i.pt[0],
                                                'y': i.pt[1],
                                                'size': i.size})
            counter += 1

        with open(os.path.join(keypoints_folder, '{}.json'.format(self.id)), 'w') as outfile:
            json.dump(data, outfile, indent=2)


    def descriptor_2_json(self, arr, folder, folder_id):
        # Create a folder for keypoints (if it doesn't exist)
        descriptors_folder = 'database/{}/{}/y-axis_FL{}_r{}'.format(folder_id, folder,self.focal_length, self.first_number)
        os.makedirs(descriptors_folder, exist_ok=True)
        data = {}
        counter = 0
        for i in arr:
            data['KeyPoint_%d_Descriptor' % counter] = []
            data['KeyPoint_%d_Descriptor' % counter].append(i.tolist())
            counter += 1

        with open(os.path.join(descriptors_folder, '{}.json'.format(self.id)), 'w') as outfile:
            json.dump(data, outfile, indent=2)
    
    def kpsFromJson(self, folder, folder_id):
            
            path = 'database/{}/{}/y-axis_FL{}_r{}/{}.json'.format(folder_id,folder,self.focal_length, self.first_number ,self.id)
            # Read JSON data from the file
            with open(path, 'r') as file:
                data = json.load(file)
            xy = []
            xy_list = []
            # Access values in the JSON dictionary and convert to integers
            for i in range(len(data)):
                keypoint_0 = data["KeyPoint_{}".format(i)]
                x_coordinate = keypoint_0[0]["x"]
                y_coordinate = keypoint_0[0]["y"]
                xy = [x_coordinate, y_coordinate]
                xy_list.append(xy)
            return xy_list
    def desFromJson(self, folder, folder_id):
            
            path = 'database/{}/{}/y-axis_FL{}_r{}/{}.json'.format(folder_id, folder,self.focal_length, self.first_number ,self.id)
            # Read JSON data from the file
            with open(path, 'r') as file:
                data = json.load(file)
            des = []
            des_list = []
            # Access values in the JSON dictionary and convert to integers
            for i in range(len(data)):
                descriptor_0 = data["KeyPoint_{}_Descriptor".format(i)]
                descriptor_0 = descriptor_0[0]
                des_list.append(descriptor_0)
            return des_list
    def get_k_matrix (self, file_path):
        with open(str(file_path) , 'r') as file:
            contents = file.readlines()
            matrix = []
            for line in contents:
                row = [float(value) for value in line.split()]
                matrix.append(row)
            matrix = matrix
        return matrix
    def get_rt_matrix (self, file_path):
        with open(str(file_path) , 'r') as file:
            contents = file.readlines()
            matrix = []
            for line in contents:
                row = [float(value) for value in line.split()]
                matrix.append(row)
            matrix = matrix
        return matrix

    def pts3D_2_json(self, arr, folder, folder_id):
        pts3D_folder = 'database/{}/{}/y-axis_FL{}_r{}'.format(folder_id, folder, self.focal_length, self.first_number)
        os.makedirs(pts3D_folder, exist_ok=True)

        data = {}
        counter = 0
        for i in arr:
            data['KeyPoint_%d' % counter] = []
            data['KeyPoint_%d' % counter].append({'x': i[0, 0],
                                                  'y': i[0, 1],
                                                  'z': i[0,2]})
            counter += 1

        with open(os.path.join(pts3D_folder, '{}.json'.format(self.id)), 'w') as outfile:
            json.dump(data, outfile, indent=2)

    
    def pts3dFromJson(self, folder,folder_id):
        path = 'database/{}/{}/y-axis_FL{}_r{}/{}.json'.format(folder_id, folder,self.focal_length, self.first_number ,self.id)
        # Read JSON data from the file
        with open(path, 'r') as file:
            data = json.load(file)
        pts = []
        pts_list = []
        # Access values in the JSON dictionary and convert to integers
        for i in range(len(data)):
            keypoint_0 = data["KeyPoint_{}".format(i)]
            x_coordinate = keypoint_0[0]["x"]
            y_coordinate = keypoint_0[0]["y"]
            z_coordinate = keypoint_0[0]["z"]
            pts = [x_coordinate, y_coordinate, z_coordinate]
            pts_list.append(pts)
        return pts_list
    

# define class for natural feature
class NaturalFeature:
    border = 100
    def __init__(self, detector, descriptor, border=border):
        # imgs : class TestImgs
        self.vis_kps = []
        self.vis_des = []
        self.ehd_kps = []
        self.vis_3d = []
        self.ehd_des = []
        self.detector = detector
        self.descriptor = descriptor
        self.ehd_detector = detector
        self.ehd_kps_int = False
        self.ehd_kps_ssc_num = 0  # Use the given number to select kps. 0: use all
        self.ehd_chunk_size = 40
        self.canny_thrd1 = 20
        self.canny_thrd2 = 40
       
        self.border = border
        return None
    
    # method generate keypoints and descriptors
    def Generate_KpsDes(self, imgs):

        border = self.border
        vis_border = cv2.copyMakeBorder(imgs, border, border,border, border, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        vis_kps, vis_des = self.Detect_Kp(vis_border)
        self.vis_kps = vis_kps
        self.vis_des = vis_des
        return None


    # method: get corresponding 3D points for both tir and vis bands
    def Get_KpEhd_3dPoints(self, dmap, kmat, rt):    
        # registe kp, des to depth map and generate 3d list for keyframe
        vis_kps_o, vis_des_o, vis_pt_3d = helper.register_kps_with_dmaps(self.vis_kps, self.vis_des, dmap, kmat,rt)
        self.vis_kps = vis_kps_o
        self.vis_des = vis_des_o
        self.vis_3d = vis_pt_3d          
        return None
    
    # method: detect keypoints
    def Detect_Kp(self, img_border):

        kps = self.detector.detect(img_border,None)
        kps, des = self.descriptor.compute(img_border, kps) 
        # redefine keypoint.pt to original image frame
        for i in range(len(kps)):
            kps[i].pt = tuple(np.array(kps[i].pt) - np.array([self.border,self.border]))    
        return list(kps), des
        
 






