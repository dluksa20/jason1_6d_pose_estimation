import numpy as np
import cv2 
import os
import math
from utils import data2sort

'''Extra utils and matrix/vector manipulators'''
'''------------------------------------------------------------------------------------------------------'''

# rotation matrix to euler angles 
def rotation_matrix_to_euler_angles(R):
    R = np.array(R)  # Convert the rotation matrix to a NumPy array
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])

    return yaw, pitch, roll


# quaternion to rotation matrix domain
def quaternion_to_rotation_matrix(q):
    # Normalize the quaternion if necessary
    q = q / np.linalg.norm(q)

    x, y, z,w = q

    # Compute the elements of the rotation matrix
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w

    yy = y * y
    yz = y * z
    yw = y * w

    zz = z * z
    zw = z * w

    # Construct the 3x3 rotation matrix
    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
    ])

    return rotation_matrix


# rotation vector to quaternion 
def rotation_vector_to_quaternion(rvec):
    # Convert the rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # Convert the rotation matrix to a quaternion
    w = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
    x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * w)
    y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * w)
    z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * w)

    return np.array([x, y, z, w])
# quaternion to euler angles
def quaternion_to_euler(q):
    w, x, y, z = q

    # Roll (x-axis rotation)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    # Pitch (y-axis rotation)
    pitch = math.asin(2 * (w * y - z * x))

    # Yaw (z-axis rotation)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return np.array([roll, pitch, yaw])

# sort test frames in ascending order
def sort_images_in_folder(folder_path):
    file_names = os.listdir(folder_path)

    image_files = [file for file in file_names if file.lower().endswith((".jpg", ".png", ".jpeg"))]

    def extract_number_from_filename(filename):
        number = int(filename.split("_")[1].split(".")[0])  # Extract number from filename
        return number
    sorted_image_files = sorted(image_files, key=extract_number_from_filename)

    sorted_image_paths = [os.path.join(folder_path, file) for file in sorted_image_files]

    return sorted_image_paths


# compute relative pose between database keyframe and test frame
def getRelativeRotAngle(path_rot, img_id, rotation_vec_):
    path_rt_ = path_rot+'/'+img_id+'.txt'
    # print(path_rt)
    rtmat = data2sort(img_id, focal_length=35).get_rt_matrix(path_rt_)
    rtmat = np.array([rtmat[0][0:3], rtmat[1][0:3], rtmat[2][0:3] ])
    rotation_matrix = cv2.Rodrigues(rotation_vec_)[0]
    # # Compute relative rotation matrix
    relative_rotation_matrix = np.dot(rotation_matrix, np.linalg.inv(rtmat))
    # # Convert relative rotation matrix to Euler angles
    relative_rotation_euler = cv2.RQDecomp3x3(relative_rotation_matrix)[0]
    return relative_rotation_euler

# update current keyframe based on relative rotation
def updateKeyframeID(current_id, relative_rotation_euler):

    current_id = current_id.split('_')
    x_value = int(current_id[3]) + relative_rotation_euler[0]
    y_value = int(current_id[2]) + relative_rotation_euler[1]
    z_value = int(current_id[1]) + relative_rotation_euler[2]

    # Step values
    x_step = 5
    y_step = 2
    z_step = 5

    # Round to the nearest multiple of the step values
    rounded_x = round(x_value / x_step) * x_step
    rounded_y = round(y_value / y_step) * y_step
    rounded_z = round(z_value / z_step) * z_step

    # Set ceiling values
    x_ceiling = 10
    z_ceiling = 100

    # Apply ceiling values
    rounded_x = min(rounded_x, x_ceiling)
    rounded_z = min(rounded_z, z_ceiling)
    # Apply ceiling values for z component with range constraints
    if rounded_z > 100:
        rounded_z = z_ceiling
    elif rounded_z < 80:
        rounded_z = 80

    # Apply ceiling for x component within range -10 to 10, database limitations
    if rounded_x > x_ceiling:
        rounded_x = x_ceiling
    elif rounded_x < -10:
        rounded_x = -10
    if rounded_y == 0:
        rounded_y=2

    new_id = '{}_{}_{}_{}'.format(current_id[0], rounded_z, rounded_y, rounded_x)
    return new_id, rounded_x, rounded_y, rounded_z

# compute variance between current frame and previous frame
def computeVariance(previous_estimate, current_estimate):
    data_points = [previous_estimate,current_estimate]
    data_array = np.array(data_points)
    mean = np.mean(data_array, axis=0)
    squared_diffs = (data_array - mean) ** 2
    mean_squared_diffs = np.mean(squared_diffs, axis=0)
    overall_variance = np.mean(mean_squared_diffs)

    return overall_variance


# compute euclidean variance between current frame and previous frame
def euclideanVariance(previous_estimate, current_estimate):

    point1_array = np.array(previous_estimate)
    point2_array = np.array(current_estimate)
    distance = np.linalg.norm(point2_array - point1_array)

    return distance/3

# transform matrix to identity matrix
def rt_transform(R_source, R):
    T = R.T
    R_target = np.dot(T, R_source)
    return R_target

def rotation_error_in_degrees_total(R1, R2):
    # Compute the relative rotation matrix
    R_err = np.dot(R2, R1.T)
    angle_rad = np.arccos((np.trace(R_err) - 1) / 2)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

import numpy as np

def rotation_to_euler_angles(R):
    '''Convert a rotation matrix to Euler angles'''
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# compute relative pose between database keyframe and test frame
def getError(rot_true, rot_pre):
    # print(path_rt)
    rt_true = rot_true
    rt_pre = rot_pre
    # # Compute relative rotation matrix
    relative_rotation_matrix = np.dot(rt_pre, np.linalg.inv(rt_true))
    # # Convert relative rotation matrix to Euler angles
    relative_rotation_euler = cv2.RQDecomp3x3(relative_rotation_matrix)[0]
    return relative_rotation_euler
