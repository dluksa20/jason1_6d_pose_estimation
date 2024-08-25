import pandas as pd 
import utils_extra
import os 
import numpy as np
from utils import helper, data2sort
from scipy.linalg import inv
# ground truth corrsponding images
paths_test = utils_extra.sort_images_in_folder('/Users/dovydasluksa/Documents/Project_MSc/Test_img/Test_images_3/images_9_28')
# ground truth processed
ground_truth = '/Users/dovydasluksa/Documents/Project_MSc/generateGroundTruth/time_stamps_images_.csv'
df = pd.read_csv(ground_truth)

rtMat = np.array([[0, -1, 0],[-0, 0, 1], [-1, 0, 0]])
rtMat_true = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
T = np.dot(rtMat_true, inv(rtMat))

print()

print(utils_extra.quaternion_to_euler([-0.0056	,0.652693	,0.011867	,-0.757509]))

rtPath = 'database/blender_data/Train_img_2/y-axis_FL35_r50/rt/50_90_300_0.txt'
id = '50_90_300_0'

rtMat = data2sort(id, focal_length=35).get_rt_matrix(rtPath)
rtMat = np.array([rtMat[0][:3],rtMat[1][:3], rtMat[2][:3]])


print(np.dot(rtMat, T).T)
print()
rt = utils_extra.quaternion_to_rotation_matrix([-0.004017,	0.534422,	0.00752	,-0.845175])
print(rt)




# for path in paths_test:
#     test_frame_id = os.path.basename(path)[:-4]
#     quaternion_true = np.array(df[df['ZED'] == test_frame_id].iloc[:, 3:7]).astype(np.float32)
#     pos_true = np.array(df[df['ZED'] == test_frame_id].iloc[:, 13:16]).astype(np.float32)
#     print('Quaternion:', test_frame_id,quaternion_true,'Position',pos_true)