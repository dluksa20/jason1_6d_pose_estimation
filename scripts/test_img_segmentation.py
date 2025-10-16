import cv2
import numpy as np
import utils_extra as miscelaneous
import os


'''Test images segmentation'''
'''------------------------------------------------------------------------------------------------------'''
# paths = miscelaneous.sort_images_in_folder('/Users/dovydasluksa/Documents/Project_MSc/Test_img/Test_images_3/images2/')
paths = miscelaneous.sort_images_in_folder('database/Test_rot/img_rot_2.0m/')

# main loops
for dir in paths:

    '''Load the image'''
    image = cv2.imread(dir, 1)

    '''define region of interset if needed''' 
    border_width = 300
    border_width2 = 300
    border_width1 = 500

    '''Create a copy of the original image'''
    image_with_borders = image.copy()


    '''Replace the left and right sides with black pixels'''
    image_with_borders[:, :border_width1] = (0, 0, 0)  # Left side
    image_with_borders[:, -border_width1:] = (0, 0, 0)  # Right side

    '''Replace the top and bottom sides with black pixels'''
    image_with_borders[:border_width, :] = (0, 0, 0)  # Top side
    image_with_borders[-border_width2:, :] = (0, 0, 0)  # Bottom side

    gray = cv2.cvtColor(image_with_borders, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (13,13),12)
    cv2.imshow('img', gray)
    cv2.waitKey(0)
    '''image thresholding'''
    ret, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the binary mask
    mask = np.zeros_like(image) # Create a mask with the same size as the image

    '''Draw the largest contour on the mask'''
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[:2]
    mask = np.zeros_like(image)

    # Draw the largest contours on the mask
    cv2.drawContours(mask, largest_contour, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Bitwise AND operation to apply the mask and remove the background
    result = cv2.bitwise_and(image, mask)
    # cv2.imshow('', result)
    # cv2.waitKey(0)

    
    '''do thresholding again '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''Pre-process the image'''
    # gray = cv2.GaussianBlur(gray,(13,13),7)
    _, thresholded = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)
    # cv2.imshow('', thresholded)
    # cv2.waitKey(0)
    # break
 
    '''Find contours in the binary mask'''
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    '''Filter out small contours (noise)'''
    min_contour_area = 100
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    '''Sort contours by area in descending order'''
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[:0]

    '''Create a mask for the lower half of the image'''
    height, width = thresholded.shape
    lower_half_mask = np.zeros((height, width), dtype=np.uint8)
    lower_half_mask[height // 2:, :] = 255

    '''Find the largest contour in the lower half of the image'''
    largest_contour = None
    for contour in contours:
        if np.any(np.bitwise_and(lower_half_mask, cv2.drawContours(np.zeros_like(lower_half_mask), [contour], 0, 255, thickness=cv2.FILLED))):
            largest_contour = contour
            break

    '''Turn the pixels of the largest contour in the lower half to black'''
    if largest_contour is not None:
        cv2.drawContours(result, [largest_contour], -1, (0, 0, 0), thickness=cv2.FILLED)
        cv2.drawContours(result, [largest_contour], -1, (0, 0, 0), thickness=15)

    '''show result'''
    cv2.imshow('modified_image.jpg', result)
    cv2.waitKey(0)

    '''Write final image to selected path'''
    # cv2.imwrite('/Users/dovydasluksa/Documents/Project_MSc/Test_img/Test_images_3/images2_/{}'.format(os.path.basename(dir)), result)
    cv2.imwrite('database/Test_rot/img_rot_2.0m_seg/{}'.format(os.path.basename(dir)), result)

    print('Written: {}'.format(os.path.basename(dir)))