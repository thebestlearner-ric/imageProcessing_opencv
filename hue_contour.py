import argparse
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())
# load the image and perform pyramid mean shift filtering
# to aid the thresholding step
image = cv2.imread(args["image"])
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Set range for lighter color and
# define mask
# hue, saturation, value(lighting)
lighter_lower = np.array([0, 60, 50])
lighter_upper = np.array([55, 90, 255])
lighter_mask = cv2.inRange(image_hsv, lighter_lower, lighter_upper)
# Set range for white colour and define mask
white_lower = np.array([0, 0, 0], np.uint8)
white_upper = np.array([0, 0, 255], np.uint8)
white_mask = cv2.inRange(image_hsv, white_lower, white_upper)

# Morphological Transform, Dilation
# for each color and bitwise_and operator
# between image and mask determines
# to detect only that particular color
kernal = np.ones((5, 5), "uint8")
# For lighter mask
lighter_mask = cv2.dilate(lighter_mask, kernal)
res_lighter = cv2.bitwise_and(image, image,
                              mask=lighter_mask)


contours, hierarchy = cv2.findContours(lighter_mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if 50000 < area < 80000:
        x, y, w, h = cv2.boundingRect(contour)
        image = cv2.rectangle(image, (x, y),
                              (x + w, y + h),
                              (0, 0, 0), 2)
        cv2.circle(image, (cX, cY), 10, (0, 0, 0), thickness=-1)
        cv2.putText(image, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 0))

# Creating contour to track white color
# contours, hierarchy = cv2.findContours(white_mask,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# for pic, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     if (area > 300):
#         x, y, w, h = cv2.boundingRect(contour)
#         image = cv2.rectangle(image, (x, y),
#                                    (x + w, y + h),
#                                    (0, 0, 255), 2)
#
#         cv2.putText(image, "White Colour", (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 5,
#                     (0, 0, 255))

# Creating contour to track green color
# contours, hierarchy = cv2.findContours(green_mask,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# for pic, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     if (area > 300):
#         x, y, w, h = cv2.boundingRect(contour)
#         image = cv2.rectangle(image, (x, y),
#                                    (x + w, y + h),
#                                    (0, 255, 0), 2)
#
#         cv2.putText(image, "Green Colour", (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     5, (0, 255, 0))
#
# # Creating contour to track blue color
# contours, hierarchy = cv2.findContours(blue_mask,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
# for pic, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     if (area > 300):
#         x, y, w, h = cv2.boundingRect(contour)
#         image = cv2.rectangle(image, (x, y),
#                                    (x + w, y + h),
#                                    (255, 0, 0), 2)
#
#         cv2.putText(image, "Blue Colour", (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     5, (255, 0, 0))
cv2.imshow("detect colour", image)
cv2.waitKey(10000)
# Now let's find the shape matching each dominant hue
# for i, peak in enumerate(peaks):
#     # First we create a mask selecting all the pixels of this hue
#     # mask = cv2.inRange(h, np.asarray(peak), np.asarray(peak))
#     mask = cv2.inRange(h, peak, peak)
#     print(mask)
#     # And use it to extract the corresponding part of the original colour image
#     blob = cv2.bitwise_and(image, image, mask=mask)
#
#     _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     for j, contour in enumerate(contours):
#         bbox = cv2.boundingRect(contour)
#         # Create a mask for this contour
#         contour_mask = np.zeros_like(mask)
#         cv2.drawContours(contour_mask, contours, j, 255, -1)
#
#         # Extract and save the area of the contour
#         region = blob.copy()[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
#         region_mask = contour_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
#         region_masked = cv2.bitwise_and(region, region, mask=region_mask)
#         cv2.imshow("region_masked", region_masked)
#
#         # Extract the pixels belonging to this contour
#         result = cv2.bitwise_and(blob, blob, mask=contour_mask)
#         # And draw a bounding box
#         top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])
#         cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
#         cv2.imshow("result", result)
#         cv2.waitKey(10000)
