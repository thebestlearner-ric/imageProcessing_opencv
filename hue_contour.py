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
cv2.imshow("detect colour", image)
cv2.waitKey(10000)

