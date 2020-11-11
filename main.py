import cv2
import numpy as np


class opencv:
    def show_image(image):
        cv2.imshow('image window', image)
        cv2.waitKey(0)

        cv2.destoyAllWindows()

    def get_image(path):
        image = cv2.imread(path)

        return image

    def change_to_hsv(path):
        image = cv2.imread(path)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        return hsv

    def make_mask(path, lower, upper):
        image = opencv.get_image(path)
        hsv = opencv.change_to_hsv(path)

        # define range of blue color in HSV
        lower_blue = np.array(lower)
        upper_blue = np.array(upper)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if(len(contours) >= 1):
            c = sorted(contours, key=cv2.contourArea, reverse=True )[0]

            # compute the rotated bounding box of the largest contour
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))

            cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

        return image

    def find_cone(path):
        lower = [0, 135, 135]
        upper = [15, 255, 255]

        return opencv.make_mask(path, lower, upper)


opencv.show_image(opencv.find_cone('demo_cone.png'))