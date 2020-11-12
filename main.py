import cv2
import numpy as np


class opencv:
    def show_image(image):
        cv2.imshow('image window', image)
        cv2.waitKey(0)

        # cv2.destoyAllWindows()

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
            c = sorted(contours, key=cv2.contourArea, reverse=True)

            for cout in c:
                # compute the rotated bounding box of the largest contour
                rect = cv2.minAreaRect(cout)
                box = np.int0(cv2.boxPoints(rect))

                cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

        return image

    def find_cone(path):
        lower = [0, 135, 135]
        upper = [15, 255, 255]

        return opencv.make_mask(path, lower, upper)

    def line_detection_method_1(path):
        image = opencv.get_image(path)

        edges = cv2.Canny(image, 50, 150)
        rho_accuracy = 1
        theta_accuracy = np.pi / 180
        min_length = 200
        lines = cv2.HoughLines(edges, rho_accuracy, theta_accuracy, min_length)

        for line in lines:
            rho, theta = line[0]
            a = np.cos ( theta )
            b = np.sin ( theta )
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return image

    def line_detection_method_2(path):
        lower = [0, 0, 255]
        upper = [255, 255, 255]

        return opencv.make_mask ( path, lower, upper )

opencv.show_image(opencv.line_detection_method_2('demo_lane.png'))