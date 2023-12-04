import cv2
import numpy as np
from time import sleep

class DiscTracker:

    def __init__(self, videoFrames):
        self.frames = videoFrames[:]
        pass

    def findBackground(self):
        grayFrames = []
        for frame in self.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayFrames.append(gray)
        background = np.mean(grayFrames, axis=0).astype('uint8')
        return background

    def findExtrema(self, contours, imageShape):
        highestPoint = (0, imageShape[0])
        lowestPoint = (0, 0)
        rightmostPoint = (0, 0)
        leftmostPoint = (imageShape[1], 0)

        for contour in contours:
            for point in contour[:, 0]:
                # Find minimum point
                if point[1] < highestPoint[1]: highestPoint = point
                # highestPoint = (min(highestPoint[0], point[0]), min(highestPoint[1], point[1]))

                # Find maximum point
                if point[1] > lowestPoint[1]: lowestPoint = point
                # lowestPoint = (max(lowestPoint[0], point[0]), max(lowestPoint[1], point[1]))

                # Find rightmost point
                if point[0] > rightmostPoint[0]: rightmostPoint = point
                # rightmostPoint = max(leftmostPoint, tuple(point), key=lambda x: x[0])

                # Find leftmost point
                if point[0] < leftmostPoint[0]: leftmostPoint = point
                # leftmostPoint = min(leftmostPoint, tuple(point), key=lambda x: x[0])
        return (highestPoint, lowestPoint, leftmostPoint, rightmostPoint)



    def findDisc(self, background):
        rects = []
        for frame in self.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_blur = cv2.GaussianBlur(gray, (9,9), 2)
            absoluteDif = cv2.absdiff(image_blur, background)
            ret, threshold = cv2.threshold(absoluteDif, 90, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            highestPoint, lowestPoint, leftmostPoint, rightmostPoint = self.findExtrema(contours, frame.shape)
            center_x = (leftmostPoint[0] + rightmostPoint[0]) // 2
            center_y = (highestPoint[1] + lowestPoint[1]) // 2
            # print("left right", leftmostPoint, rightmostPoint)
            width = np.abs(rightmostPoint[0] - leftmostPoint[0])
            height = np.abs(lowestPoint[1] - highestPoint[1])

            rects.append(center_x, center_y, width, height)

            # cv2.rectangle(frame, (center_x - width // 2, center_y - height // 2),
            #             (center_x + width // 2, center_y + height // 2), (255, 0, 0), 2)
            # # cv2.imshow('threshold', threshold)
            # cv2.imshow('frame', frame)
            # sleep(0.1)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        return rects

    def findDiscSpeed(self, discMovement, pixelToRealRatio):
        deltas = []
        for discPos in discMovement:
            d