import cv2
import numpy as np
from time import sleep
import src.functions as functions

class DiscTracker:

    def __init__(self, videoFrames, pixelToRealRatio, fps):
        self.frames = videoFrames[:]
        self.pixelToRealRatio = pixelToRealRatio
        self.fps = fps
        self.frameShape = self.frames[0].shape

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
                if point[1] < highestPoint[1]: highestPoint = point
                if point[1] > lowestPoint[1]: lowestPoint = point
                if point[0] > rightmostPoint[0]: rightmostPoint = point
                if point[0] < leftmostPoint[0]: leftmostPoint = point
        return (highestPoint, lowestPoint, leftmostPoint, rightmostPoint)



    def findDisc(self, background):
        rects = []
        firstFrameIndex, lastFrameIndex = None, None
        for i, frame in enumerate(self.frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_blur = cv2.GaussianBlur(gray, (9,9), 2)
            absoluteDif = cv2.absdiff(image_blur, background)
            ret, threshold = cv2.threshold(absoluteDif, 90, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            highestPoint, lowestPoint, leftmostPoint, rightmostPoint = self.findExtrema(contours, frame.shape)

            center_x = (leftmostPoint[0] + rightmostPoint[0]) // 2
            center_y = (highestPoint[1] + lowestPoint[1]) // 2
            width = np.abs(rightmostPoint[0] - leftmostPoint[0])
            height = np.abs(lowestPoint[1] - highestPoint[1])

            # TODO - fix ugly code
            if not (rightmostPoint[0] >= frame.shape[1] - 1 or leftmostPoint[0] <= 1 or highestPoint[1] <= 1 or lowestPoint[1] >= frame.shape[0]):
                rects.append((center_x, center_y, width, height, i))
                color = (255, 0, 0)
            else: #doesn't actually track the proper end for Brandon flick cropped because it leaves off the top
                if firstFrameIndex == None: firstFrameIndex = i
                elif firstFrameIndex != None and lastFrameIndex == None: lastFrameIndex = i
                color = (0, 0, 255)

            cv2.rectangle(frame, (center_x - width // 2, center_y - height // 2),
                        (center_x + width // 2, center_y + height // 2), color, 2)

            # if leftmostPoint[0] == frame.shape[1] and rightmostPoint[0] == 0:  # alt method to ignore frames with no disc

            if not leftmostPoint[0] >= rightmostPoint[0]:
                if firstFrameIndex == None: firstFrameIndex = i
                # cv2.imshow('threshold', threshold)
                cv2.imshow('frame', frame)
                if cv2.waitKey(60) == ord('q'):
                    break

        print("HERE!!!!", firstFrameIndex, lastFrameIndex)
        for i in range(firstFrameIndex, lastFrameIndex + 1):
            cv2.imshow('frame', self.frames[i])
            # sleep(0.1)
            if cv2.waitKey(60) == ord('q'):
                break
        return rects

    def findDiscSpeed(self, discs):
        dt = 1/self.fps
        deltas = []
        skip = 1
        for i in range(1, len(discs)):
            distance = functions.distanceCalc(
                (discs[i][0], discs[i][1]),
                (discs[i-1][0], discs[i-1][1])
            )
            if distance <= 0 or discs[i][3] > self.frameShape[0] / 2:
                skip += 1
            else:
                deltas.append(distance / skip)
                skip = 1
        constant = self.pixelToRealRatio * (1 / dt)
        deltas = functions.remove_outliers(deltas, 1)
        # print("Ratio:", self.pixelToRealRatio)
        # print("Deltas: ", deltas)
        speeds = [val * constant for val in deltas]
        # median = np.median(speeds)
        print("Speeds: ", speeds)
        return np.mean(speeds)
        # return median

