import cv2
import numpy as np

class PerspectiveCalculator:
    def __init__(self, R):
        self.R = R


    def process_frame(self, frame):
        discs = self.detect_disc(frame)
        radii = []
        for disc in discs:
            x, y, r = disc
            self.draw_circles(frame, x, y, r)
            radii.append(r)
        return self.calculate_ratio(max(radii)) if radii else None

    def detect_disc(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 2)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 2, 100, param1=200, param2=100)
        if circles is None:
            return []
        circles = np.uint16(np.around(circles))
        return circles[0]

    def draw_circles(self, frame, x, y, r):
        cv2.circle(frame, (int(x), int(y)), int(r), (0, 0, 255), 2)

    def calculate_ratio(self, r):
        return self.R / r
