import cv2
import numpy as np

class PerspectiveCalculator:
    """ Class for calculating perspective of the camera."""
    def __init__(self, calibration_path, R):
        """ Loads calibration from file and stores disc radius.
            Arguments:
                calibration_path: path to calibration file
                R: disc radius in cm
        """
        self.focal, self.centerx, self.centery = np.loadtxt(calibration_path,delimiter=' ')
        self.R = R


    def process_frame(self, frame):
        """ Detect disc in frame, estimate 3D positions, and draw on image
            Arguments:
                image: image to be processed
        """
        # Detect balls
        discs = self.detect_disc(frame)
        for disc in discs:
            x, y, r = disc
            X, Y, Z = self.calculate_position(x, y, r)
            self.draw_circles(frame, x, y, r, Z)


    def detect_disc(self, frame):
        """ Detect a disc in a frame.
            Arguments:
                image: RGB image in which to detect disc
            Returns:
                tuple (x, y, radius)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 2)

        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 2, 100, param1=200, param2=100)
        if circles is None:
            return []
        circles = np.uint16(np.around(circles))

        print("Num circles detected:", len(circles[0]))
        return circles[0]

    def calculate_position(self, x, y, r):
        """ Calculate disc's Z position in world coordinates
            Arguments:
                r: radius of ball in image
            Returns:
                X, Y, Z position of ball in world coordinates
        """
        Z = self.focal * self.R / r

        X = (x - self.centerx) * Z / self.focal
        Y = (y - self.centery) * Z / self.focal

        return X, Y, Z

    def draw_circles(self, frame, x, y, r, Z):
        """ Draw circle on ball and write depth estimate  in center
            Arguments:
                image: image on which to draw
                x,y,r: 2D position and radius of ball
                Z: estimated depth of ball
        """
        cv2.circle( frame, (int(x),int(y)), int(r), (0,0,255),2)
        cv2.putText( frame, str(int(Z)) + ' cm', (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

