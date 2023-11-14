import cv2 as cv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('video', help='path to input video file')
parser.add_argument('--output', help='path to output video file (optional)')
parser.add_argument('--calibration', default='iphone_calib.txt', help='path to calibration file')
parser.add_argument('--ball_radius', type=float, default=27.305, help='radius of disc in cm')
args = parser.parse_args()

vid = cv.VideoCapture(args.video)
