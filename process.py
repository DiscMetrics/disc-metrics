import cv2
import numpy as np
import argparse
from src import disc_tracker, perspective_calculator, functions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='path to input video file')
    parser.add_argument('--output', help='path to output video file (optional)')
    parser.add_argument('--calibration', default='data/calib.txt', help='path to calibration file')
    parser.add_argument('--disc_radius', type=float, default=13.6525, help='radius of disc in cm')
    parser.add_argument('--fps', type=int, default=60, help='frames per second of video')
    args = parser.parse_args()

    radius = args.disc_radius / 100

    vid = cv2.VideoCapture(args.video)

    frames = []

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        frames.append(frame)

        # cv2.imshow('frame',frame)
        # if cv2.waitKey(5) == ord('q'):
        #     break

    frames = np.array(frames)
    vid.release()

    ratios = []
    for frame in frames[:args.fps//2]:  # first half second
        perspective = perspective_calculator.PerspectiveCalculator(radius)
        ratios.append(perspective.process_frame(frame))
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) == ord('q'):
            break

    ratio = np.mean(functions.remove_outliers(ratios))
    # print("RATIO:", ratio)


    lastHalf = frames[len(frames)//2:]
    leftHalf = [frame[:, :frame.shape[1]//2] for frame in lastHalf]
    discTracker = disc_tracker.DiscTracker(leftHalf, ratio, args.fps)
    background = discTracker.findBackground()
    discs = discTracker.findDisc(background)
    speed = discTracker.findDiscSpeed(discs)
    print(f"Speed = {speed} m/s, {speed*2.23694} mph")


if __name__ == '__main__':
    main()
