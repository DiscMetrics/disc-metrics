import cv2
import numpy as np
import argparse
from src import disc_tracker, perspective_calculator, functions, pose_tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='path to input video file')
    parser.add_argument('--no-video', action='store_true', help='do not view popup video')
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

    # should move to perspective_calculator.py
    ratios = []
    for frame in frames[:args.fps//2]:  # first half second
        perspective = perspective_calculator.PerspectiveCalculator(radius)
        ratio = perspective.process_frame(frame)
        if ratio is not None:
            ratios.append(ratio)
        if not args.no_video:
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) == ord('q'):
                cv2.destroyWindow('frame')
                break

    ratio = np.mean(functions.remove_outliers(ratios))


    lastHalf = frames[len(frames)//2:]
    leftHalf = [frame[:, :frame.shape[1]//2] for frame in lastHalf]
    discTracker = disc_tracker.DiscTracker(leftHalf, ratio, args.fps, args.no_video)
    background = discTracker.findBackground()
    discs = discTracker.findDisc(background)
    realSpeed, angle, pixelSpeed = discTracker.findDiscSpeedAngle(discs)
    frameIndex = discTracker.getFirstFrameIndex()
    poseAnalysisFrames = frames[frameIndex-2*args.fps:frameIndex+1*args.fps]
    TrimmedFrameIndex = 2 * args.fps
    rightHalf = [frame[:, frame.shape[1]//2:] for frame in poseAnalysisFrames]
    PoseTracker = pose_tracker.PoseTracker(rightHalf, ratio, args.fps, TrimmedFrameIndex)
    # PoseTracker.getReleaseFrame(TrimmedFrameIndex, pixelSpeed, pos)
    print(f"Speed = {realSpeed} m/s, {realSpeed * 2.23694} mph")
    print(f"Angle = {angle} radians, {angle * 180 / np.pi} degrees")


if __name__ == '__main__':
    main()
