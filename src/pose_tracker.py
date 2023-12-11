import cv2
import numpy as np
from time import sleep
import src.functions as functions
import os
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


POSE_DETECTION_MODEL_PATH = r'.\models\pose_landmarker_lite.task'
model_path = os.path.abspath(POSE_DETECTION_MODEL_PATH)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")


class PoseTracker:

    def __init__(self, videoFrames, pixelToRealRatio, fps, frameIndex):
        self.frames = videoFrames[:]
        self.pixelToRealRatio = pixelToRealRatio
        self.fps = fps
        self.frameShape = self.frames[0].shape
        self.modelPath = POSE_DETECTION_MODEL_PATH
        self.frameIndex = frameIndex

    def draw_landmarks_on_original_image(self, original_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(original_image)

        # Trimmed pixels
        trim = 60

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Scale the landmarks back to the original image size
            scaled_landmarks = []
            for landmark in pose_landmarks:
                x_scaled = int(landmark.x * 960)
                y_scaled = int(landmark.y * 960) + trim
                scaled_landmarks.append((x_scaled, y_scaled))

            # Draw the scaled pose landmarks on the original image
            for landmark in scaled_landmarks:
                cv2.circle(annotated_image, landmark, 5, (0, 255, 0), -1)

        return annotated_image

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
                )
        return annotated_image

    def findKeypoints(self):
        keypointedFrames = []
        landmarkedPoses = []
        rgbFrames = np.array(self.frames)[:, :, :, ::-1]
        for i, frame in enumerate(rgbFrames):
            trimmed = frame[60:-60,:,:]
            resized = cv2.resize(trimmed, (256, 256))
            # cv2.imshow("resized", resized)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)

            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.modelPath),
                running_mode=VisionRunningMode.IMAGE
                )

            with PoseLandmarker.create_from_options(options) as landmarker:
                pose_landmarker_result = landmarker.detect(image)

            #Process the detection result. In this case, visualize it.
            # cv2.imshow("frame", cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR))
            # if cv2.waitKey(0) == ord('q'): break
            annotated_image = self.draw_landmarks_on_original_image(frame, pose_landmarker_result)
            # annotated_image = self.draw_landmarks_on_image(image.numpy_view(), pose_landmarker_result)
            # cv2.imshow("frame", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # if cv2.waitKey(1) == ord('q'): break
            print(f"Analyzing frame: {i+1} of {len(rgbFrames)}")

            landmarkedPoses.append(pose_landmarker_result)
            keypointedFrames.append(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        return landmarkedPoses, keypointedFrames

    def createWireFrame( self, landmarkedPoses, frameIndex, ax=plt.axes(projection='3d') ):
        if type(frameIndex) is not int or frameIndex < 0 or frameIndex >= len(landmarkedPoses):
            print(f"Invalid frameIndex: {frameIndex}")
            return None

        ax.cla()

        landmarks = (landmarkedPoses[frameIndex]).pose_landmarks[0]

        headToLeftShoulder = [(landmarks[0].x, landmarks[12].x), (landmarks[0].y, landmarks[12].y), (landmarks[0].z, landmarks[12].z)]
        headToRightShoulder = [(landmarks[0].x, landmarks[11].x), (landmarks[0].y, landmarks[11].y), (landmarks[0].z, landmarks[11].z)]
        betweenShoulders = [(landmarks[11].x, landmarks[12].x), (landmarks[11].y, landmarks[12].y), (landmarks[11].z, landmarks[12].z)]
        rightShoulderToElbow = [(landmarks[12].x, landmarks[14].x), (landmarks[12].y, landmarks[14].y), (landmarks[12].z, landmarks[14].z)]
        rightElbowToPalm = [(landmarks[14].x, landmarks[16].x), (landmarks[14].y, landmarks[16].y), (landmarks[14].z, landmarks[16].z)]
        leftShoulderToElbow = [(landmarks[11].x, landmarks[13].x), (landmarks[11].y, landmarks[13].y), (landmarks[11].z, landmarks[13].z)]
        leftElbowToPalm = [(landmarks[13].x, landmarks[15].x), (landmarks[13].y, landmarks[15].y), (landmarks[13].z, landmarks[15].z)]
        rightShoulderToHip = [(landmarks[24].x, landmarks[12].x), (landmarks[24].y, landmarks[12].y), (landmarks[24].z, landmarks[12].z)]
        leftShoulderToHip = [(landmarks[11].x, landmarks[23].x), (landmarks[11].y, landmarks[23].y), (landmarks[11].z, landmarks[23].z)]
        betweenHips = [(landmarks[24].x, landmarks[23].x), (landmarks[24].y, landmarks[23].y), (landmarks[24].z, landmarks[23].z)]
        rightHipToKnee = [(landmarks[24].x, landmarks[26].x), (landmarks[24].y, landmarks[26].y), (landmarks[24].z, landmarks[26].z)]
        rightKneeToAnkle = [(landmarks[26].x, landmarks[28].x), (landmarks[26].y, landmarks[28].y), (landmarks[26].z, landmarks[28].z)]
        leftHipToKnee = [(landmarks[23].x, landmarks[25].x), (landmarks[23].y, landmarks[25].y), (landmarks[23].z, landmarks[25].z)]
        leftKneeToAnkle = [(landmarks[25].x, landmarks[27].x), (landmarks[25].y, landmarks[27].y), (landmarks[25].z, landmarks[27].z)]

        lines = [headToLeftShoulder, headToRightShoulder, betweenShoulders, rightShoulderToElbow, rightElbowToPalm, leftShoulderToElbow, leftElbowToPalm, rightShoulderToHip, leftShoulderToHip, betweenHips, rightHipToKnee, rightKneeToAnkle, leftHipToKnee, leftKneeToAnkle]

        for line in lines:
            ax.plot3D(line[0], line[1], line[2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.show()
        plt.show()

    def getReleaseFrame(self, speed, point):
        x0 = self.frames.shape[2] // 2
        t0 = self.frameIndex
        x1 = point.x
        t2 = abs(x1 - x0) / speed + t0
        return t2
