import cv2
import numpy as np
from time import sleep
import src.functions as functions
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


POSE_DETECTION_MODEL_PATH = r'C:\Users\pfnas\OneDrive\Documents\2Sophomore\23Fall\Computer Vision\disc-metrics\models\pose_landmarker_lite.task'
model_path = os.path.abspath(POSE_DETECTION_MODEL_PATH)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")


class PoseTracker:

    def __init__(self, videoFrames, pixelToRealRatio, fps):
        self.frames = videoFrames[:]
        self.pixelToRealRatio = pixelToRealRatio
        self.fps = fps
        self.frameShape = self.frames[0].shape
        self.modelPath = POSE_DETECTION_MODEL_PATH

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
            print(f"Iteration: {i}")
            keypointedFrames.append(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        return keypointedFrames

            
