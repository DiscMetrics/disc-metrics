import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class WireframeAnimator:

    def __init__(self, videoFrames, fps, landmarkedPoses):
        self.frames = videoFrames[:]
        self.fps = fps
        self.landmarkedPoses = landmarkedPoses
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
    
    def initialize_animation(self):
        return self.ax

    def update(self, frameIndex):
        self.ax.cla()
        self.ax.grid(False)
        self.ax.set_axis_off()
        # self.ax.set_xlabel('X')
        # self.ax.set_ylabel('Y')
        # self.ax.set_zlabel('Z')
        self.ax.set_xlim(-.5, .5)
        self.ax.set_ylim(-.5, .5)
        self.ax.set_zlim(-.5, .5)
        # print("frame:", frameIndex + 1, "/ total frames:", len(self.landmarkedPoses))

        try:
            landmarks = self.landmarkedPoses[frameIndex].pose_landmarks[0]
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
                self.ax.plot3D(line[0], line[1], line[2])
        except IndexError:
            pass

        return self.ax, 

    def animateWireframe(self):
        # print("len(self.frames):", len(self.frames))
        self.num_frames = len(self.frames)

        self.animation = FuncAnimation(self.fig, self.update, frames=self.num_frames, init_func=self.initialize_animation, repeat=False)

        plt.show()