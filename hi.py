#!/usr/bin/env python3
"""
pose_corrector.py — Real-time Pose Corrector using MediaPipe and OpenCV
"""

import math
import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import os
import warnings

# Silence absl and mediapipe warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# --------- CONFIG ----------
BACK_STRAIGHT_MIN = 160
BACK_STRAIGHT_MAX = 180
KNEE_ALIGNMENT_MIN = 165
KNEE_ALIGNMENT_MAX = 180
ELBOW_EXTENDED_MIN = 160
ELBOW_EXTENDED_MAX = 180
NECK_TILT_MAX = 20
SMOOTHING_WINDOW = 5
# ----------------------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
L = mp_pose.PoseLandmark


def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cosang = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def smooth_deque(dq, value):
    dq.append(value)
    if len(dq) > SMOOTHING_WINDOW:
        dq.popleft()
    return float(np.median(np.array(dq)))


def check_back_straight(shoulder, hip, ankle): return angle_between(shoulder, hip, ankle)
def check_knee_alignment(hip, knee, ankle): return angle_between(hip, knee, ankle)
def check_elbow_extension(shoulder, elbow, wrist): return angle_between(shoulder, elbow, wrist)


def neck_tilt_angle(shoulder, ear):
    s, e = np.array(shoulder), np.array(ear)
    v = s - e
    vert = np.array([0.0, 1.0])
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return 0.0
    cosang = np.clip(np.dot(v / norm_v, vert), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def main():
    print("🟢 Pose Corrector starting...")
    cap = cv2.VideoCapture(0)

    # Force camera open check
    if not cap.isOpened():
        print("❌ Could not open webcam. Try closing other apps using it.")
        return
    else:
        print("✅ Webcam opened successfully. Press 'Q' or 'Esc' to quit.")

    # Warm up camera for a moment
    time.sleep(1.0)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        hip_angle_buf, knee_left_buf, knee_right_buf = deque(), deque(), deque()
        elbow_left_buf, elbow_right_buf, neck_buf = deque(), deque(), deque()

        fps_time = time.time()
        cv2.namedWindow("Pose Corrector", cv2.WINDOW_NORMAL)  # Force window creation

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not received. Retrying...")
                continue

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            feedbacks = []

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                def p(idx): return (lm[idx].x * w, lm[idx].y * h)

                left_sh, right_sh = p(L.LEFT_SHOULDER), p(L.RIGHT_SHOULDER)
                left_hip, right_hip = p(L.LEFT_HIP), p(L.RIGHT_HIP)
                left_ankle, right_ankle = p(L.LEFT_ANKLE), p(L.RIGHT_ANKLE)
                left_knee, right_knee = p(L.LEFT_KNEE), p(L.RIGHT_KNEE)
                left_el, right_el = p(L.LEFT_ELBOW), p(L.RIGHT_ELBOW)
                left_w, right_w = p(L.LEFT_WRIST), p(L.RIGHT_WRIST)
                left_ear, right_ear = p(L.LEFT_EAR), p(L.RIGHT_EAR)

                mid_sh = ((left_sh[0] + right_sh[0]) / 2, (left_sh[1] + right_sh[1]) / 2)
                mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                mid_ankle = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)

                hip_angle_s = smooth_deque(hip_angle_buf, check_back_straight(mid_sh, mid_hip, mid_ankle))
                left_knee_s = smooth_deque(knee_left_buf, check_knee_alignment(left_hip, left_knee, left_ankle))
                right_knee_s = smooth_deque(knee_right_buf, check_knee_alignment(right_hip, right_knee, right_ankle))
                left_elbow_s = smooth_deque(elbow_left_buf, check_elbow_extension(left_sh, left_el, left_w))
                right_elbow_s = smooth_deque(elbow_right_buf, check_elbow_extension(right_sh, right_el, right_w))

                neck_angle = 0.0
                if lm[L.LEFT_EAR].visibility > 0.2:
                    neck_angle = neck_tilt_angle(left_sh, left_ear)
                elif lm[L.RIGHT_EAR].visibility > 0.2:
                    neck_angle = neck_tilt_angle(right_sh, right_ear)
                neck_s = smooth_deque(neck_buf, neck_angle)

                if hip_angle_s < BACK_STRAIGHT_MIN:
                    feedbacks.append(f"Back bent ({int(hip_angle_s)}°)")
                if left_knee_s < KNEE_ALIGNMENT_MIN:
                    feedbacks.append(f"Left knee bent ({int(left_knee_s)}°)")
                if right_knee_s < KNEE_ALIGNMENT_MIN:
                    feedbacks.append(f"Right knee bent ({int(right_knee_s)}°)")
                if left_elbow_s < ELBOW_EXTENDED_MIN:
                    feedbacks.append(f"L elbow bent ({int(left_elbow_s)}°)")
                if right_elbow_s < ELBOW_EXTENDED_MIN:
                    feedbacks.append(f"R elbow bent ({int(right_elbow_s)}°)")
                if neck_s > NECK_TILT_MAX:
                    feedbacks.append(f"Neck tilt {int(neck_s)}°")

                if not feedbacks:
                    feedbacks.append("✅ Posture OK!")

                mp_drawing.draw_landmarks(
                    frame_out, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                for i, msg in enumerate(feedbacks):
                    cv2.putText(frame_out, msg, (10, 40 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame_out, "No pose detected.", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            cv2.putText(frame_out, f"FPS: {int(fps)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Pose Corrector", frame_out)

            # Force refresh window
            cv2.waitKey(1)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                print("👋 Exiting Pose Corrector.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
