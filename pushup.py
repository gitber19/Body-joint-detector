#!/usr/bin/env python3
"""
pushup_detector.py — Detects perfect push-ups using MediaPipe and OpenCV
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import warnings
from collections import deque

# Suppress TensorFlow & Mediapipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
L = mp_pose.PoseLandmark

# ---------- CONFIG ----------
ANGLE_SMOOTH = 5
UP_THRESHOLD = 160      # Elbow angle for "up" position
DOWN_THRESHOLD = 90     # Elbow angle for "down" position
BACK_STRAIGHT_MIN = 160 # Minimum back straightness angle
# ----------------------------

def angle(a, b, c):
    """Compute angle between three points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0: return 0
    cosang = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def smooth(dq, val):
    dq.append(val)
    if len(dq) > ANGLE_SMOOTH:
        dq.popleft()
    return np.mean(dq)

def main():
    print("💪 Push-Up Detector starting...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    time.sleep(1.0)
    cv2.namedWindow("Push-Up Detector", cv2.WINDOW_NORMAL)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        left_elbow_buf, right_elbow_buf, back_angle_buf = deque(), deque(), deque()
        pushup_count = 0
        position = "up"

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not received. Retrying...")
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            output = frame.copy()
            feedback = ""

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                def p(idx): return (lm[idx].x * w, lm[idx].y * h)

                left_sh, right_sh = p(L.LEFT_SHOULDER), p(L.RIGHT_SHOULDER)
                left_el, right_el = p(L.LEFT_ELBOW), p(L.RIGHT_ELBOW)
                left_w, right_w = p(L.LEFT_WRIST), p(L.RIGHT_WRIST)
                left_hip, right_hip = p(L.LEFT_HIP), p(L.RIGHT_HIP)
                left_ankle, right_ankle = p(L.LEFT_ANKLE), p(L.RIGHT_ANKLE)

                mid_shoulder = ((left_sh[0] + right_sh[0]) / 2, (left_sh[1] + right_sh[1]) / 2)
                mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                mid_ankle = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)

                # Angles
                left_elbow_angle = smooth(left_elbow_buf, angle(left_sh, left_el, left_w))
                right_elbow_angle = smooth(right_elbow_buf, angle(right_sh, right_el, right_w))
                back_angle = smooth(back_angle_buf, angle(mid_shoulder, mid_hip, mid_ankle))

                # Average both sides for robustness
                elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

                # Push-up form check
                if back_angle < BACK_STRAIGHT_MIN:
                    feedback = "Keep your body straight!"
                elif elbow_angle > UP_THRESHOLD and position == "down":
                    pushup_count += 1
                    position = "up"
                elif elbow_angle < DOWN_THRESHOLD and position == "up":
                    position = "down"
                elif elbow_angle > UP_THRESHOLD:
                    feedback = "Go Lower!"
                elif elbow_angle < DOWN_THRESHOLD:
                    feedback = "Push Up!"

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )

                # Show count
                cv2.putText(output, f"Push-Ups: {pushup_count}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Feedback in red
                if feedback:
                    cv2.putText(output, feedback, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            else:
                cv2.putText(output, "No pose detected.", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Push-Up Detector", output)

            key = cv2.waitKey(10) & 0xFF
            if key in [27, ord('q')]:
                print(f"👋 Total Push-Ups: {pushup_count}")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
