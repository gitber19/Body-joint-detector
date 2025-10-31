#!/usr/bin/env python3
"""
bicep_curl_counter.py — Detects and counts bicep curls from a side view using MediaPipe and OpenCV
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
UP_THRESHOLD = 150     # Elbow extended (down position)
DOWN_THRESHOLD = 40    # Elbow fully curled (up position)
ARM_SIDE = "right"     # "right" or "left" arm
# ----------------------------

def angle(a, b, c):
    """Compute the angle (in degrees) formed at point b by points a-b-c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0: 
        return 0
    cosang = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def smooth(dq, val):
    dq.append(val)
    if len(dq) > ANGLE_SMOOTH:
        dq.popleft()
    return np.mean(dq)

def main():
    print("💪 Bicep Curl Counter starting...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    time.sleep(1.0)
    cv2.namedWindow("Bicep Curl Counter", cv2.WINDOW_NORMAL)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        elbow_buf = deque()
        curl_count = 0
        position = "down"

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

                if ARM_SIDE == "right":
                    shoulder, elbow, wrist = p(L.RIGHT_SHOULDER), p(L.RIGHT_ELBOW), p(L.RIGHT_WRIST)
                else:
                    shoulder, elbow, wrist = p(L.LEFT_SHOULDER), p(L.LEFT_ELBOW), p(L.LEFT_WRIST)

                # Compute elbow angle
                elbow_angle = smooth(elbow_buf, angle(shoulder, elbow, wrist))

                # Count logic
                if elbow_angle > UP_THRESHOLD and position == "up":
                    position = "down"
                elif elbow_angle < DOWN_THRESHOLD and position == "down":
                    curl_count += 1
                    position = "up"

                # Feedback
                if elbow_angle > UP_THRESHOLD:
                    feedback = "Curl Up!"
                elif elbow_angle < DOWN_THRESHOLD:
                    feedback = "Go Lower!"
                else:
                    feedback = "Good Form!"

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )

                # Display angle
                cv2.putText(output, f"Angle: {int(elbow_angle)}°", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Display count
                cv2.putText(output, f"Curls: {curl_count}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Display feedback (in red)
                cv2.putText(output, feedback, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            else:
                cv2.putText(output, "No pose detected.", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Bicep Curl Counter", output)

            key = cv2.waitKey(10) & 0xFF
            if key in [27, ord('q')]:
                print(f"👋 Total Curls: {curl_count}")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
