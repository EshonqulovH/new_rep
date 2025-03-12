import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import threading
import time

# MediaPipe Pose modeli
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit holatini saqlash
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

def run_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        st.error("Kamera ochilmadi. Qurilma ulanganini tekshiring.")
        return

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as pose:

            previous_landmarks = None
            video_placeholder = st.empty()
            status_placeholder = st.empty()

            while st.session_state.run_camera:
                success, frame = cap.read()
                if not success:
                    status_placeholder.error("Kameradan kadr o'qib bo'lmadi!")
                    break

                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(frame_rgb)
                moving_parts = []

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                    )

                    current_landmarks = results.pose_landmarks.landmark
                    if previous_landmarks:
                        for idx in range(len(current_landmarks)):
                            if idx < len(previous_landmarks):
                                prev = previous_landmarks[idx]
                                curr = current_landmarks[idx]
                                diff = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2 + (curr.z - prev.z)**2)
                                if diff > 0.015:
                                    moving_parts.append(f"Nuqta {idx}")

                    previous_landmarks = current_landmarks

                detected_movement = ", ".join(moving_parts) if moving_parts else "Harakat aniqlanmadi"
                cv2.putText(frame_rgb, f"Harakat: {detected_movement}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                time.sleep(0.03)

    finally:
        cap.release()
        status_placeholder.info("Kamera to'xtatildi")


# **Streamlit UI**
st.set_page_config(page_title="Skelet Aniqlash", page_icon="🎥", layout="wide")
st.title("Real Vaqt Skelet Aniqlash")

st.warning("""
Eslatma: 
1. Kamerangiz ruxsatini berishingiz kerak
2. Agar kamera ishlamasa, brauzeringizni yangilang
3. Chrome/Edge ishlatish tavsiya etiladi
""")

st.info("'Kamerani ochish' tugmasini bosing va ruxsat bering")

# Kamerani boshlash yoki to'xtatish tugmalari
if not st.session_state.run_camera:
    if st.button("Kamerani ochish"):
        st.session_state.run_camera = True
        threading.Thread(target=run_camera).start()
else:
    if st.button("To'xtatish"):
        st.session_state.run_camera = False
