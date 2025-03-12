import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import time

# MediaPipe Pose modeli
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit app
st.set_page_config(page_title="Skelet Aniqlash", page_icon="🎥", layout="wide")
st.title("Real Vaqt Skelet Aniqlash")

st.warning("""
Eslatma: 
1. Kamerangiz ruxsatini berishingiz kerak
2. Agar kamera ishlamasa, brauzeringizni yangilang
3. Chrome/Edge ishlatish tavsiya etiladi
""")

st.info("'Kamerani ochish' tugmasini bosing va ruxsat bering")

# Camera stream placeholder
video_placeholder = st.empty()
status_placeholder = st.empty()

# Body parts dictionary
body_parts = {
    "Bosh": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Tana": [11, 12, 13, 14, 23, 24],
    "Chap qo'l": [11, 13, 15, 17, 19, 21],
    "O'ng qo'l": [12, 14, 16, 18, 20, 22],
    "Chap oyoq": [23, 25, 27, 29, 31],
    "O'ng oyoq": [24, 26, 28, 30, 32]
}

# Start camera button
start_button = st.button("Kamerani ochish")

if start_button:
    # Initialize camera
    try:
        # Try different camera index options (0, 1) and backends
        camera_options = [
            (0, cv2.CAP_ANY),         # Default camera with default backend
            (0, cv2.CAP_V4L2),        # Default camera with V4L2 backend (Linux)
            (0, cv2.CAP_DSHOW),       # Default camera with DirectShow (Windows)
            (1, cv2.CAP_ANY)          # Secondary camera if available
        ]
        
        cap = None
        for cam_idx, backend in camera_options:
            status_placeholder.info(f"Kamera ochilmoqda... (index:{cam_idx}, backend:{backend})")
            cap = cv2.VideoCapture(cam_idx, backend)
            if cap.isOpened():
                status_placeholder.success(f"Kamera muvaffaqiyatli ochildi! (index:{cam_idx})")
                break
            cap.release()  # Release if not successful
        
        if not cap or not cap.isOpened():
            status_placeholder.error("Kamera ochilmadi. Qurilma ulanganini tekshiring.")
            st.stop()
            
        # Create MediaPipe Pose instance
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as pose:
            previous_landmarks = None
            stop_button_pressed = False
            
            # Stop button
            stop_button_col = st.empty()
            
            while not stop_button_pressed:
                # Check for stop button
                if stop_button_col.button("To'xtatish", key=f"stop_{time.time()}"):
                    stop_button_pressed = True
                    break
                
                # Read frame
                success, frame = cap.read()
                if not success:
                    status_placeholder.error("Kameradan kadr o'qib bo'lmadi!")
                    time.sleep(1)  # Wait a bit before trying again
                    continue
                
                # Horizontal flip for more intuitive display
                frame = cv2.flip(frame, 1)
                
                # Resize frame
                frame = cv2.resize(frame, (640, 480))
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = pose.process(frame_rgb)
                
                # Initialize moving parts list
                moving_parts = []
                
                # If pose detected
                if results.pose_landmarks:
                    # Draw skeleton
                    mp_drawing.draw_landmarks(
                        frame_rgb, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                    )
                    
                    current_landmarks = results.pose_landmarks.landmark
                    
                    # Check for movement if we have previous frame data
                    if previous_landmarks:
                        # Check each body part for movement
                        for part_name, indexes in body_parts.items():
                            movement_detected = False
                            total_diff = 0
                            valid_points = 0
                            
                            for idx in indexes:
                                if idx < len(current_landmarks) and idx < len(previous_landmarks):
                                    prev = previous_landmarks[idx]
                                    curr = current_landmarks[idx]
                                    
                                    # Calculate distance between points
                                    diff = np.sqrt(
                                        (curr.x - prev.x) ** 2 + 
                                        (curr.y - prev.y) ** 2
                                    )
                                    
                                    if diff > 0.01:  # Movement threshold
                                        movement_detected = True
                                        total_diff += diff
                                        valid_points += 1
                            
                            # If significant movement detected
                            if movement_detected and valid_points > 0 and total_diff/valid_points > 0.015:
                                moving_parts.append(part_name)
                    
                    # Save landmarks for next frame
                    previous_landmarks = current_landmarks
                
                # Add movement info to frame
                detected_movement = ", ".join(moving_parts) if moving_parts else "Harakat aniqlanmadi"
                cv2.putText(
                    frame_rgb, 
                    f"Harakat: {detected_movement}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Display the frame in Streamlit
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Short delay to reduce CPU usage
                time.sleep(0.03)
        
        # Release camera when done
        cap.release()
        status_placeholder.info("Kamera to'xtatildi")
        stop_button_col.empty()
        
    except Exception as e:
        status_placeholder.error(f"Xatolik yuz berdi: {str(e)}")
        st.error(f"Dastur xatoligi: {str(e)}")
        if 'cap' in locals() and cap is not None and cap.isOpened():
            cap.release()