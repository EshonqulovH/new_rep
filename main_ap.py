import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import time

# MediaPipe Pose modelini ishga tushirish
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    st.set_page_config(page_title="Skelet Aniqlash", page_icon="🎥", layout="wide")
    st.title("Real Vaqt Skelet Aniqlash")
    
    st.warning("""
    Eslatma: 
    1. Kamerangiz ruxsatini berishingiz kerak
    2. Agar kamera ishlamasa, brauzeringizni yangilang
    3. Chrome/Edge ishlatish tavsiya etiladi
    """)
    
    st.info("'Kamerani ochish' tugmasini bosing va ruxsat bering")
    
    # Kamerani ishga tushirish uchun tugma
    start_camera = st.button("Kamerani ochish")
    
    if start_camera:
        # Streamda ko'rsatish uchun joy
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # To'xtatish tugmasi
        stop_button_placeholder = st.empty()
        
        try:
            # OpenCV kamerani ochish
            cap = cv2.VideoCapture(0)
            
            # Kamera ochilganini tekshirish
            if not cap.isOpened():
                st.error("Kamera ochilmadi. Qurilma ulanganini tekshiring.")
                return
                
            status_placeholder.success("Kamera muvaffaqiyatli ishga tushdi!")
            stop_button = stop_button_placeholder.button("To'xtatish")
            
            # MediaPipe Pose modelini ishga tushirish
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Eng yengil model
                smooth_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            ) as pose:
                previous_landmarks = None
                
                # Kuzatiladigan tana qismlari
                body_parts = {
                    "Bosh": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "Tana": [11, 12, 13, 14, 23, 24],
                    "Chap qo'l": [11, 13, 15, 17, 19, 21],
                    "O'ng qo'l": [12, 14, 16, 18, 20, 22],
                    "Chap oyoq": [23, 25, 27, 29, 31],
                    "O'ng oyoq": [24, 26, 28, 30, 32]
                }
                
                while cap.isOpened() and not stop_button:
                    # Kadrni o'qish
                    success, frame = cap.read()
                    
                    if not success:
                        status_placeholder.error("Kameradan kadr o'qib bo'lmadi!")
                        break
                    
                    # Tasvir hajmini kamaytirish
                    frame = cv2.resize(frame, (640, 480))
                    
                    # RGB ga o'zgartirish (MediaPipe uchun)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Skeletni aniqlash
                    results = pose.process(frame_rgb)
                    
                    moving_parts = []
                    
                    if results.pose_landmarks:
                        # Skeletni chizish
                        mp_drawing.draw_landmarks(
                            frame_rgb, 
                            results.pose_landmarks, 
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                        )
                        
                        # Harakatni aniqlash
                        current_landmarks = results.pose_landmarks.landmark
                        
                        if previous_landmarks:
                            # Har bir tana qismi uchun harakatni tekshirish
                            for part_name, indexes in body_parts.items():
                                movement_detected = False
                                total_diff = 0
                                valid_points = 0
                                
                                for idx in indexes:
                                    if idx < len(current_landmarks) and idx < len(previous_landmarks):
                                        prev = previous_landmarks[idx]
                                        curr = current_landmarks[idx]
                                        
                                        # Nuqtalar orasidagi masofani hisoblash
                                        diff = np.sqrt(
                                            (curr.x - prev.x) ** 2 + 
                                            (curr.y - prev.y) ** 2 + 
                                            (curr.z - prev.z) ** 2
                                        )
                                        
                                        if diff > 0.01:  # Minimal harakat chegarasi
                                            movement_detected = True
                                            total_diff += diff
                                            valid_points += 1
                                
                                # Agar harakat sezilarli bo'lsa, qo'shish
                                if movement_detected and valid_points > 0 and total_diff/valid_points > 0.015:
                                    moving_parts.append(part_name)
                        
                        # Oldingi kadr ma'lumotlarini saqlash
                        previous_landmarks = current_landmarks
                    
                    # Aniqlangan harakatlarni ko'rsatish
                    detected_movement = ", ".join(moving_parts) if moving_parts else "Harakat aniqlanmadi"
                    
                    # Harakatlarni kadrga yozish
                    cv2.putText(
                        frame_rgb, 
                        f"Harakat: {detected_movement}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Streamlit-ga ko'rsatish
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # To'xtatish tugmasini yangilash
                    stop_button = stop_button_placeholder.button("To'xtatish", key=str(time.time()))
                    
                    # Kadrlar oralig'ida kutish
                    time.sleep(0.03)  # ~30fps
                
                # Kamerani yopish
                cap.release()
                status_placeholder.info("Kamera to'xtatildi")
                stop_button_placeholder.empty()
                
        except Exception as e:
            st.error(f"Xatolik yuz berdi: {str(e)}")
            st.info("Iltimos, brauzeringizni yangilang yoki boshqa brauzer ishlatib ko'ring")

if __name__ == "__main__":
    main()