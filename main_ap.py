import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tempfile
import os
from PIL import Image
import time

# MediaPipe Pose modelini chaqirish
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# MediaPipe model keshini yozish mumkin bo'lgan joyga o'rnatish
temp_dir = tempfile.gettempdir()
os.environ["MEDIAPIPE_MODEL_PATH"] = temp_dir

def main():
    st.title("Skelet Harakatlarini Aniqlash")
    
    st.warning("""
    Eslatma: 
    1. Kamerangiz ruxsatini berishingiz kerak
    2. Agar ilovada xatolik yuz bersa, brauzeringizni yangilang
    3. Chrome yoki Edge brauzerini ishlatish tavsiya etiladi
    """)
    
    # Streamlit yon panel sozlamalari
    st.sidebar.header("Sozlamalar")
    
    # Variantni tanlash
    option = st.sidebar.radio(
        "Kamera ochish usulini tanlang:",
        ["Variant 1: Oddiy kamera", "Variant 2: Rasmlar yuklash"]
    )
    
    if "Variant 1" in option:
        use_webcam_simple()
    else:
        use_image_upload()

def use_webcam_simple():
    """Oddiy Streamlit kamera usuli"""
    st.subheader("Kamera orqali skelet aniqlash")
    
    # Kamerani boshlash tugmasi
    if st.button("Kamerani ochish"):
        # Veb-kamera oqimini boshlash
        cap = cv2.VideoCapture(0)
        
        # Kamera ochilganini tekshirish
        if not cap.isOpened():
            st.error("Kamerani ochib bo'lmadi. Kamera qurilmasini tekshiring.")
            return
        
        # Videoni ko'rsatish uchun joy
        stframe = st.empty()
        movement_text = st.empty()
        
        # MediaPipe Pose modelini ishga tushirish
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Eng yengil model
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as pose:
            previous_landmarks = None
            
            # Body parts dictionary
            body_parts = {
                "Bosh": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "Tana": [11, 12, 13, 14, 23, 24],
                "Chap qo'l": [11, 13, 15, 17, 19, 21],
                "O'ng qo'l": [12, 14, 16, 18, 20, 22],
                "Chap oyoq": [23, 25, 27, 29, 31],
                "O'ng oyoq": [24, 26, 28, 30, 32]
            }
            
            # To'xtatish tugmasi
            stop_button_placeholder = st.empty()
            stop = stop_button_placeholder.button("To'xtatish")
            
            while not stop:
                ret, frame = cap.read()
                if not ret:
                    st.error("Kadrni o'qib bo'lmadi")
                    break
                
                # Rasmni kichiklashtirish
                frame = cv2.resize(frame, (640, 480))
                
                # Rangni RGB ga o'zgartirish
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Pozani aniqlash
                results = pose.process(rgb_frame)
                
                moving_parts = []
                
                if results.pose_landmarks:
                    # Skeletni chizish
                    mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    
                    # Harakatni aniqlash
                    current_landmarks = results.pose_landmarks.landmark
                    
                    if previous_landmarks is not None:
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
                cv2.putText(rgb_frame, f"Harakat: {detected_movement}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Kadrni ko'rsatish
                stframe.image(rgb_frame, caption='Skelet aniqlash', channels="RGB")
                movement_text.text(f"Aniqlangan harakatlar: {detected_movement}")
                
                # To'xtatish tugmasini yangilash
                stop = stop_button_placeholder.button("To'xtatish", key=f"stop_{time.time()}")
                
                # Kadrlar oralig'ida kichik kutish
                time.sleep(0.1)
            
            # Kamerani yopish
            cap.release()
        
        st.success("Kamera yopildi")

def use_image_upload():
    """Rasmlar yuklash orqali skelet aniqlash"""
    st.subheader("Rasmlar yuklash orqali skelet aniqlash")
    
    # Rasmlarni yuklash
    uploaded_file = st.file_uploader("Rasm yuklang", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Rasmni o'qish
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Rasmni kichiklashtirish
        image_np = cv2.resize(image_np, (640, 480))
        
        # MediaPipe Pose modelini ishga tushirish
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5
        ) as pose:
            # RGB rangga o'zgartirish
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Pozani aniqlash
            results = pose.process(rgb_image)
            
            # Agar skelet aniqlansa
            if results.pose_landmarks:
                # Skeletni chizish
                mp_drawing.draw_landmarks(rgb_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Natijani ko'rsatish
                st.image(rgb_image, caption='Skelet aniqlash', channels="RGB")
                
                # Marker nuqtalar haqida ma'lumot
                st.subheader("Skelet nuqtalari")
                
                # Marker nuqtalar ro'yxati
                body_parts = {
                    "Bosh": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "Tana": [11, 12, 13, 14, 23, 24],
                    "Chap qo'l": [11, 13, 15, 17, 19, 21],
                    "O'ng qo'l": [12, 14, 16, 18, 20, 22],
                    "Chap oyoq": [23, 25, 27, 29, 31],
                    "O'ng oyoq": [24, 26, 28, 30, 32]
                }
                
                for part_name, indexes in body_parts.items():
                    st.write(f"**{part_name}:** {len(indexes)} nuqta aniqlandi")
            else:
                st.warning("Rasmda skelet aniqlanmadi. Boshqa rasm bilan urinib ko'ring.")
    else:
        st.info("Iltimos, rasm yuklang")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Skelet Aniqlash",
        page_icon="🎥",
        layout="wide"
    )
    main()