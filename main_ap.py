import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image
import io

# Sahifa konfiguratsiyasi
st.set_page_config(page_title="Tana Harakatlarini Aniqlash", page_icon="🏃‍♂️", layout="wide")
st.title("Tana Harakatlarini Aniqlash Ilovasi")

# MediaPipe pose modelini o'rnatish
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Harakat aniqlash chegarasi
threshold = st.sidebar.slider("Harakat sezgirligi", min_value=0.01, max_value=0.1, value=0.03, step=0.01)

# Asosiy tana qismlari indekslari
body_parts = {
    0: "Bosh",           # Bosh (Nose)
    11: "Tana",          # Tana (left_shoulder)
    12: "Tana",          # Tana (right_shoulder)
    13: "Chap qo'l",     # Chap qo'l (left_elbow)
    14: "O'ng qo'l",     # O'ng qo'l (right_elbow)
    15: "Chap qo'l",     # Chap qo'l (left_wrist)
    16: "O'ng qo'l",     # O'ng qo'l (right_wrist)
    23: "Chap oyoq",     # Chap oyoq (left_knee)
    24: "O'ng oyoq",     # O'ng oyoq (right_knee)
    25: "Chap oyoq",     # Chap oyoq (left_ankle)
    26: "O'ng oyoq",     # O'ng oyoq (right_ankle)
}

# Reset interval
reset_interval = st.sidebar.slider("Harakatlanish belgilash vaqti (soniya)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

# Holatlar kolonkalarini yaratish
col1, col2 = st.columns([3, 1])

# Streamlit o'ng panelda tana qismlarini ko'rsatamiz
with col2:
    st.header("Tana holati")
    status_placeholders = {}
    for part in ["Bosh", "Tana", "Chap qo'l", "O'ng qo'l", "Chap oyoq", "O'ng oyoq"]:
        status_placeholders[part] = st.empty()

# Qaysi tana qismlari harakatlanganligi haqida ma'lumot
moving_parts = {
    "Bosh": False,
    "Tana": False,
    "Chap qo'l": False,
    "O'ng qo'l": False,
    "Chap oyoq": False,
    "O'ng oyoq": False
}

# Reset moving status after some time
last_detection_time = {
    "Bosh": 0,
    "Tana": 0,
    "Chap qo'l": 0,
    "O'ng qo'l": 0,
    "Chap oyoq": 0,
    "O'ng oyoq": 0
}

# Video frameni o'qish va qayta ishlash
def process_frame(image, prev_landmarks):
    # RGB formatiga o'tkazish (MediaPipe RGB formatida ishlaydi)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Pose model orqali harakatlarni aniqlash
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # O'zgartirilgan rasmni yaratish
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Harakatlarni aniqlash
        current_time = time.time()
        
        if prev_landmarks:
            current_landmarks = results.pose_landmarks.landmark
            
            # Har bir muhim tana qismi uchun tekshirish
            for idx, part_name in body_parts.items():
                # Agar bu tana qismi avval harakatlanayotgan deb belgilangan bo'lsa va vaqt o'tgan bo'lsa
                if moving_parts[part_name] and (current_time - last_detection_time[part_name]) > reset_interval:
                    moving_parts[part_name] = False
                
                prev = prev_landmarks[idx]
                curr = current_landmarks[idx]
                
                dx = abs(curr.x - prev.x)
                dy = abs(curr.y - prev.y)
                
                # Harakat aniqlangan bo'lsa
                if dx > threshold or dy > threshold:
                    # Agar bu tana qismi hali harakatlanayotgan deb belgilanmagan bo'lsa
                    if not moving_parts[part_name]:
                        st.write(f"{part_name} harakatlandi!")
                        moving_parts[part_name] = True
                        last_detection_time[part_name] = current_time
        
        # O'ng panelda holatlarni yangilash
        for part, is_moving in moving_parts.items():
            status = "Harakatlanmoqda 🟢" if is_moving else "Harakatsiz 🔴"
            status_placeholders[part].write(f"{part}: {status}")
        
        # Joriy holatni qaytarish
        return annotated_image, results.pose_landmarks.landmark
    
    return image, prev_landmarks

def main():
    # Video oqimini boshlash
    start_button = st.button("Kamerani yoqish")
    stop_button = st.button("To'xtatish")
    
    stframe = col1.empty()
    
    # Kamerani yoqish
    if start_button or ('camera_on' in st.session_state and st.session_state['camera_on']):
        st.session_state['camera_on'] = True
        
        if stop_button:
            st.session_state['camera_on'] = False
            return
        
        cap = cv2.VideoCapture(0)
        prev_landmarks = None
        
        if cap.isOpened():
            st.write("Kamera muvaffaqiyatli ishga tushdi!")
            
            while st.session_state['camera_on']:
                success, image = cap.read()
                if not success:
                    st.error("Kameradan videoni olishda xatolik yuz berdi.")
                    break
                
                # Frameni qayta ishlash
                processed_image, prev_landmarks = process_frame(image, prev_landmarks)
                
                # Frameni ko'rsatish
                stframe.image(processed_image, channels="BGR", use_column_width=True)
                
                # Streamlit-da framerate ni pasaytirish uchun kichik kutish
                time.sleep(0.01)
            
            cap.release()
        else:
            st.error("Kamerani ishga tushirishda xatolik yuz berdi.")
    
    # Qo'shimcha ma'lumot
    with st.expander("Ilova haqida ma'lumot"):
        st.write("""
        Bu ilova mediapipe va OpenCV kutubxonalaridan foydalanib real vaqt rejimida tana qismlarining harakatini aniqlaydi.
        
        Quyidagi tana qismlarining harakatlari aniqlanadi:
        - Bosh
        - Tana
        - Chap qo'l
        - O'ng qo'l
        - Chap oyoq
        - O'ng oyoq
        
        Ilovani ishlatish uchun "Kamerani yoqish" tugmasini bosing.
        """)

if __name__ == "__main__":
    main()