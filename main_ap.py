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
    # Input turini tanlash
    input_type = st.sidebar.radio("Kirish turini tanlang", ["Video yuklash", "Test rejimi"])
    
    stframe = col1.empty()
    
    if input_type == "Video yuklash":
        uploaded_file = st.sidebar.file_uploader("Video faylni yuklang", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            # Video faylni saqlash
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            
            # Video faylni ochish
            cap = cv2.VideoCapture(temp_file)
            prev_landmarks = None
            
            # Video parametrlarini olish
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Ishga tushirish va to'xtatish tugmalari
            col_start, col_stop = st.columns(2)
            start_button = col_start.button("Videoni qayta ishlash")
            stop_button = col_stop.button("To'xtatish")
            
            # To'xtatish o'zgaruvchisi
            if 'processing' not in st.session_state:
                st.session_state.processing = False
            
            if start_button:
                st.session_state.processing = True
            
            if stop_button:
                st.session_state.processing = False
                st.warning("Video qayta ishlash to'xtatildi!")
            
            # Video qayta ishlash
            if st.session_state.processing:
                frame_counter = 0
                
                while st.session_state.processing and frame_counter < frame_count:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Frameni qayta ishlash
                    processed_frame, prev_landmarks = process_frame(frame, prev_landmarks)
                    
                    # Frameni ko'rsatish
                    stframe.image(processed_frame, channels="BGR", use_column_width=True)
                    
                    # Progress barni yangilash
                    progress_bar.progress((frame_counter + 1) / frame_count)
                    
                    # FPS ni nazorat qilish
                    time.sleep(1/fps)
                    
                    frame_counter += 1
                    
                    # To'xtatish tugmasi bosilganini tekshirish
                    if not st.session_state.processing:
                        break
                
                cap.release()
                if frame_counter >= frame_count:
                    st.success("Video qayta ishlandi!")
                    st.session_state.processing = False
    
    elif input_type == "Test rejimi":
        st.warning("Bu rejim test uchun mo'ljallangan. Haqiqiy kamera o'rniga test tasvir ishlatiladi.")
        
        # Test tasvir yaratish
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img.fill(200)  # Kulrang fon
        
        # Test tasvirga odam shaklini chizish
        cv2.line(img, (320, 100), (320, 300), (0, 0, 255), 2)  # Tana
        cv2.circle(img, (320, 100), 30, (0, 0, 255), -1)  # Bosh
        cv2.line(img, (320, 150), (250, 220), (0, 0, 255), 2)  # Chap qo'l
        cv2.line(img, (320, 150), (390, 220), (0, 0, 255), 2)  # O'ng qo'l
        cv2.line(img, (320, 300), (280, 400), (0, 0, 255), 2)  # Chap oyoq
        cv2.line(img, (320, 300), (360, 400), (0, 0, 255), 2)  # O'ng oyoq
        
        stframe.image(img, channels="BGR", use_column_width=True)
        
        st.info("Test rejimida harakatlarni aniqlash tavsiya etilmaydi. Haqiqiy videoni yuklang.")
    
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
        
        Ilovani ishlatish uchun video faylni yuklang va "Videoni qayta ishlash" tugmasini bosing.
        To'xtatish uchun "To'xtatish" tugmasini bosing.
        """)

if __name__ == "__main__":
    main()