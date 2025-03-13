import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image
import io
import base64
import os

# Sahifa konfiguratsiyasi
st.set_page_config(page_title="Tana Harakatlarini Aniqlash", page_icon="🏃‍♂️", layout="wide")
st.title("Tana Harakatlarini Aniqlash Ilovasi")

# MediaPipe pose modeli
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Sidebar sozlamalari
st.sidebar.header("Sozlamalar")
# Harakat aniqlash chegarasi
threshold = st.sidebar.slider("Harakat sezgirligi", min_value=0.01, max_value=0.1, value=0.03, step=0.01)
# Reset interval set as a constant instead of user-configurable
reset_interval = 1.0  # Default value in seconds

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

# Holatlar kolonklarini yaratish
col1, col2 = st.columns([3, 1])

# Asosiy oynaga kirish turini tanlash
input_type = st.radio("Kirish turini tanlang", ["Video yuklash", "Test rejimi"], horizontal=True)

# Streamlit ong panelda tana qismlarini korsatish uchun joy
with col2:
    st.header("Tana holati")
    status_placeholders = {}
    for part in ["Bosh", "Tana", "Chap qo'l", "O'ng qo'l", "Chap oyoq", "O'ng oyoq"]:
        status_placeholders[part] = st.empty()

# Qaysi tana qismlari harakatlangani haqida malumot 
moving_parts = {
    "Bosh": False,
    "Tana": False,
    "Chap qo'l": False,
    "O'ng qo'l": False,
    "Chap oyoq": False,
    "O'ng oyoq": False
}


last_detection_time = {
    "Bosh": 0,
    "Tana": 0,
    "Chap qo'l": 0,
    "O'ng qo'l": 0,
    "Chap oyoq": 0,
    "O'ng oyoq": 0
}

# Temp fayllar uchun papka yaratish
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Video frameni oqish va qayta ishlash
def process_frame(image, prev_landmarks):
    try:
        # RGB formatiga otkazish mediapipe un
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Pose model orqali harakatlarni aniqlash
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Ozgartirilgan rasmni yaratish
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Harakatlarni aniqlash
            current_time = time.time()
            
            if prev_landmarks:
                current_landmarks = results.pose_landmarks.landmark
                
                # Har bir muhim tana qismi un tekshirish
                for idx, part_name in body_parts.items():
                    # Agar bu tana qismi avval harakatlanayotgan deb belgilangan bolsa va vaqt otgan bolsa
                    if moving_parts[part_name] and (current_time - last_detection_time[part_name]) > reset_interval:
                        moving_parts[part_name] = False
                    
                    prev = prev_landmarks[idx]
                    curr = current_landmarks[idx]
                    dx = abs(curr.x - prev.x)
                    dy = abs(curr.y - prev.y)
                    
                    # Harakat aniqlangan bosa
                    if dx > threshold or dy > threshold:
                        # Agar bu tana qismi hali harakatlanayotgan deyilsa
                        if not moving_parts[part_name]:
                            moving_parts[part_name] = True
                            last_detection_time[part_name] = current_time
            
            # Ong panelda holatlarni yangilash
            for part, is_moving in moving_parts.items():
                status = "Harakatlanmoqda 🟢" if is_moving else "Harakatsiz 🔴"
                status_placeholders[part].write(f"{part}: {status}")
            
            # Joriy holatni qaytarish..
            return annotated_image, results.pose_landmarks.landmark
        
        return image, prev_landmarks
    except Exception as e:
        st.error(f"Frameni qayta ishlashda xatolik: {str(e)}")
        return image, prev_landmarks
# video yuklash
def main():
    stframe = col1.empty()
    
    if input_type == "Video yuklash":        
        st.markdown("""
        **Video yuklash bo'yicha ma'lumot:**
        
        - Telefonda video yuklashda muammo bo'lsa, video hajmi kichikroq bo'lishiga ishonch hosil qiling
        - MP4 formatidagi videolarni tanlash tavsiya etiladi
        - Video yuklangandan keyin "Videoni qayta ishlash" tugmasini bosing
        """)
        
        uploaded_file = st.file_uploader("Video faylni yuklang", type=["mp4", "avi", "mov"], accept_multiple_files=False)
        
        if uploaded_file is not None:
            try:
                # Temp papka yaratishh
                ensure_dir('temp')
                
                # Video faylni saqlas
                temp_file = "temp/video.mp4"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"Video muvaffaqiyatli yuklandi! Hajmi: {round(os.path.getsize(temp_file)/1024/1024, 2)} MB")
                
                # Video faylni ochish
                cap = cv2.VideoCapture(temp_file)
                
                if not cap.isOpened():
                    st.error("Video faylini ochib bo'lmadi. Fayl formati qo'llab-quvvatlanmasligi mumkin.")
                else:
                    prev_landmarks = None
                    
                    # Video parmetrlarini olish
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    if fps <= 0:
                        fps = 25  
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count <= 0:
                        # Frameni hisoblash
                        frame_count = 1000  # Taxminiy qiymat olinishi
                    
                    # Video malumotlarini ko'rsatish
                    st.write(f"Video uzunligi: ~{frame_count/fps:.1f} soniya, FPS: {fps}")
                    
                    # Progres bar
                    progress_bar = st.progress(0)
                    
                    # Ishga tushirish va toxtatish tugmalari
                    col_start, col_stop = st.columns(2)
                    start_button = col_start.button("Videoni qayta ishlash")
                    stop_button = col_stop.button("To'xtatish")
                    
                    # Toxtatish 
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
                            
                            try:
                                # Frameni qayta ishlash
                                processed_frame, prev_landmarks = process_frame(frame, prev_landmarks)
                                
                                # Frameni korsatish
                                stframe.image(processed_frame, channels="BGR", use_container_width=True)
                                
                                # Progress barni yangilash
                                progress_value = min(1.0, (frame_counter + 1) / frame_count)
                                progress_bar.progress(progress_value)
                                
    
                                time.sleep(0.1)  # Telefonlar uchun sekinroq ko'rsatish
                                
                                frame_counter += 1
                                
                                # To'xtatish tugmasi bosilganini tekshirish uchun
                                if not st.session_state.processing:
                                    break
                            except Exception as e:
                                st.error(f"Frame {frame_counter} qayta ishlashda xatolik: {str(e)}")
                                break
                        
                        cap.release()
                        
                        if frame_counter >= frame_count:
                            st.success("Video qayta ishlandi!")
                        else:
                            st.info(f"Video qayta ishlash {frame_counter}/{frame_count} frameda to'xtadi.")
                        
                        st.session_state.processing = False
            except Exception as e:
                st.error(f"Video bilan ishlashda xato: {str(e)}")
    
    elif input_type == "Test rejimi":
        st.warning("Bu rejim test uchun mo'ljallangan. Haqiqiy kamera o'rniga test tasvir ishlatiladi.")
        
        # Test uchun tasvir yaratish
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img.fill(200) 
        
        # Test tasvirga odam shaklini chizish skletini aniqrogi
        cv2.line(img, (320, 100), (320, 300), (0, 0, 255), 2)  # Tana
        cv2.circle(img, (320, 100), 30, (0, 0, 255), -1)  # Bosh
        cv2.line(img, (320, 150), (250, 220), (0, 0, 255), 2)  # Chap qo'l
        cv2.line(img, (320, 150), (390, 220), (0, 0, 255), 2)  # O'ng qo'l
        cv2.line(img, (320, 300), (280, 400), (0, 0, 255), 2)  # Chap oyoq
        cv2.line(img, (320, 300), (360, 400), (0, 0, 255), 2)  # O'ng oyoq
        
        stframe.image(img, channels="BGR", use_container_width=True)
        
        st.info("Test rejimida harakatlarni aniqlash tavsiya etilmaydi. Haqiqiy videoni yuklang.")
    
    # Qoshimcha malumot
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