import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# MediaPipe Pose modeli
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Body parts dictionary
body_parts = {
    "Bosh": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Tana": [11, 12, 13, 14, 23, 24],
    "Chap qo'l": [11, 13, 15, 17, 19, 21],
    "O'ng qo'l": [12, 14, 16, 18, 20, 22],
    "Chap oyoq": [23, 25, 27, 29, 31],
    "O'ng oyoq": [24, 26, 28, 30, 32]
}

# Video protsessor
class PoseDetector(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.previous_landmarks = None
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Rasmni kichiklashtirish
        img = cv2.resize(img, (640, 480))
        
        # Rangni RGB ga o'zgartirish
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Pozani aniqlash
        results = self.pose.process(img_rgb)
        
        moving_parts = []
        
        if results.pose_landmarks:
            # Skeletni chizish
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )
            
            # Harakatni aniqlash
            current_landmarks = results.pose_landmarks.landmark
            
            if self.previous_landmarks:
                # Har bir tana qismi uchun harakatni tekshirish
                for part_name, indexes in body_parts.items():
                    movement_detected = False
                    total_diff = 0
                    valid_points = 0
                    
                    for idx in indexes:
                        if idx < len(current_landmarks) and idx < len(self.previous_landmarks):
                            prev = self.previous_landmarks[idx]
                            curr = current_landmarks[idx]
                            
                            # Nuqtalar orasidagi masofani hisoblash
                            diff = np.sqrt(
                                (curr.x - prev.x) ** 2 + 
                                (curr.y - prev.y) ** 2
                            )
                            
                            if diff > 0.01:  # Minimal harakat chegarasi
                                movement_detected = True
                                total_diff += diff
                                valid_points += 1
                    
                    # Agar harakat sezilarli bo'lsa, qo'shish
                    if movement_detected and valid_points > 0 and total_diff/valid_points > 0.015:
                        moving_parts.append(part_name)
            
            # Oldingi kadr ma'lumotlarini saqlash
            self.previous_landmarks = current_landmarks
        
        # Aniqlangan harakatlarni ko'rsatish
        detected_movement = ", ".join(moving_parts) if moving_parts else "Harakat aniqlanmadi"
        cv2.putText(
            img, 
            f"Harakat: {detected_movement}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        return img

# Streamlit app
st.set_page_config(page_title="Skelet Aniqlash", page_icon="🎥", layout="wide")
st.title("Real Vaqt Skelet Aniqlash")

st.warning("""
### Eslatma:
1. Kamerangiz ruxsatini berishingiz kerak
2. Brauzer Chrome yoki Firefox bo'lishi kerak
3. Kamera ochilganda kutib turing
""")

# Dastur haqida ma'lumot
st.info("""
#### Dastur imkoniyatlari:
- Skeletni real vaqt rejimida aniqlash
- Harakatlarni aniqlash va ko'rsatish
- Barcha tana qismlarini kuzatish
""")

# RTC konfiguratsiyasi
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="pose-detection",
    video_processor_factory=PoseDetector,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False
    },
    async_processing=True
)

# Kamera ochilganda qo'shimcha ma'lumot
if webrtc_ctx.state.playing:
    st.success("Kamera muvaffaqiyatli ishga tushdi!")
    st.markdown("""
    ### Qo'llanma:
    1. Kamerada ko'rinishingizni tekshiring
    2. Harakatlaringiz pastki qismda ko'rsatiladi
    3. Dasturni yopish uchun "Stop" tugmasini bosing
    """)
else:
    st.error("Kamera ishga tushmadi. Kamera ruxsatini tekshiring.")