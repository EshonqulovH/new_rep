import cv2
import streamlit as st
import numpy as np
import time

def main():
    st.set_page_config(page_title="Kamera ilovasi", page_icon="📷", layout="wide")
    st.title("Kamera orqali tasvir ko'rish")
    
    st.warning("""
    ### Eslatma:
    1. Kamerangiz ruxsatini berishingiz kerak
    2. Agar kamera ishlamasa, brauzer sozlamalarini tekshiring
    3. Kamera ochilishini kutib turing
    """)
    
    # Kamera variantlari
    cam_options = st.sidebar.selectbox(
        "Kamera variantini tanlang:",
        ["Asosiy kamera (0)", "Ikkinchi kamera (1)", "Uchinchi kamera (2)"]
    )
    
    # Kamera indeksi
    if "Asosiy" in cam_options:
        cam_index = 0
    elif "Ikkinchi" in cam_options:
        cam_index = 1
    else:
        cam_index = 2
    
    # Kamera ochish tugmasi
    if st.button("Kamerani ochish"):
        # Kamera rahm o'rni
        video_frame = st.empty()
        status_text = st.empty()
        stop_button_place = st.empty()
        
        try:
            # Kamera ochishga harakat qilish
            status_text.info(f"Kamera ochilmoqda... ({cam_index})")
            
            # Turli kamera ochish variantlarini sinab ko'rish
            cap = None
            
            # Oddiy OpenCV kamera ochish
            cap = cv2.VideoCapture(cam_index)
            
            # Kamera ochilishini tekshirish
            if not cap.isOpened():
                status_text.error(f"Kamera {cam_index} ochilmadi.")
                st.error("Kamerani ochishda xatolik yuz berdi.")
                if cap:
                    cap.release()
                return
            
            # Kamera ayrim parametrlarini sinab ko'rish
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Kamera ochildi
            status_text.success("Kamera muvaffaqiyatli ochildi!")
            
            stop_button = False
            
            # Kamera tasvirlarini uzluksiz ko'rsatish
            while not stop_button:
                # Kamera kadrlarini o'qish
                ret, frame = cap.read()
                
                if not ret:
                    status_text.error("Kamera kadrlarini o'qishda muammo!")
                    time.sleep(0.5)
                    continue
                
                # Kadrni ko'rsatish
                frame = cv2.flip(frame, 1)  # Ko'zgu tasvir
                
                # Rangli tasvir formatini o'zgartirish
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Vaqtni qo'shish
                current_time = time.strftime("%H:%M:%S")
                cv2.putText(rgb_frame, f"Vaqt: {current_time}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Kadrni ko'rsatish
                video_frame.image(rgb_frame, use_column_width=True)
                
                # To'xtatish tugmasi
                stop_button = stop_button_place.button("To'xtatish", key=f"stop_{time.time()}")
                
                # Kadrlar o'rtasida kutish
                time.sleep(0.05)
            
            # Kamerani yopish
            cap.release()
            status_text.info("Kamera yopildi.")
            
        except Exception as e:
            st.error(f"Xatolik yuz berdi: {str(e)}")
            if 'cap' in locals() and cap:
                cap.release()

if __name__ == "__main__":
    main()