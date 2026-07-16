import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import urllib.request
import os
import math
from ultralytics import YOLO

# ==============================================================================
# NASTAVENIE STRÁNKY
# ==============================================================================
st.set_page_config(page_title="UAV Tracking - Bakalárska práca", layout="wide")
st.title("UAV Autonómne Sledovacie Rozhranie")

# Vložíme vysvetlenie priamo pod nadpis
st.markdown("""
### Cieľ práce:
Systém implementuje **vizuálnu servovú slučku**, ktorá v reálnom čase deteguje objekt (človeka) a vypočítava odchýlku od optickej osi pre autonómne riadenie dronu.
""")

# ==============================================================================
# WEBRTC ENGINE (BEZ ZMENY)
# ==============================================================================
VIDEO_PATH = "vtest.avi"
VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
if not os.path.exists(VIDEO_PATH):
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
model = load_model()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 360))
    results = model(img, imgsz=160, verbose=False)
    
    hud = np.zeros((360, 300, 3), dtype=np.uint8)
    CENTER_X, CENTER_Y = 320, 180
    
    person_detected = False
    for box in results[0].boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            err_x, err_y = cx - CENTER_X, cy - CENTER_Y
            dist = math.sqrt(err_x**2 + err_y**2)
            reward = (conf * 2.5) - (dist * 0.0015)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(img, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
            cv2.putText(hud, "STAV: ZAMERANE", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(hud, f"ISTOTA: {conf*100:.1f}%", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"ODCHYLKA: X:{err_x} Y:{err_y}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"SKORE: {reward:+.3f}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            break
            
    if not person_detected:
        cv2.putText(hud, "HLADAM...", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    final_frame = np.hstack((img, hud))
    return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

webrtc_streamer(key="uav-stream", mode=WebRtcMode.RECVONLY, 
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_frame_callback=video_frame_callback)

# ==============================================================================
# LEGENDA METRÍK (TOTO SA ZOBRAZÍ POD VIDEOM)
# ==============================================================================
st.markdown("---")
st.subheader("Vysvetlenie telemetrických veličín")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    * **Odchýlka (Error X, Y):** Vzdialenosť cieľa od stredu obrazu (v pixeloch). Používa sa ako vstup pre PID reguláciu dronu.
    * **Vzdialenosť:** Euklidovská vzdialenosť od stredu, indikujúca mieru vyosenia cieľa.
    """)
with col2:
    st.markdown("""
    * **Istota (Confidence):** Úspešnosť klasifikácie neurónovou sieťou YOLOv8 (v %).
    * **RL Skóre (Reward):** Vypočítaná odmena pre agenta podľa vzorca: $R = (conf \\cdot 2.5) - (dist \\cdot 0.0015)$.
    """)
