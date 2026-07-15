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
# KONFIGURÁCIA
# ==============================================================================
st.set_page_config(page_title="UAV Profi Tracking", layout="wide")
st.title("UAV Live Tracking (WebRTC HUD)")
st.markdown("---")
st.info("Klikni na tlačidlo **START** nižšie pre načítanie WebRTC kanála.")

VIDEO_PATH = "test_video.mp4"
VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"

if not os.path.exists(VIDEO_PATH):
    with st.spinner("Sťahujem testovacie video..."):
        urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    return model

model = load_model()

# ==============================================================================
# WEBRTC ENGINE & HUD KOMPOZÍCIA
# ==============================================================================
def create_player():
    """Vytvorí prehrávač lokálneho videa pre WebRTC"""
    from aiortc.contrib.media import MediaPlayer
    return MediaPlayer(VIDEO_PATH)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Spracuje každú snímku naživo bez latencie"""
    # Prevod z WebRTC formátu do OpenCV (BGR)
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 360))
    
    # Rýchla AI Detekcia
    results = model(img, imgsz=160, verbose=False)
    
    # Vytvorenie čierneho bočného panelu pre HUD (Heads-Up Display)
    # Rozmery: výška 360px, šírka 300px
    hud = np.zeros((360, 300, 3), dtype=np.uint8)
    CENTER_X, CENTER_Y = 640 // 2, 360 // 2
    
    person_detected = False
    
    for box in results[0].boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Výpočty
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            err_x = cx - CENTER_X
            err_y = cy - CENTER_Y
            distance = math.sqrt(err_x**2 + err_y**2)
            reward = (conf * 2.5) - (distance * 0.0015)
            
            # Kreslenie zameriavačov do hlavného videa
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(img, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.line(img, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
            
            # --- HUD: Výrez (ROI) ---
            roi = img[max(0, y1-20):min(360, y2+20), max(0, x1-20):min(640, x2+20)]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, (150, 150))
                # Umiestnenie výrezu do HUD panelu
                hud[20:170, 75:225] = roi_resized
                cv2.rectangle(hud, (75, 20), (225, 170), (0, 255, 0), 2)
                cv2.putText(hud, "TAKTVICKY VYREZ", (75, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                
            # --- HUD: Telemetria a výpočty ---
            cv2.putText(hud, "STAV: ZAMERANE", (15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(hud, f"ISTOTA: {conf*100:.1f} %", (15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"ODCHYLKA: X:{err_x}px | Y:{err_y}px", (15, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"VZDIA.:   {distance:.1f}px", (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"RL SKORE: {reward:+.4f}", (15, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            break # Sledujeme prvý cieľ
            
    if not person_detected:
        cv2.putText(hud, "HLADAM CIEL...", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.drawMarker(img, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
        
    # Spojenie hlavného videa (640px) a HUD panelu (300px) do jedného obrazu šírky 940px
    final_frame = np.hstack((img, hud))
    
    return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

# Nastavenie STUN servera, ak to bežíš v cloude (obchádza sieťové firewally)
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Samotný WebRTC Streamer
webrtc_streamer(
    key="uav-stream",
    mode=WebRtcMode.RECVONLY,
    rtc_configuration=rtc_config,
    player_factory=create_player,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)
