import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import urllib.request
import os
import math
from ultralytics import YOLO
import queue

# ==============================================================================
# 1. KONFIGURÁCIA A INICIALIZÁCIA
# ==============================================================================
st.set_page_config(page_title="UAV Profi Tracking", layout="wide")
st.title("UAV Live Tracking (Rozdelené UI)")
st.markdown("---")

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

# Vytvorenie komunikačnej fronty medzi videom a Streamlitom
if "webrtc_queue" not in st.session_state:
    st.session_state.webrtc_queue = queue.Queue()

# ==============================================================================
# 2. WEBRTC ENGINE A DETEKCIA
# ==============================================================================
def create_player():
    from aiortc.contrib.media import MediaPlayer
    return MediaPlayer(VIDEO_PATH)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 360))
    
    results = model(img, imgsz=160, verbose=False)
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
            
            # Vykreslenie iba do hlavného videa (žiaden text)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(img, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.line(img, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
            
            # Príprava ROI
            roi = img[max(0, y1-20):min(360, y2+20), max(0, x1-20):min(640, x2+20)]
            if roi.size > 0:
                roi = cv2.resize(roi, (150, 150))
            
            # Zmazanie starých dát a odoslanie najnovších do Streamlitu
            while not st.session_state.webrtc_queue.empty():
                try:
                    st.session_state.webrtc_queue.get_nowait()
                except queue.Empty:
                    break
                    
            st.session_state.webrtc_queue.put({
                "status": "found",
                "roi": roi,
                "conf": conf,
                "err_x": err_x,
                "err_y": err_y,
                "distance": distance,
                "reward": reward
            })
            break 
            
    if not person_detected:
        cv2.drawMarker(img, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
        empty_roi = np.zeros((150, 150, 3), dtype=np.uint8)
        cv2.putText(empty_roi, "HLADAM...", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        while not st.session_state.webrtc_queue.empty():
            try: st.session_state.webrtc_queue.get_nowait()
            except queue.Empty: break
            
        st.session_state.webrtc_queue.put({"status": "searching", "roi": empty_roi})
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==============================================================================
# 3. ROZLOŽENIE UI (FRONTEND)
# ==============================================================================
col1, col2 = st.columns([2, 1])

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with col1:
    st.subheader("UAV Hlavná Kamera")
    webrtc_ctx = webrtc_streamer(
        key="uav-stream",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=rtc_config,
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("Taktický Výrez (ROI)")
    roi_placeholder = st.empty()
    
    st.subheader("Telemetria a Výpočty")
    math_placeholder = st.empty()

# ==============================================================================
# 4. AKTUALIZÁCIA TEXTU NAŽIVO
# ==============================================================================
# Keď je video spustené, tento cyklus neustále preberá dáta z fronty a ukazuje ich napravo
if webrtc_ctx.state.playing:
    while True:
        try:
            data = st.session_state.webrtc_queue.get(timeout=0.1)
            
            if data["status"] == "found":
                roi_placeholder.image(data["roi"], channels="BGR")
                
                # Oddelený čistý text napravo
                math_placeholder.markdown(f"""
* **Stav:** `ZAMERANÉ`
* **Istota AI:** `{data['conf']:.2%}`
* **Odchýlka $e_x, e_y$:** `X: {data['err_x']}px | Y: {data['err_y']}px`
* **Vzdialenosť:** `{data['distance']:.2f}px`
* **RL Skóre (Odmena):** `{data['reward']:+.4f}`

**Rovnica pre agenta:**
$$R = ({data['conf']:.2f} \cdot 2.5) - ({data['distance']:.2f} \cdot 0.0015)$$
""")
            else:
                roi_placeholder.image(data["roi"], channels="BGR")
                math_placeholder.markdown("""
* **Stav:** `HĽADANIE CIEĽA...`
* **Istota AI:** `0.00%`
""")
        except queue.Empty:
            # Ak chvíľu neprídu dáta, len ignorujeme
            pass
