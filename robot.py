import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
import math
import time
from ultralytics import YOLO

# ==============================================================================
# 1. NASTAVENIE STRÁNKY
# ==============================================================================
st.set_page_config(page_title="UAV Live Tracking", layout="wide")
st.title("UAV Live Tracking Platform")
st.markdown("---")

VIDEO_PATH = "test_video.mp4"
VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"

if not os.path.exists(VIDEO_PATH):
    with st.spinner("Sťahujem testovacie video pre simuláciu..."):
        urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    return model

model = load_model()

# ==============================================================================
# 2. ROZLOŽENIE STRÁNKY
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hlavný Optický Stream (Live)")
    video_placeholder = st.empty()

with col2:
    st.subheader("Taktický Výrez (ROI)")
    roi_placeholder = st.empty()
    
    st.subheader("Telemetria a Výpočty")
    math_placeholder = st.empty()

# Prvotné zobrazenie pred štartom
empty_roi = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(empty_roi, "CAKAM NA START", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
roi_placeholder.image(empty_roi, channels="RGB")

math_placeholder.markdown("""
* **Stav:** `PRIPRAVENÝ`
* Klikni na tlačidlo pre spustenie misie.
""")

# ==============================================================================
# 3. HLAVNÝ CYKLUS (OPRAVENÉ VYKRESLOVANIE)
# ==============================================================================
if st.button("Spustiť aktívne sledovanie"):
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        st.error("Chyba: Nepodarilo sa otvoriť video súbor.")
    else:
        CENTER_X, CENTER_Y = 480 // 2, 270 // 2
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Koniec videa
            
            frame_idx += 1
            # Preskočíme každú druhú snímku pre plynulosť cloudu
            if frame_idx % 2 != 0:
                continue
            
            frame = cv2.resize(frame, (480, 270))
            results = model(frame, imgsz=128, verbose=False)
            person_detected = False
            
            for box in results[0].boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    err_x = cx - CENTER_X
                    err_y = cy - CENTER_Y
                    distance = math.sqrt(err_x**2 + err_y**2)
                    reward = (conf * 2.5) - (distance * 0.0015)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 15, 1)
                    cv2.line(frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
                    
                    roi = frame[max(0, y1-15):min(270, y2+15), max(0, x1-15):min(480, x2+15)]
                    if roi.size > 0:
                        roi = cv2.resize(roi, (150, 150))
                        # OPRAVA: Konverzia do RGB
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi_placeholder.image(roi_rgb, channels="RGB")
                    
                    math_placeholder.markdown(f"""
* **Stav:** `ZAMERANÉ`
* **Istota AI:** `{conf:.2%}`
* **Odchýlka $e_x, e_y$:** `X: {err_x}px | Y: {err_y}px`
* **Vzdialenosť:** `{distance:.2f}px`
* **RL Skóre:** `{reward:+.4f}`

**Rovnica pre agenta:**
$$R = ({conf:.2f} \\cdot 2.5) - ({distance:.2f} \\cdot 0.0015)$$
""")
                    break
            
            if not person_detected:
                cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 15, 1)
                
                searching_roi = np.zeros((150, 150, 3), dtype=np.uint8)
                cv2.putText(searching_roi, "HLADAM...", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                roi_placeholder.image(searching_roi, channels="RGB")
                
                math_placeholder.markdown("""
* **Stav:** `HĽADANIE CIEĽA...`
* **Istota AI:** `0.00%`
""")

            # OPRAVA 1: Explicitná konverzia farieb z BGR do RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # OPRAVA 2: Odoslanie ako čisté RGB
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # OPRAVA 3: Krátke uspanie kódu donúti server okamžite odoslať obrázok do tvojho prehliadača (zamedzí blokácii)
            time.sleep(0.03)
            
        cap.release()
        st.success("Misia úspešne dokončená.")
