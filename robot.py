import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
import math
import time
from ultralytics import YOLO

st.set_page_config(page_title="UAV Tracking", layout="wide")
st.title("UAV Live Tracking Platform")
st.markdown("---")

VIDEO_PATH = "test_video.mp4"
VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"

# ==========================================================
# 1. AGRESÍVNA KONTROLA SÚBORU (Tu bol pravdepodobne problém)
# ==========================================================
def ensure_video_exists():
    # Ak súbor neexistuje, alebo je menší ako 1MB (poškodený), stiahne ho znova
    if not os.path.exists(VIDEO_PATH) or os.path.getsize(VIDEO_PATH) < 1000000:
        with st.spinner("Sťahujem čisté video pre simuláciu..."):
            urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)

ensure_video_exists()

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    return model

model = load_model()

# ==========================================================
# 2. ROZLOŽENIE UI
# ==========================================================
col1, col2 = st.columns([2, 1])

with col1:
    video_placeholder = st.empty()

with col2:
    roi_placeholder = st.empty()
    math_placeholder = st.empty()

# Úvodná obrazovka
empty_roi = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(empty_roi, "CAKAM NA START", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
roi_placeholder.image(empty_roi, channels="RGB")
math_placeholder.info("Klikni na tlačidlo nižšie pre štart misie.")

# ==========================================================
# 3. BEZPEČNÝ CYKLUS PRE PREHLIADAČ
# ==========================================================
if st.button("🚀 SPUSTIŤ SLEDOVANIE", type="primary"):
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        st.error("Kritická chyba: Server nedokáže otvoriť video súbor. Skús reštartovať Streamlit appku.")
    else:
        WIDTH, HEIGHT = 480, 270
        CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
        
        cached_box = None
        frame_idx = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                
                # YOLO každú 4. snímku
                if frame_idx % 4 == 0:
                    results = model(frame, imgsz=128, verbose=False)
                    person_found = False
                    
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cached_box = (x1, y1, x2, y2, conf)
                            person_found = True
                            break
                    
                    if not person_found:
                        cached_box = None
                
                if cached_box is not None:
                    x1, y1, x2, y2, conf = cached_box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    err_x = cx - CENTER_X
                    err_y = cy - CENTER_Y
                    distance = math.sqrt(err_x**2 + err_y**2)
                    reward = (conf * 2.5) - (distance * 0.0015)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 15, 1)
                    cv2.line(frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
                    
                    roi = frame[max(0, y1-15):min(HEIGHT, y2+15), max(0, x1-15):min(WIDTH, x2+15)]
                    if roi.size > 0:
                        roi_rgb = cv2.cvtColor(cv2.resize(roi, (150, 150)), cv2.COLOR_BGR2RGB)
                        roi_placeholder.image(roi_rgb, channels="RGB")
                    
                    math_placeholder.markdown(f"""
**Stav:** ZAMERANÉ 🎯 | **AI:** {conf:.1%}
* $e_x$: {err_x}px | $e_y$: {err_y}px
* **Vzdialenosť:** {distance:.1f}px
* **Skóre:** {reward:+.3f}
""")
                else:
                    cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 15, 1)
                    searching_roi = np.zeros((150, 150, 3), dtype=np.uint8)
                    cv2.putText(searching_roi, "HLADAM...", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    roi_placeholder.image(searching_roi, channels="RGB")
                    math_placeholder.warning("Hľadám cieľ...")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Zabráni spadnutiu prehliadača
                time.sleep(0.02)
                
        except Exception as e:
            st.error(f"Slučka spadla kvôli chybe siete/servera: {e}")
            
        finally:
            cap.release()
            st.success("Video ukončené.")
