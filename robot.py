import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
from ultralytics import YOLO

# ==============================================================================
# 1. NASTAVENIE A INICIALIZÁCIA
# ==============================================================================
st.set_page_config(page_title="UAV Live Tracking", layout="wide")
st.title("UAV Live Tracking Platform")
st.markdown("---")

# Jednoduché a garantované stiahnutie videa
VIDEO_PATH = "test_video.mp4"
VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"

if not os.path.exists(VIDEO_PATH):
    with st.spinner("Sťahujem testovacie video, chvíľu strpenia..."):
        urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)

# Načítanie YOLO modelu
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    return model

model = load_model()

# ==============================================================================
# 2. ROZLOŽENIE STRÁNKY (PLACEHOLDERY)
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hlavný Stream (Live)")
    main_placeholder = st.empty()

with col2:
    st.subheader("Výrez (ROI)")
    roi_placeholder = st.empty()
    
    st.subheader("Výpočty")
    math_placeholder = st.empty()

# ==============================================================================
# 3. PRIAMY LIVE CYKLUS (BEZ PRÍPRAVY)
# ==============================================================================
if st.button("Spustiť video a live výpočty"):
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        st.error("Nepodarilo sa otvoriť video súbor.")
    else:
        CENTER_X, CENTER_Y = 640 // 2, 360 // 2
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Koniec videa
            
            # Zmenšenie pre rýchlejší prenos a AI
            frame = cv2.resize(frame, (640, 360))
            
            # Spustenie YOLO
            results = model(frame, imgsz=160, verbose=False)
            person_detected = False
            
            for box in results[0].boxes:
                # 0 = trieda 'person' v YOLO
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
                    person_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Výpočet ťažiska a odchýliek
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    err_x = cx - CENTER_X
                    err_y = cy - CENTER_Y
                    distance = np.sqrt(err_x**2 + err_y**2)
                    reward = (conf * 2.5) - (distance * 0.0015)
                    
                    # 1. Kreslenie do hlavného videa
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                    cv2.line(frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
                    cv2.putText(frame, f"TRACKING {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 2. Vytvorenie a zobrazenie ROI výrezu
                    roi = frame[max(0, y1-20):min(360, y2+20), max(0, x1-20):min(640, x2+20)]
                    if roi.size > 0:
                        roi_placeholder.image(cv2.resize(roi, (150, 150)), channels="BGR")
                    
                    # 3. Výpis Live výpočtov
                    math_placeholder.markdown(f"""
* **Stav:** `ZAMERANÉ`
* **Istota (Confidence):** `{conf:.2%}`
* **Odchýlka od stredu:** `X: {err_x}px | Y: {err_y}px`
* **Vzdialenosť:** `{distance:.2f}px`
* **RL Skóre:** `{reward:+.4f}`

**Rovnica:**
$$R = ({conf:.2f} \cdot 2.5) - ({distance:.2f} \cdot 0.0015)$$
""")
                    break # Sledujeme iba prvú nájdenú osobu
            
            # Ak sa osoba nenájde, zobrazíme hľadanie
            if not person_detected:
                empty_roi = np.zeros((150, 150, 3), dtype=np.uint8)
                cv2.putText(empty_roi, "HĽADÁM...", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                
                roi_placeholder.image(empty_roi, channels="BGR")
                math_placeholder.markdown("""
* **Stav:** `HĽADANIE CIEĽA...`
* Čakám na vizuálny kontakt.
""")

            # Odoslanie snímky priamo na obrazovku v prehliadači
            main_placeholder.image(frame, channels="BGR", use_container_width=True)
            
        cap.release()
        st.success("Prehrávanie videa skončilo.")
