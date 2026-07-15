import streamlit as st
import cv2
import os
import time
import urllib.request
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# KONFIGURÁCIA STRÁNKY
# ==============================================================================
st.set_page_config(
    page_title="UAV Active Tracking Platform",
    layout="wide"
)

st.title("UAV Active Tracking Platform")
st.markdown("---")

# Načítanie robustného detekčného modelu YOLOv8 pre reálne zábery ľudí
@st.cache_resource
def load_resources():
    model = YOLO('yolov8n.pt')
    model.to("cpu")  # Vynútené stabilné CPU pre cloud servery
    return model

model = load_resources()

# ==============================================================================
# MATEMATICKÉ JADRO A VÝPOČTY
# ==============================================================================
class TrajectoryPredictor:
    def __init__(self, history_len=6):
        self.history = []
        self.history_len = history_len

    def update(self, x, y):
        self.history.append((x, y))
        if len(self.history) > self.history_len: self.history.pop(0)

    def predict_next(self):
        if len(self.history) < 2: return None
        deltas_x = [self.history[i][0] - self.history[i-1][0] for i in range(1, len(self.history))]
        deltas_y = [self.history[i][1] - self.history[i-1][1] for i in range(1, len(self.history))]
        return int(self.history[-1][0] + sum(deltas_x)/len(deltas_x)), int(self.history[-1][1] + sum(deltas_y)/len(deltas_y))

    def reset(self):
        self.history.clear()

def evaluate_reward(error_x, error_y, confidence):
    return (confidence * 2.5) - (np.sqrt(error_x**2 + error_y**2) * 0.0015)

# Inicializácia reálneho videa
video_source = "intel_people_detection.mp4"

if not os.path.exists(video_source):
    st.info("Inicializujem reálne testovacie video z internetu (Intel Benchmark). Prosím, počkajte chvíľu...")
    try:
        url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
        urllib.request.urlretrieve(url, video_source)
        st.success("Reálne video úspešne stiahnuté a pripravené na analýzu!")
        st.rerun()
    except Exception as e:
        st.error(f"Nepodarilo sa stiahnuť video: {e}")

# ==============================================================================
# ROZLOZENIE STRÁNKY (LAYOUT)
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hlavny Opticky Stream (UAV Main Sensor)")
    main_placeholder = st.empty()

with col2:
    st.subheader("Takticky Mikro-Vyrez (ROI)")
    zoom_placeholder = st.empty()
    
    st.subheader("Telemetria a Vypocty")
    telemetry_placeholder = st.empty()

# Predpripravené statické okná pred štartom
SEARCH_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(SEARCH_SCREEN, "SEARCHING...", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")

# Zobrazenie statického náhľadu videa pred spustením
if os.path.exists(video_source):
    cap_preview = cv2.VideoCapture(video_source)
    ret, frame_preview = cap_preview.read()
    if ret:
        main_placeholder.image(cv2.resize(frame_preview, (640, 360)), channels="BGR", use_container_width=True)
    cap_preview.release()

start_button = st.button("Spustiť optimalizované plynulé sledovanie")

# ==============================================================================
# ASYNCHRÓNNA ANALÝZA A VYSOKO-RÝCHLOSTNÉ PREHRÁVANIE
# ==============================================================================
if start_button:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Chyba pri otváraní video streamu.")
    else:
        CENTER_X, CENTER_Y = 640 // 2, 360 // 2
        predictor = TrajectoryPredictor()
        occlusion_counter = 0
        
        # Buffery pre ukladanie hotových snímok a telemetrie do RAM
        frames_cache = []
        crops_cache = []
        telemetry_cache = []
        
        # 1. KROK: Ultrarýchly "pre-processing" na serveri (len každá 3. snímka pre úsporu)
        status_box = st.empty()
        status_box.info("AI vykonáva rýchlu analýzu trajektórií na pozadí...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            web_frame = cv2.resize(frame, (640, 360))
            found = False
            
            # Detekcia YOLOv8
            results = model(web_frame, imgsz=320, verbose=False)
            
            for box in results[0].boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    predictor.update(cx, cy)
                    err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                    reward = evaluate_reward(err_x, err_y, conf)
                    
                    # Nakreslenie zameriavača
                    cv2.rectangle(web_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(web_frame, "TRACKING ACTIVE", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Výrez ROI
                    crop = web_frame[max(0, y1-15):min(360, y2+15), max(0, x1-15):min(640, x2+15)]
                    crop_resized = cv2.resize(crop, (150, 150)) if crop.size > 0 else SEARCH_SCREEN
                    
                    # Uloženie dát
                    frames_cache.append(web_frame)
                    crops_cache.append(crop_resized)
                    telemetry_cache.append({
                        "status": "TRACKING ACTIVE",
                        "conf": f"{conf:.2%}",
                        "reward": f"{reward:+.4f}",
                        "err_x": err_x,
                        "err_y": err_y
                    })
                    
                    found = True
                    occlusion_counter = 0
                    break

            # Lineárny prediktor pri strate zamerania
            if not found:
                prediction = predictor.predict_next()
                if prediction and occlusion_counter < 30:
                    occlusion_counter += 1
                    px, py = prediction
                    predictor.update(px, py)
                    
                    cv2.circle(web_frame, (px, py), 10, (0, 255, 255), 2)
                    cv2.putText(web_frame, f"PREDIKCIA ({occlusion_counter})", (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    PRED_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
                    cv2.putText(PRED_SCREEN, "PREDICTING...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    frames_cache.append(web_frame)
                    crops_cache.append(PRED_SCREEN)
                    telemetry_cache.append({
                        "status": f"PREDIKCIA (ZÁKRYT {occlusion_counter}/30)",
                        "conf": "0.00%",
                        "reward": "0.0000",
                        "err_x": px,
                        "err_y": py
                    })
                else:
                    predictor.reset()
                    frames_cache.append(web_frame)
                    crops_cache.append(SEARCH_SCREEN)
                    telemetry_cache.append({
                        "status": "VYHLADAVANIE / MIMO DOSAH",
                        "conf": "0.00%",
                        "reward": "N/A",
                        "err_x": 0,
                        "err_y": 0
                    })
                    
        cap.release()
        status_box.empty()
        
        # 2. KROK: Vykreslenie z pamäte (RAM) bez akéhokoľvek sieťového oneskorenia (Stabilných 25-30 FPS)
        for i in range(len(frames_cache)):
            # Render hlavného obrazu drona naživo
            main_placeholder.image(frames_cache[i], channels="BGR", use_container_width=True)
            
            # Render ROI výrezu
            zoom_placeholder.image(crops_cache[i], channels="BGR")
            
            # Aktualizácia telemetrie
            t = telemetry_cache[i]
            telemetry_placeholder.markdown(f"""
* **Stav systemu:** `{t['status']}` 
* **Confidence (Istota AI):** `{t['conf']}` 
* **RL Reward (Skore odmeny):** `{t['reward']}` 
* **Error X/Y (Odchylka od stredu):** `X: {t['err_x']}px | Y: {t['err_y']}px`
""")
            # Simulácia stabilného 30 FPS snímania (33 milisekúnd pauza)
            time.sleep(0.033)
