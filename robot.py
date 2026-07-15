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
    # R = R_conf - P_dist (Maximalizácia istoty detekcie s penalizáciou za pixelovú odchýlku od stredu)
    return (confidence * 2.5) - (np.sqrt(error_x**2 + error_y**2) * 0.0015)

# Inicializácia reálneho videa z internetu (Intel Benchmark)
video_source = "intel_people_detection.mp4"

if not os.path.exists(video_source):
    st.info("Inicializujem reálne testovacie video z internetu (Intel Benchmark). Prosím, počkajte chvíľu...")
    try:
        url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
        urllib.request.urlretrieve(url, video_source)
        st.success("Reálne video úspešne stiahnuté a pripravené!")
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
    
    # Úvodný statický náhľad videa pred spustením
    if os.path.exists(video_source):
        cap_preview = cv2.VideoCapture(video_source)
        ret, frame_preview = cap_preview.read()
        if ret:
            main_placeholder.image(cv2.resize(frame_preview, (640, 360)), channels="BGR", use_container_width=True)
        cap_preview.release()

with col2:
    st.subheader("Takticky Mikro-Vyrez (ROI)")
    zoom_placeholder = st.empty()
    
    st.subheader("Telemetria a Vypocty (LIVE)")
    telemetry_placeholder = st.empty()

# Predpripravené statické okná pred štartom
SEARCH_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(SEARCH_SCREEN, "SEARCHING...", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")

start_button = st.button("Spustiť aktívne sledovanie")

# ==============================================================================
# ŽIVÝ CYKLUS - SYNCHRONIZOVANÉ LIVE VYKRESLENIE VŠETKÝCH PANELOV
# ==============================================================================
if start_button:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Chyba pri otváraní video streamu.")
    else:
        CENTER_X, CENTER_Y = 640 // 2, 360 // 2
        predictor = TrajectoryPredictor()
        occlusion_counter = 0
        current_frame_idx = 0
        frame_skip = 2  # Spracovávame každú 2. snímku pre zachovanie plynulosti a live odozvy
        
        PRED_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
        cv2.putText(PRED_SCREEN, "PREDICTING...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            current_frame_idx += 1
            if current_frame_idx % frame_skip != 0:
                continue
            
            web_frame = cv2.resize(frame, (640, 360))
            found = False

            # Inferenčný krok YOLOv8 (imgsz=160 extrémne zrýchli beh na CPU pre live odozvu)
            results = model(web_frame, imgsz=160, verbose=False)
            
            for box in results[0].boxes:
                # 0 = person v datasete COCO
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    predictor.update(cx, cy)
                    err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                    reward = evaluate_reward(err_x, err_y, conf)
                    
                    # Vykreslenie zameriavacieho kríža a boxu do hlavného videa
                    cv2.rectangle(web_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(web_frame, f"ID_0: PERSON {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.drawMarker(web_frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                    cv2.line(web_frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
                    
                    # LIVE aktualizácia taktického výrezu ROI
                    crop = web_frame[max(0, y1-15):min(360, y2+15), max(0, x1-15):min(640, x2+15)]
                    if crop.size > 0:
                        zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                        
                    # LIVE aktualizácia telemetrických parametrov a prepočtov
                    telemetry_placeholder.markdown(f"""
* **Stav UAV:** `TRACKING ACTIVE` 
* **Confidence (Istota AI):** `{conf:.2%}` 
* **Regulačná odchýlka $e_x, e_y$:** `X: {err_x}px | Y: {err_y}px`
* **Euklidovská vzdialenosť:** `{np.sqrt(err_x**2 + err_y**2):.2f}px`
* **RL Reward (Odmena pre agenta):** `{reward:+.4f}`
***
**Matematický výpočet odmeny:**
$$R = (2.5 \\cdot {conf:.2f}) - (0.0015 \\cdot {np.sqrt(err_x**2 + err_y**2):.2f}) = {reward:+.4f}$$
""")
                    found = True
                    occlusion_counter = 0
                    break  # Sledujeme prioritne jedného človeka v zábere

            # Logika prediktívneho filtra pri oklúzii (prekážke / strate vizuálneho kontaktu)
            if not found:
                prediction = predictor.predict_next()
                if prediction and occlusion_counter < 30:
                    occlusion_counter += 1
                    px, py = prediction
                    predictor.update(px, py)
                    
                    # Vykreslenie predpovedaného bodu žltou farbou
                    cv2.circle(web_frame, (px, py), 10, (0, 255, 255), 2)
                    cv2.putText(web_frame, f"PREDIKCIA ({occlusion_counter})", (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.drawMarker(web_frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                    
                    zoom_placeholder.image(PRED_SCREEN, channels="BGR")
                    telemetry_placeholder.markdown(f"""
* **Stav UAV:** `PREDIKCIA (ZÁKRYT)` 
* **Snímky naslepo:** `{occlusion_counter} / 30`
* **Predpovedaná poloha:** `X: {px}px | Y: {py}px`
* **Regulačná odchýlka:** `X: {px - CENTER_X}px | Y: {py - CENTER_Y}px`
* **RL Reward (Odmena pre agenta):** `0.0000` (Penalizácia za stratu priameho vizuálneho kontaktu)
""")
                else:
                    predictor.reset()
                    zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")
                    telemetry_placeholder.markdown("""
* **Stav UAV:** `VYHLADAVANIE / SEARCHING` 
* **Confidence (Istota AI):** `0.00%`
* **Regulačná odchýlka:** `N/A`
* **RL Reward (Odmena pre agenta):** `N/A`
""")
            
            # LIVE prekreslenie hlavného optického streamu na obrazovke
            main_placeholder.image(web_frame, channels="BGR", use_container_width=True)
            
            # Veľmi krátka pauza (10ms) na uvoľnenie sieťového vlákna pre prehliadač
            time.sleep(0.01)
            
        cap.release()
        st.success("Sledovanie úspešne dokončené.")
