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

# 1. KROK: Absolútne spoľahlivá inicializácia reálneho videa
video_source = "intel_people_detection.mp4"

# Sťahovanie videa s jasnou vizuálnou informáciou pre používateľa
if not os.path.exists(video_source):
    st.info("Inicializujem reálne testovacie video z internetu (Intel Benchmark). Prosím, počkajte chvíľu...")
    try:
        # Oficiálne testovacie video od Intelu pre detekciu osôb
        url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
        urllib.request.urlretrieve(url, video_source)
        st.success("Reálne video úspešne stiahnuté a pripravené na analýzu!")
        st.rerun() # Reštart pre načítanie súboru
    except Exception as e:
        st.error(f"Nepodarilo sa stiahnuť video z primárneho zdroja: {e}")
        st.info("Pokúšam sa o alternatívny zdroj...")
        # Záložný link v prípade výpadku primárneho GitHubu
        backup_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
        try:
            urllib.request.urlretrieve(backup_url, video_source)
            st.success("Záložné video úspešne stiahnuté!")
            st.rerun()
        except Exception as err:
            st.error(f"Chyba siete: {err}. Spustite aplikáciu znova.")

# ==============================================================================
# ROZLOZENIE STRÁNKY (LAYOUT)
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hlavny Opticky Stream (UAV Main Sensor)")
    main_placeholder = st.empty()
    
    # Ak sa ešte nezačalo spracovanie, ukážeme statický úvodný náhľad
    if os.path.exists(video_source):
        cap_preview = cv2.VideoCapture(video_source)
        ret, frame_preview = cap_preview.read()
        if ret:
            main_placeholder.image(cv2.resize(frame_preview, (640, 360)), channels="BGR", use_container_width=True)
        cap_preview.release()

with col2:
    st.subheader("Takticky Mikro-Vyrez (ROI)")
    zoom_placeholder = st.empty()
    
    st.subheader("Telemetria a Vypocty")
    telemetry_placeholder = st.empty()

# Predpripravené statické okná pred štartom
SEARCH_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(SEARCH_SCREEN, "SEARCHING...", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")

start_button = st.button("Spustiť aktívne sledovanie misie")

# ==============================================================================
# ŽIVÝ CYKLUS - SYNCHRONIZOVANÉ VYKRESLENIE
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
        frame_skip = 4  # Optimalizácia prechodu snímok na CPU servery
        
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

            # Inferenčný krok neurónovej siete YOLOv8 (detekcia ľudí - index triedy 0)
            results = model(web_frame, imgsz=320, verbose=False)
            
            for box in results[0].boxes:
                # 0 = person v datasete COCO
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    predictor.update(cx, cy)
                    err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                    reward = evaluate_reward(err_x, err_y, conf)
                    
                    status = "TRACKING ACTIVE"
                    cv2.rectangle(web_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(web_frame, f"ID_0: PERSON {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Dynamický výrez ROI okolo detegovaného človeka
                    crop = web_frame[max(0, y1-15):min(360, y2+15), max(0, x1-15):min(640, x2+15)]
                    if crop.size > 0:
                        zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                        
                    # Aktualizácia telemetrie
                    telemetry_placeholder.markdown(f"""
* **Stav systemu:** `TRACKING ACTIVE` 
* **Confidence (Istota AI):** `{conf:.2%}` 
* **RL Reward (Skore odmeny):** `{reward:+.4f}` 
* **Error X/Y (Odchylka od stredu):** `X: {err_x}px | Y: {err_y}px`
""")
                    found = True
                    occlusion_counter = 0
                    break  # Sledujeme prioritne prvého nájdeného človeka

            # Logika prediktívneho filtra pri oklúzii (prekážke / strate vizuálneho kontaktu)
            if not found:
                prediction = predictor.predict_next()
                if prediction and occlusion_counter < 30:
                    occlusion_counter += 1
                    px, py = prediction
                    predictor.update(px, py)
                    
                    cv2.circle(web_frame, (px, py), 10, (0, 255, 255), 2)
                    cv2.putText(web_frame, f"PREDIKCIA ({occlusion_counter})", (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    zoom_placeholder.image(PRED_SCREEN, channels="BGR")
                    telemetry_placeholder.markdown(f"""
* **Stav systemu:** `PREDIKCIA (STRATA KONTAKTU)` 
* **Snímky naslepo:** `{occlusion_counter} / 30`
* **Predpovedaný X/Y:** `X: {px}px | Y: {py}px`
""")
                else:
                    predictor.reset()
                    zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")
                    telemetry_placeholder.markdown("""
* **Stav systemu:** `VYHLADAVANIE / MIMO DOSAH` 
* **Confidence (Istota AI):** `0.00%`
* **RL Reward (Skore odmeny):** `N/A`
""")
            
            # Vykreslenie hlavného obrazu drona naživo
            main_placeholder.image(web_frame, channels="BGR", use_container_width=True)
            time.sleep(0.01)
            
        cap.release()
