import streamlit as st
import cv2
import os
import time
import numpy as np

# ==============================================================================
# KONFIGURÁCIA STRÁNKY
# ==============================================================================
st.set_page_config(
    page_title="UAV Active Tracking Platform",
    layout="wide"
)

st.title("UAV Active Tracking Platform")
st.markdown("---")

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
        # Výpočet priemernej rýchlosti zmeny (diferencie) súradníc medzi snímkami
        deltas_x = [self.history[i][0] - self.history[i-1][0] for i in range(1, len(self.history))]
        deltas_y = [self.history[i][1] - self.history[i-1][1] for i in range(1, len(self.history))]
        return int(self.history[-1][0] + sum(deltas_x)/len(deltas_x)), int(self.history[-1][1] + sum(deltas_y)/len(deltas_y))

    def reset(self):
        self.history.clear()

def evaluate_reward(error_x, error_y, confidence):
    # R = R_conf - P_dist (Maximalizácia istoty detekcie s penalizáciou za pixelovú odchýlku od stredu)
    return (confidence * 2.5) - (np.sqrt(error_x**2 + error_y**2) * 0.0015)

# Automatický generátor scenára (spustí sa bez nutnosti klikania na iné tlačidlá)
def ensure_default_video(filename):
    if not os.path.exists(filename):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 360))
        for i in range(100):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cx = int(60 + i * 5.2)
            cy = 180
            
            # Simulácia prekážky (zákryt / oklúzia) medzi snímkami 40 a 65
            if 40 <= i <= 65:
                cv2.rectangle(frame, (250, 0), (370, 360), (60, 60, 60), -1)
                cv2.putText(frame, "PREKAZKA", (275, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Generovanie syntetického cieľa pre stabilné sledovanie
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
                cv2.putText(frame, "TARGET", (cx-25, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            out.write(frame)
        out.release()

# Spustenie automatického generovania hneď pri načítaní
video_source = "temp_default_scene.mp4"
ensure_default_video(video_source)

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
        frame_skip = 2 # Fixne nastavená úspora CPU pre plynulý chod na cloude
        
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

            # Spracovanie obrazu pomocou farebného filtra (Computer Vision)
            hsv = cv2.cvtColor(web_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 40:
                    x, y, w, h = cv2.boundingRect(c)
                    cx, cy = x + w//2, y + h//2
                    predictor.update(cx, cy)
                    
                    err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                    reward = evaluate_reward(err_x, err_y, 0.95)
                    
                    cv2.rectangle(web_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(web_frame, "TRACKING ACTIVE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Výrez ROI
                    crop = web_frame[max(0, y-15):min(360, y+h+15), max(0, x-15):min(640, x+w+15)]
                    if crop.size > 0:
                        zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                        
                    # Aktualizácia telemetrie
                    telemetry_placeholder.markdown(f"""
* **Stav systemu:** `TRACKING ACTIVE` 
* **Confidence (Istota AI):** `95.00%` 
* **RL Reward (Skore odmeny):** `{reward:+.4f}` 
* **Error X/Y (Odchylka od stredu):** `X: {err_x}px | Y: {err_y}px`
""")
                    found = True
                    occlusion_counter = 0

            # Logika prediktívneho filtra pri oklúzii (prekážke)
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
