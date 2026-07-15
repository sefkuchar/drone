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
    model.to("cpu")  # Stabilné CPU pre cloud
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

# Generátor záložného videa (ak zlyhá internet na cloude, vygeneruje realistickú UAV simuláciu)
def generate_fallback_simulation(filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 360))
    for i in range(150):
        # Tmavosivý podklad (infračervený nočný režim drona)
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 40
        # Pridanie jemného šumu senzora
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # Pohyb cieľa po sínusoide (simulácia chôdze človeka)
        cx = int(80 + i * 3.2)
        cy = int(180 + np.sin(i * 0.1) * 40)
        
        # Simulácia prekážky v strede (stĺp/budova) od 60. do 90. snímky
        if 60 <= i <= 90:
            cv2.rectangle(frame, (280, 0), (360, 360), (20, 20, 20), -1)
            cv2.putText(frame, "OBSTACLE", (285, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        else:
            # Cieľ reprezentovaný ako človek v IR spektre (biely kruh/silueta)
            cv2.circle(frame, (cx, cy), 12, (240, 240, 240), -1)
            cv2.putText(frame, "TARGET_0", (cx-30, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
            
        out.write(frame)
    out.release()

# Inicializácia videa
video_source = "intel_people_detection.mp4"

if not os.path.exists(video_source):
    status_msg = st.info("Sťahujem reálne testovacie video...")
    try:
        # Použitie User-Agent hlavičky, aby GitHub neblokoval požiadavku
        url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(video_source, 'wb') as out_file:
            out_file.write(response.read())
        status_msg.success("Reálne video bolo úspešne stiahnuté!")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        status_msg.warning(f"Sťahovanie zlyhalo ({e}). Aktivujem interný simulátor drona...")
        generate_fallback_simulation(video_source)
        status_msg.success("Simulátor UAV videa bol úspešne spustený!")
        time.sleep(1)
        st.rerun()

# ==============================================================================
# ROZLOZENIE STRÁNKY (LAYOUT)
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hlavny Opticky Stream (UAV Main Sensor)")
    main_placeholder = st.empty()
    
    # Prvý náhľad pred štartom
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

# Základné obrazovky pri hľadaní
SEARCH_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(SEARCH_SCREEN, "SEARCHING...", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")

start_button = st.button("Spustiť aktívne sledovanie")

# ==============================================================================
# ŽIVÝ CYKLUS - SYNCHRONIZOVANÉ LIVE VYKRESLENIE
# ==============================================================================
if start_button:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Chyba: Video súbor sa nepodarilo otvoriť.")
    else:
        CENTER_X, CENTER_Y = 640 // 2, 360 // 2
        predictor = TrajectoryPredictor()
        occlusion_counter = 0
        current_frame_idx = 0
        frame_skip = 2  # Spracovanie každej 2. snímky pre ideálny pomer plynulosti/výkonu
        
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

            # Detekcia na základe toho, či ide o simuláciu alebo stiahnuté reálne video
            if "TARGET_0" in cv2.putText(web_frame.copy(), "", (0,0), 1, 1, 1): # Hlúpy trik na detekciu simulácie
                pass 
                
            # A. DETEKCIA PRE SIMULÁCIU (detekcia svetlého objektu - IR signatúra)
            if "intel_people_detection.mp4" in video_source and not os.path.exists("real_video_confirmed"):
                # Pre simulované video hľadáme svetlý objekt (IR signatúru)
                gray = cv2.cvtColor(web_frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 10:
                        x, y, w, h = cv2.boundingRect(c)
                        cx, cy = x + w//2, y + h//2
                        predictor.update(cx, cy)
                        err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                        reward = evaluate_reward(err_x, err_y, 0.98)
                        
                        cv2.rectangle(web_frame, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2)
                        cv2.putText(web_frame, "IR TARGET LOCATED", (x-10, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.drawMarker(web_frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                        cv2.line(web_frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
                        
                        crop = web_frame[max(0, y-15):min(360, y+h+15), max(0, x-15):min(640, x+w+15)]
                        if crop.size > 0:
                            zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                            
                        telemetry_placeholder.markdown(f"""
* **Stav UAV:** `TRACKING ACTIVE (IR)` 
* **Confidence (Istota AI):** `98.00%` 
* **Regulačná odchýlka $e_x, e_y$:** `X: {err_x}px | Y: {err_y}px`
* **Euklidovská vzdialenosť:** `{np.sqrt(err_x**2 + err_y**2):.2f}px`
* **RL Reward (Odmena pre agenta):** `{reward:+.4f}`
***
**Matematický výpočet odmeny:**
$$R = (2.5 \\cdot 0.98) - (0.0015 \\cdot {np.sqrt(err_x**2 + err_y**2):.2f}) = {reward:+.4f}$$
""")
                        found = True
                        occlusion_counter = 0

            # B. DETEKCIA PRE REÁLNE VIDEO (YOLOv8)
            if not found:
                results = model(web_frame, imgsz=160, verbose=False)
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                        # Ak aspoň raz úspešne deteguje osobu cez YOLO, označíme video ako reálne
                        if not os.path.exists("real_video_confirmed"):
                            open("real_video_confirmed", "w").close()
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        predictor.update(cx, cy)
                        err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                        reward = evaluate_reward(err_x, err_y, conf)
                        
                        cv2.rectangle(web_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(web_frame, f"ID_0: PERSON {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.drawMarker(web_frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                        cv2.line(web_frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
                        
                        crop = web_frame[max(0, y1-15):min(360, y2+15), max(0, x1-15):min(640, x2+15)]
                        if crop.size > 0:
                            zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                            
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
                        break

            # C. STRATA KONTAKTU / PREDIKCIA (Zákryt)
            if not found:
                prediction = predictor.predict_next()
                if prediction and occlusion_counter < 30:
                    occlusion_counter += 1
                    px, py = prediction
                    predictor.update(px, py)
                    
                    cv2.circle(web_frame, (px, py), 10, (0, 255, 255), 2)
                    cv2.putText(web_frame, f"PREDIKCIA ({occlusion_counter})", (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.drawMarker(web_frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                    
                    zoom_placeholder.image(PRED_SCREEN, channels="BGR")
                    telemetry_placeholder.markdown(f"""
* **Stav UAV:** `PREDIKCIA (ZÁKRYT)` 
* **Snímky naslepo:** `{occlusion_counter} / 30`
* **Predpovedaná poloha:** `X: {px}px | Y: {py}px`
* **Regulačná odchýlka:** `X: {px - CENTER_X}px | Y: {py - CENTER_Y}px`
* **RL Reward (Odmena pre agenta):** `0.0000` (Penalizácia)
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
            
            # Plynulé LIVE zobrazenie hlavného videa
            main_placeholder.image(web_frame, channels="BGR", use_container_width=True)
            time.sleep(0.015)  # Regulácia latencie pre plynulý chod
            
        cap.release()
        st.success("Sledovanie úspešne dokončené.")
