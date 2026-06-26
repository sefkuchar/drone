import streamlit as st
import cv2
import os
import time
import torch
import numpy as np
from ultralytics import YOLO
import yt_dlp

# ==============================================================================
# KONFIGURÁCIA STRÁNKY
# ==============================================================================
st.set_page_config(
    page_title="UAV Active Tracking Platform",
    layout="wide"
)

# Názov aplikácie podľa tvojho dizajnu
st.title("UAV Active Tracking Platform")
st.markdown("---")

# Načítanie modelov a hardvéru (s limitáciou na CPU pre cloud servery)
@st.cache_resource
def load_resources():
    model = YOLO('yolov8n.pt')
    model.to("cpu")  # Cloud servery nemajú GPU, vynútime stabilné CPU
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

# ==============================================================================
# OVLÁDACÍ PANEL I VSTUPY (Presne podľa tvojho screenshotu)
# ==============================================================================
st.sidebar.header("Ovladaci Panel i Vstupy")

input_type = st.sidebar.radio(
    "Vyberte typ vstupu:", 
    ["Nahrat vlastne video", "YouTube Link"]
)

video_source = None

if input_type == "Nahrat vlastne video":
    uploaded_file = st.sidebar.file_uploader("Nahrajte MP4 video:", type=["mp4", "avi", "mov"])
    if uploaded_file:
        video_source = "input_video.mp4"
        with open(video_source, "wb") as f:
            f.write(uploaded_file.read())
            
elif input_type == "YouTube Link":
    yt_url = st.sidebar.text_input("Zadajte YouTube URL adresu:", value="https://www.youtube.com/shorts/YdpN9")
    if yt_url:
        video_source = "input_video.mp4"
        # Tlačidlo presne podľa teba
        if st.sidebar.button("Stiahnut a analyzovat YouTube stream"):
            with st.spinner("Sťahujem video z YouTube..."):
                ydl_opts = {
                    'format': 'mp4[height=360]/mp4[height<=480]', # Nízke rozlíšenie pre plynulosť cloudu
                    'outtmpl': video_source,
                    'overwrites': True,
                    'quiet': True,
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                    st.sidebar.success("Video z YouTube stiahnuté.")
                except Exception as e:
                    st.sidebar.error(f"Chyba sťahovania: {e}")

st.sidebar.markdown("---")
st.sidebar.header("Nastavenia filtrov")
frame_skip = st.sidebar.slider("Frame Skip (Uspora CPU)", min_value=1, max_value=10, value=3)

# ==============================================================================
# ROZLOZENIE STRÁNKY (LAYOUT - PREDPRIPRAVENÉ PODĽA TVOJHO VIZUÁLU)
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

# Vykreslenie východiskových prázdnych stavov, aby stránka nevyzerala prázdna
SEARCH_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(SEARCH_SCREEN, "SEARCHING...", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")

# Texty na spodku pravej strany
explanation_text = """
**Vysvetlenie telemetrickych velicin:**

* **Stav systemu:** Aktualny rezim riadenia UAV. *TRACKING ACTIVE* znamena stabilne zameranie. *ACTIVE VISION* deteguje pokles istoty a simuluje orbitalny manever pre lepsi uhol pohladu. *PREDIKCIA* znamena, ze objekt je skryty a dron leti podla zotrvacnosti linearneho filtra.
* **Confidence (Istota AI):** Vyjadruje percentualnu istotu detekcneho modelu YOLOv8.
"""
st.markdown("---")

# ==============================================================================
# SPRACOVANIE VIDEA A ZÁPIS (ZABRAŇUJE CHYBE MEDIAFILESTORAGEERROR)
# ==============================================================================
if st.button("Spustiť analýzu a generovanie výstupu"):
    if not video_source or not os.path.exists(video_source):
        st.error("Chyba: Najskôr pridaj/stiahni video v bočnom paneli!")
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Nepodarilo sa otvoriť vstupné video.")
        else:
            output_path = "output_processed.mp4"
            
            # Odstránenie starého súboru ak existuje
            if os.path.exists(output_path):
                os.remove(output_path)
                
            # Použitie vysoko kompatibilného MJPEG/MP4V pre bezpečné ukladanie na cloude
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 360))
            
            FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if FRAME_WIDTH == 0: FRAME_WIDTH, FRAME_HEIGHT = 640, 360
            
            CENTER_X, CENTER_Y = 640 // 2, 360 // 2
            predictor = TrajectoryPredictor()
            occlusion_counter = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0: total_frames = 100
            
            current_frame_idx = 0
            frames_written = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                current_frame_idx += 1
                
                # Rešpektovanie posuvníka Frame Skip na úsporu CPU
                if current_frame_idx % frame_skip != 0:
                    continue
                
                progress_bar.progress(min(current_frame_idx / total_frames, 1.0))
                status_text.text(f"Spracovávam snímku {current_frame_idx} / {total_frames}...")
                
                web_frame = cv2.resize(frame, (640, 360))
                found = False

                # YOLOv8 hľadanie človeka (trieda 0)
                results = model(web_frame, imgsz=320, verbose=False)
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.20:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        predictor.update(cx, cy)
                        err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                        reward = evaluate_reward(err_x, err_y, conf)
                        
                        status = "ACTIVE VISION" if conf < 0.50 else "TRACKING ACTIVE"
                        color = (0, 165, 255) if conf < 0.50 else (0, 255, 0)
                        
                        cv2.rectangle(web_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(web_frame, status, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Aktualizácia detailného výrezu pre telemetriu
                        crop = web_frame[max(0, y1-15):min(360, y2+15), max(0, x1-15):min(640, x2+15)]
                        if crop.size > 0:
                            zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                        
                        telemetry_placeholder.markdown(f"""
* **Stav systemu:** `{status}` 
* **Confidence (Istota AI):** `{conf:.2%}` 
* **RL Reward (Skore odmeny):** `{reward:+.4f}` 
* **Error X/Y (Odchylka od stredu):** `X: {err_x}px | Y: {err_y}px`
""")
                        found = True
                        occlusion_counter = 0
                        break

                # Predikcia lineárnym filtrom pri strate kontaktu
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
                
                out_writer.write(web_frame)
                frames_written += 1
                
            cap.release()
            out_writer.release()
            
            progress_bar.empty()
            status_text.empty()
            
            # 🔥 KRITICKÁ POISTKA PROTI CHYBE 404/STORAGE ERROR:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                st.success("Spracovanie úspešne dokončené! Video bolo vygenerované nižšie.")
                # Zobrazenie plynulého videa
                main_placeholder.video(output_path)
            else:
                st.error("Chyba: Nepodarilo sa zapísať spracované video na disk servera. Skontrolujte formát vstupného súboru.")
                
        st.sidebar.markdown(explanation_text)
