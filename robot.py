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

st.title("UAV Active Tracking Platform")
st.markdown("---")

# Načítanie modelov a hardvéru (s limitáciou na CPU pre cloud servery)
@st.cache_resource
def load_resources():
    model = YOLO('yolov8n.pt')
    model.to("cpu")  # Vynútené stabilné CPU pre cloud
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

# Lokálny generátor testovacieho videa (100% funguje offline)
def generate_default_scenario(filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 360))
    for i in range(90):  
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cx = int(80 + i * 5)
        cy = 200
        if not (40 <= i <= 65):
            cv2.circle(frame, (cx, cy-40), 15, (200, 200, 200), -1)
            cv2.rectangle(frame, (cx-10, cy-25), (cx+10, cy+20), (200, 200, 200), -1)
            cv2.line(frame, (cx-5, cy+20), (cx-5, cy+50), (200, 200, 200), 4)
            cv2.line(frame, (cx+5, cy+20), (cx+5, cy+50), (200, 200, 200), 4)
        else:
            cv2.rectangle(frame, (260, 0), (380, 360), (60, 60, 60), -1)
            cv2.putText(frame, "PREKAZKA", (280, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(frame)
    out.release()

# ==============================================================================
# OVLÁDACÍ PANEL I VSTUPY
# ==============================================================================
st.sidebar.header("Ovladaci Panel i Vstupy")

input_type = st.sidebar.radio(
    "Vyberte typ vstupu:", 
    ["Ukazkove video (Default)", "Nahrat vlastne video", "YouTube Link"]
)

video_source = None

if input_type == "Ukazkove video (Default)":
    video_source = "temp_default_scene.mp4"
    if st.sidebar.button("Pripravit defaultne video"):
        with st.spinner("Generujem integrovanú simuláciu..."):
            generate_default_scenario(video_source)
        st.sidebar.success("Defaultné video bolo pripravené.")

elif input_type == "Nahrat vlastne video":
    uploaded_file = st.sidebar.file_uploader("Nahrajte MP4 video:", type=["mp4", "avi", "mov"])
    if uploaded_file:
        video_source = "user_input_video.mp4"
        with open(video_source, "wb") as f:
            f.write(uploaded_file.read())
            
elif input_type == "YouTube Link":
    yt_url = st.sidebar.text_input("Zadajte YouTube URL adresu:", value="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    if yt_url:
        video_source = "youtube_downloaded.mp4"
        if st.sidebar.button("Stiahnut a analyzovat YouTube stream"):
            with st.spinner("Sťahujem video z YouTube..."):
                ydl_opts = {
                    'format': 'mp4[height<=360]', 
                    'outtmpl': video_source,
                    'overwrites': True,
                    'quiet': True,
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                    st.sidebar.success("Video z YouTube úspešne stiahnuté.")
                except Exception as e:
                    st.sidebar.error(f"Sťahovanie zlyhalo: {e}")

st.sidebar.markdown("---")
st.sidebar.header("Nastavenia filtrov")
frame_skip = st.sidebar.slider("Frame Skip (Uspora CPU)", min_value=1, max_value=10, value=2)

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

explanation_text = """
***
**Vysvetlenie telemetrickych velicin:**

* **Stav systemu:** Regionalny rezim riadenia UAV. *TRACKING ACTIVE* znamena stabilne zameranie. *ACTIVE VISION* deteguje pokles istoty a simuluje orbitalny manever pre lepsi uhol pohladu. *PREDIKCIA* znamena, ze objekt je skryty a dron leti podla zotrvacnosti linearneho filtra.
* **Confidence (Istota AI):** Vyjadruje percentualnu istotu detekcneho modelu YOLOv8, ze dany objekt je naozaj hladany clovek.
* **RL Reward (Skore odmeny):** Matematicky vystup z odmenovacej funkcie pre posilnovane ucenie ($R = R_{conf} - P_{dist}$).
* **Error X/Y (Odchylka od stredu):** Diferencia objektu od optickeho stredu senzora v pixeloch.
"""
st.markdown(explanation_text)

start_button = st.button("Spustiť spracovanie a vygenerovať plynulý stream")

# ==============================================================================
# INTELIGENTNÉ SPRACOVANIE S UKLADANÍM DO BUFFERA
# ==============================================================================
if start_button:
    if not video_source or not os.path.exists(video_source):
        st.error("Chyba: Vstupný zdroj nie je pripravený. Klikni najskôr na tlačidlo v bočnom paneli!")
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Chyba pri inicializácii video streamu.")
        else:
            CENTER_X, CENTER_Y = 640 // 2, 360 // 2
            predictor = TrajectoryPredictor()
            occlusion_counter = 0
            current_frame_idx = 0
            
            # Zoznamy, kam si uložíme vygenerované dáta pred spustením prehrávača
            frames_buffer = []
            crops_buffer = []
            telemetry_buffer = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0: total_frames = 90

            # 1. KROK: Rýchla analýza na pozadí a naplnenie pamäte
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                current_frame_idx += 1
                if current_frame_idx % frame_skip != 0:
                    continue
                
                progress_bar.progress(min(current_frame_idx / total_frames, 1.0))
                status_text.text(f"AI analyzuje video na serveri... Snímka {current_frame_idx} / {total_frames}")
                
                web_frame = cv2.resize(frame, (640, 360))
                found = False
                
                # Defaultné nastavenia pre prípad, že objekt neexistuje
                crop_img = SEARCH_SCREEN.copy()
                tel_data = {
                    "status": "VYHLADAVANIE / MIMO DOSAH",
                    "conf": 0.0,
                    "reward": None,
                    "err_x": 0,
                    "err_y": 0,
                    "frames_blind": 0
                }

                # YOLOv8 Detekcia osôb
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
                        
                        # Príprava ROI výrezu do pamäte
                        c_raw = web_frame[max(0, y1-15):min(360, y2+15), max(0, x1-15):min(640, x2+15)]
                        if c_raw.size > 0:
                            crop_img = cv2.resize(c_raw, (150, 150))
                        
                        # Príprava telemetrických dát
                        tel_data = {
                            "status": status,
                            "conf": conf,
                            "reward": reward,
                            "err_x": err_x,
                            "err_y": err_y,
                            "frames_blind": 0
                        }
                        
                        found = True
                        occlusion_counter = 0
                        break

                # Lineárna predikcia pri strate kontaktu
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
                        crop_img = PRED_SCREEN
                        
                        tel_data = {
                            "status": "PREDIKCIA (STRATA KONTAKTU)",
                            "conf": 0.0,
                            "reward": None,
                            "err_x": px,
                            "err_y": py,
                            "frames_blind": occlusion_counter
                        }
                    else:
                        predictor.reset()

                # Uloženie všetkého do bufferov (preklápame BGR na RGB pre Streamlit)
                frames_buffer.append(cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB))
                crops_buffer.append(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                telemetry_buffer.append(tel_data)
                
            cap.release()
            progress_bar.empty()
            status_text.empty()
            
            # 2. KROK: ULTRA-PLYnulé PREHRÁVANIE ZO ZDIEĽANEJ PAMÄTE Servera
            if frames_buffer:
                st.success("Analýza dokončená! Spúšťam synchronizovaný taktický stream.")
                
                # Prechádzame predpripravené dáta z pamäte, čo eliminujem sieťové oneskorenie
                for idx in range(len(frames_buffer)):
                    # Vykreslenie hlavného obrazu
                    main_placeholder.image(frames_buffer[idx], use_container_width=True)
                    
                    # Vykreslenie výrezu (ROI)
                    zoom_placeholder.image(crops_buffer[idx])
                    
                    # Vykreslenie dynamických textových polí v pravom stĺpci
                    t = telemetry_buffer[idx]
                    if t["status"] == "PREDIKCIA (STRATA KONTAKTU)":
                        telemetry_placeholder.markdown(f"""
* **Stav systemu:** `PREDIKCIA (STRATA KONTAKTU)` 
* **Snímky naslepo:** `{t['frames_blind']} / 30`
* **Predpovedaný X/Y:** `X: {t['err_x']}px | Y: {t['err_y']}px`
""")
                    elif "VYHLADAVANIE" in t["status"]:
                        telemetry_placeholder.markdown("""
* **Stav systemu:** `VYHLADAVANIE / MIMO DOSAH` 
* **Confidence (Istota AI):** `0.00%`
* **RL Reward (Skore odmeny):** `N/A`
""")
                    else:
                        telemetry_placeholder.markdown(f"""
* **Stav systemu:** `{t['status']}` 
* **Confidence (Istota AI):** `{t['conf']:.2%}` 
* **RL Reward (Skore odmeny):** `{t['reward']:+.4f}` 
* **Error X/Y (Odchylka od stredu):** `X: {t['err_x']}px | Y: {t['err_y']}px`
""")
                    
                    # Riadenie rýchlosti vykresľovania na webe (cca 25-30 FPS)
                    time.sleep(0.035)
                    
                st.balloons()
            else:
                st.error("Chyba: Nepodarilo sa získať žiadne dáta z videa.")
