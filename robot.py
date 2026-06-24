import streamlit as st
import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
import yt_dlp

# ==============================================================================
# KONFIGURÁCIA STRÁNKY
# ==============================================================================
st.set_page_config(
    page_title="Tracking Research Platform",
    layout="wide"
)

st.title("Platforma pre sledovanie objektov pomocou dronov")
st.markdown("---")

# ==============================================================================
# AKADEMICKÝ ÚVOD A CIELE PRÁCE
# ==============================================================================
st.subheader("Popis projektu")
st.markdown("""
Tento Proof of Concept rieši problémy, s ktorými sa stretávajú bežné drony pri automatickom sledovaní objektov. Klasické systémy fungujú iba reaktívne – snažia sa držať objekt v strede záberu, ale akonáhle sa človek schová alebo zmení smer, dron ho stratí a jeho úloha zlyhá.

Aplikácia demonštruje riešenie pomocou dvoch hlavných algoritmov:
1. **Prediktívny filter pri strate vizuálneho kontaktu:** Ak sa sledovaný objekt schová za prekážku (strom, budova), riadiaci systém nestratí stopu. Na základe histórie posledných súradńic (X, Y) kód matematicky dopočíta predpokladanú rýchlosť a smer pohybu. Dron tak pokračuje v lete "naslepo" rovnakým smerom, čo výrazne zvyšuje šancu, že objekt znova zachytí, keď vyjde zpoza prekážky.
2. **Aktívne vyhľadávanie optimálneho uhla:** Systém nečaká pasívne na to, kým algoritmus úplne stratí cieľ. Neustále vyhodnocuje úroveň istoty (Confidence Score) modelu YOLOv8. Ak kvalita detekcie klesne pod kritickú hranicu (napr. kvôli zlému svetlu alebo otočeniu objektu), systém simuluje letové príkazy na zmenu polohy drona v priestore, aby získal lepší výhľad a stabilizoval detekciu.
""")

st.subheader("Plánovaný postup a ďalší vývoj")
st.markdown("""
Súčasný softvérový prototyp slúži na overenie logiky spracovania obrazu a matematických výpočtov. V ďalšej fáze vývoja sú stanovené tieto ciele:
* **Nasadenie Kalmanovho filtra:** Súčasný prediktor predpokladá, že objekt sa pohybuje stále rovnakou rýchlosťou. Pre presnejšie sledovanie v reálnom svete implementujeme Kalmanov filter, ktorý dokáže odfiltrovať šum z kamery a lepšie reagovať na prudké zmeny rýchlosti či zrýchlenie objektu.
* **Hardvérové testovanie na reálnom drone:** Presunúť výpočty z počítača priamo na palubný hardvér drona. Cieľom je otestovať, ako rýchlo dokáže systém reagovať v reálnom čase a aký vplyv to bude mať na spotrebu drona.
* **Optimalizácia neurónovej siete:** Upraviť a skomprimovať AI model YOLOv8 tak, aby mal menšiu pamäťovú náročnosť a dosahoval vysokú rýchlosť spracovania aj na slabšom palubnom hardvéri drona.
""")
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

# Lokálne generátory fungujúce 100% offline
def generate_scenario_1_circle(filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 360))
    for i in range(120):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cx = int(100 + i * 3.5)
        cy = int(180 + np.sin(i * 0.1) * 50)
        if not (50 <= i <= 85):
            cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
            cv2.putText(frame, "TEST_OBJECT", (cx-40, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.rectangle(frame, (270, 0), (370, 360), (50, 50, 50), -1)
            cv2.putText(frame, "PREKAZKA", (285, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(frame)
    out.release()

def generate_scenario_2_person(filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 360))
    for i in range(120):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cx = int(80 + i * 3.5)
        cy = 200
        if not (55 <= i <= 90):
            cv2.circle(frame, (cx, cy-40), 15, (200, 200, 200), -1)
            cv2.rectangle(frame, (cx-10, cy-25), (cx+10, cy+20), (200, 200, 200), -1)
            cv2.line(frame, (cx-5, cy+20), (cx-5, cy+50), (200, 200, 200), 4)
            cv2.line(frame, (cx+5, cy+20), (cx+5, cy+50), (200, 200, 200), 4)
        else:
            cv2.rectangle(frame, (270, 0), (380, 360), (60, 60, 60), -1)
            cv2.putText(frame, "BUDOVA", (295, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(frame)
    out.release()

# ==============================================================================
# OVLÁDACIE ROZHRANIE (SIDEBAR)
# ==============================================================================
st.sidebar.header("Konfigurácia vstupov")
input_type = st.sidebar.radio(
    "Zdroj videa:", 
    ["Ukážkové video (Default)", "Lokálny videosúbor", "YouTube odkaz"]
)

video_source = None

if input_type == "Ukážkové video (Default)":
    default_selection = st.sidebar.selectbox(
        "Vyberte testovací scenár:",
        ["Scenár 1: Matematický test predikcie (Zelený kruh za stenou)", 
         "Scenár 2: Generovať test s človekom (Syntetický chodec)"]
    )
    video_source = "input_video.mp4"
    if st.sidebar.button("Pripraviť ukážkové video"):
        if "Scenár 1" in default_selection:
            generate_scenario_1_circle(video_source)
        else:
            generate_scenario_2_person(video_source)
        st.sidebar.success("Súbor pripravený!")

elif input_type == "Lokálny videosúbor":
    uploaded_file = st.sidebar.file_uploader("Nahrajte MP4 video:", type=["mp4", "avi", "mov"])
    if uploaded_file:
        video_source = "input_video.mp4"
        with open(video_source, "wb") as f:
            f.write(uploaded_file.read())
            
elif input_type == "YouTube odkaz":
    yt_url = st.sidebar.text_input("Zadajte YouTube URL adresu:")
    if yt_url:
        video_source = "input_video.mp4"
        if st.sidebar.button("Stiahnuť a analyzovať z YT"):
            with st.spinner("Sťahujem video z YouTube (môže to chvíľu trvať)..."):
                ydl_opts = {
                    'format': 'mp4[height<=360]', # Nízke rozlíšenie pre slabé cloud servery
                    'outtmpl': video_source,
                    'overwrites': True,
                    'quiet': True,
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                    st.sidebar.success("Video úspešne stiahnuté.")
                except Exception as e:
                    st.sidebar.error(f"Sťahovanie zlyhalo: {e}")

# ==============================================================================
# SPRACOVANIE VIDEA (ZÁPIS DO SÚBORU - PRE 30 FPS WEB PREHRÁVANIE)
# ==============================================================================
if st.button("Spustiť a vyhodnotiť analýzu"):
    if not video_source or not os.path.exists(video_source):
        st.error("Chyba: Najskôr priprav video/nahraj súbor v bočnom paneli!")
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Nepodarilo sa otvoriť video.")
        else:
            # H264 kodek je klúčový – jedine ten vedia prehrávače v prehliadači spustiť
            output_path = "output_processed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            
            # Zistenie rozmerov
            FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if FRAME_WIDTH == 0: FRAME_WIDTH, FRAME_HEIGHT = 640, 360
            
            out_writer = cv2.VideoWriter(output_path, fourcc, 25.0, (640, 360))
            
            CENTER_X, CENTER_Y = 640 // 2, 360 // 2
            predictor = TrajectoryPredictor()
            occlusion_counter = 0
            
            is_circle_scenario = (input_type == "Ukážkové video (Default)" and "Scenár 1" in default_selection)
            
            # Streamlit Progress bar na indikáciu spracovania
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0: total_frames = 120
            
            current_frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                current_frame_idx += 1
                progress_bar.progress(min(current_frame_idx / total_frames, 1.0))
                status_text.text(f"AI spracováva video na serveri... Snímka {current_frame_idx}/{total_frames}")
                
                web_frame = cv2.resize(frame, (640, 360))
                found = False

                # 1. Hľadanie farbou (Scenár 1)
                if is_circle_scenario:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) > 50:
                            x, y, w, h = cv2.boundingRect(c)
                            cx, cy = x + w//2, y + h//2
                            predictor.update(cx, cy)
                            cv2.rectangle(web_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(web_frame, "TRACKING", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            found = True
                            occlusion_counter = 0

                # 2. YOLOv8 hľadanie
                if not found:
                    results = model(web_frame, imgsz=320, verbose=False) # Zmenšené imgsz kvôli rýchlosti na CPU
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.20:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            predictor.update(cx, cy)
                            cv2.rectangle(web_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(web_frame, "STABILNE SLEDOVANIE", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            found = True
                            occlusion_counter = 0
                            break

                # 3. Predikcia
                if not found:
                    prediction = predictor.predict_next()
                    if prediction and occlusion_counter < 35:
                        occlusion_counter += 1
                        px, py = prediction
                        predictor.update(px, py)
                        cv2.circle(web_frame, (px, py), 8, (0, 255, 255), 2)
                        cv2.putText(web_frame, f"PREDIKCIA ({occlusion_counter})", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    else:
                        predictor.reset()
                        cv2.putText(web_frame, "HLADAM...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Zápis upravenej snímky do nového súboru
                out_writer.write(web_frame)
                
            cap.release()
            out_writer.release()
            
            status_text.success("Analýza dokončená! Video bolo úspešne vygenerované.")
            
            # Zobrazenie hotového plynulého výsledku v 30 FPS prehrávači
            st.subheader("Výsledné video z misie drona")
            st.video(output_path)
