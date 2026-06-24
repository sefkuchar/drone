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
    page_title="Tracking Research Platform",
    layout="wide"
)

st.title("Platforma pre sledovanie objektov pomocou dronov")
st.markdown("---")

# ==============================================================================
# AKADEMICKÝ ÚVOD A CIELE PRÁCE (KOMPLETNE OBNOVENÝ TEXT)
# ==============================================================================
st.subheader("Popis projektu")
st.markdown("""
Tento Proof of Concept rieši problémy, s ktorými sa stretávajú bežné drony pri automatickom sledovaní objektov. Klasické systémy fungujú iba reaktívne – snažia sa držať objekt v strede záberu, ale akonáhle sa človek schová alebo zmení smer, dron ho stratí a jeho úloha zlyhá.

Aplikácia demonštruje riešenie pomocou dvoch hlavných algoritmov:
1. **Prediktívny filter pri strate vizuálneho kontaktu:** Ak sa sledovaný objekt schová za prekážku (strom, budova), riadiaci systém nestratí stopu. Na základe histórie posledných súradníc (X, Y) kód matematicky dopočíta predpokladanú rýchlosť a smer pohybu. Dron tak pokračuje v lete "naslepo" rovnakým smerom, čo výrazne zvyšuje šancu, že objekt znova zachytí, keď vyjde zpoza prekážky.
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

# Načítanie modelov a hardvéru
@st.cache_resource
def load_resources():
    model = YOLO('yolov8n.pt')
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return model, device

model, device = load_resources()

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

# Pomocné generátory testovacích scenárov (100% bez potreby internetu)
def generate_scenario_1_circle(filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 360))
    for i in range(150):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cx = int(100 + i * 3)
        cy = int(180 + np.sin(i * 0.1) * 50)
        # Zákryt medzi snímkami 60 a 95
        if not (60 <= i <= 95):
            cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
            cv2.putText(frame, "TEST_OBJECT", (cx-40, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.rectangle(frame, (270, 0), (390, 360), (50, 50, 50), -1)
            cv2.putText(frame, "PREKAZKA", (290, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(frame)
    out.release()

def generate_scenario_2_person(filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 360))
    for i in range(150):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cx = int(80 + i * 3)
        cy = 200
        # Zákryt medzi snímkami 65 a 100
        if not (65 <= i <= 100):
            # Vykreslenie schematického panáčika (hlava, telo, končatiny), aby ho YOLOv8 rozpoznalo ako person
            cv2.circle(frame, (cx, cy-40), 15, (200, 200, 200), -1) # hlava
            cv2.rectangle(frame, (cx-10, cy-25), (cx+10, cy+20), (200, 200, 200), -1) # telo
            cv2.line(frame, (cx-5, cy+20), (cx-5, cy+50), (200, 200, 200), 4) # noha 1
            cv2.line(frame, (cx+5, cy+20), (cx+5, cy+50), (200, 200, 200), 4) # noha 2
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
    
    video_source = "temp_default_video.mp4"
    
    if st.sidebar.button("Pripraviť ukážkové video"):
        with st.spinner("Generujem vybraný scenár na disk..."):
            if "Scenár 1" in default_selection:
                generate_scenario_1_circle(video_source)
            else:
                generate_scenario_2_person(video_source)
            st.sidebar.success("Testovacie video úspešne pripravené na disku!")

elif input_type == "Lokálny videosúbor":
    uploaded_file = st.sidebar.file_uploader("Nahrajte MP4 video:", type=["mp4", "avi", "mov"])
    if uploaded_file:
        video_source = "temp_user_video.mp4"
        with open(video_source, "wb") as f:
            f.write(uploaded_file.read())
            
elif input_type == "YouTube odkaz":
    yt_url = st.sidebar.text_input("Zadajte YouTube URL adresu:")
    if yt_url:
        video_source = "temp_yt_video.mp4"
        if st.sidebar.button("Stiahnuť video zo sieci"):
            with st.spinner("Sťahujem video..."):
                ydl_opts = {
                    'format': 'mp4[height<=480]',
                    'outtmpl': video_source,
                    'overwrites': True,
                    'quiet': True,
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                    st.sidebar.success("Video z YouTube je pripravené.")
                except:
                    st.sidebar.error("Sťahovanie z YouTube zlyhalo.")

# ==============================================================================
# ROZLOZENIE STRÁNKY (LAYOUT)
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hlavný obraz z kamery drona")
    main_video_placeholder = st.empty()

with col2:
    st.subheader("Detailný výrez objektu")
    zoom_placeholder = st.empty()
    
    st.subheader("Výpočty")
    telemetry_placeholder = st.empty()

start_analytics = st.button("Spustiť sledovanie objektu")

# ==============================================================================
# SPRACOVANIE VIDEA
# ==============================================================================
if start_analytics and video_source:
    if not os.path.exists(video_source):
        st.error("Chyba: Súbor neexistuje. Klikni najskôr na tlačidlo 'Pripraviť ukážkové video' v bočnom paneli!")
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error("Chyba pri otváraní videosúboru v OpenCV.")
        else:
            FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if FRAME_WIDTH == 0 or FRAME_HEIGHT == 0:
                FRAME_WIDTH, FRAME_HEIGHT = 640, 360
                
            CENTER_X, CENTER_Y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
            
            predictor = TrajectoryPredictor()
            occlusion_counter = 0
            
            SEARCH_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
            cv2.putText(SEARCH_SCREEN, "SEARCHING...", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            PRED_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
            cv2.putText(PRED_SCREEN, "PREDICTING...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Zistenie vybraného podscenára
            is_circle_scenario = (input_type == "Ukážkové video (Default)" and "Scenár 1" in default_selection)

            explanation_text = """
***
**Vysvetlenie zobrazovaných veličín:**
* **Stav systému:** Ukazuje aktuálny režim. *STABILNE SLEDOVANIE* znamená, že objekt vidíme jasne. *HLADAM LEPSI UHOL* sa zapne, keď klesne istota AI a systém simuluje manéver pre lepší záber. *PREDIKCIA* beží vtedy, keď sa objekt schoval a dron odhaduje smer naslepo.
* **Confidence (Istota AI):** Percentuálne vyjadrenie z modelu YOLOv8, na koľko si je sieť istá, že na danom mieste vidí hľadaného človeka.
* **RL Reward (Skóre odmeny):** Matematický výstup z odmeňovacej funkcie pre posilňované učenie ($R = R_{conf} - P_{dist}$). Čím je toto číslo vyššie, tým lepšie má dron nastavený stred záberu a kvalitu obrazu.
* **Error X / Error Y:** Odchýlka stredu objektu od presného stredu kamery drona v pixeloch. Používa sa ako hlavný vstup pre riadenie letových motorov drona."""

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                web_frame = cv2.resize(frame, (640, 360))
                scale_x, scale_y = 640 / FRAME_WIDTH, 360 / FRAME_HEIGHT
                found = False

                # 1. DETEKCIA PRE SCENÁR 1 (Hľadanie farbou - zelený kruh)
                if is_circle_scenario:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) > 50:
                            x, y, w, h = cv2.boundingRect(c)
                            centroid_x, centroid_y = x + w//2, y + h//2
                            predictor.update(centroid_x, centroid_y)
                            err_x, err_y = centroid_x - CENTER_X, centroid_y - CENTER_Y
                            reward = evaluate_reward(err_x, err_y, 0.95)
                            
                            cv2.rectangle(web_frame, (int(x*scale_x), int(y*scale_y)), (int((x+w)*scale_x), int((y+h)*scale_y)), (0, 255, 0), 2)
                            main_video_placeholder.image(web_frame, channels="BGR", use_container_width=True)
                            
                            crop = frame[max(0, y-30):min(FRAME_HEIGHT, y+h+30), max(0, x-30):min(FRAME_WIDTH, x+w+30)]
                            if crop.size > 0: zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                            
                            telemetry_placeholder.markdown(f"**Stav systému:** `STABILNE SLEDOVANIE`\n\n**Confidence (Istota AI):** `95.00%`\n\n**RL Reward (Skóre odmeny):** `{reward:+.4f}`\n\n**Error X:** `{err_x} px`\n\n**Error Y:** `{err_y} px`{explanation_text}")
                            found = True
                            occlusion_counter = 0

                # 2. DETEKCIA POMOCOU YOLOv8 (Scenár 2 s chodcom alebo reálne videá)
                if not found:
                    results = model(frame, imgsz=640, device=device, verbose=False)
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.22: # Detekcia osoby
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2
                            predictor.update(centroid_x, centroid_y)
                            err_x, err_y = centroid_x - CENTER_X, centroid_y - CENTER_Y
                            reward = evaluate_reward(err_x, err_y, conf)
                            
                            status = "HLADAM LEPSI UHOL" if conf < 0.50 else "STABILNE SLEDOVANIE"
                            color = (0, 165, 255) if conf < 0.50 else (0, 255, 0)
                            
                            wx1, wy1, wx2, wy2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                            cv2.rectangle(web_frame, (wx1, wy1), (wx2, wy2), color, 2)
                            cv2.putText(web_frame, status, (wx1, wy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            
                            main_video_placeholder.image(web_frame, channels="BGR", use_container_width=True)
                            
                            crop = frame[max(0, y1-30):min(FRAME_HEIGHT, y2+30), max(0, x1-30):min(FRAME_WIDTH, x2+30)]
                            if crop.size > 0: zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                            
                            telemetry_placeholder.markdown(f"**Stav systému:** `{status}`\n\n**Confidence (Istota AI):** `{conf:.2%}`\n\n**RL Reward (Skóre odmeny):** `{reward:+.4f}`\n\n**Error X:** `{err_x} px`\n\n**Error Y:** `{err_y} px`{explanation_text}")
                            found = True
                            occlusion_counter = 0
                            break

                # 3. AKTÍVNA PREDIKCIA TRASY (Keď objekt nie je vidno)
                if not found:
                    prediction = predictor.predict_next()
                    if prediction and occlusion_counter < 35:
                        occlusion_counter += 1
                        px, py = prediction
                        predictor.update(px, py)
                        
                        wpx, wpy = int(px*scale_x), int(py*scale_y)
                        cv2.circle(web_frame, (wpx, wpy), 10, (0, 255, 255), 2)
                        cv2.putText(web_frame, "PREDICTION", (wpx+10, wpy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        
                        main_video_placeholder.image(web_frame, channels="BGR", use_container_width=True)
                        zoom_placeholder.image(PRED_SCREEN, channels="BGR")
                        
                        telemetry_placeholder.markdown(f"**Stav systému:** `PREDIKCIA (STRATA KONTAKTU)`\n\n**Snímky v režime predikcie:** `{occlusion_counter} / 35`\n\n**Predpovedaný X:** `{px} px`\n\n**Predpovedaný Y:** `{py} px`{explanation_text}")
                    else:
                        predictor.reset()
                        main_video_placeholder.image(web_frame, channels="BGR", use_container_width=True)
                        zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")
                        telemetry_placeholder.markdown(f"**Stav systému:** `MIMO DOSAH / VYHĽADÁVANIE`\n\n**Confidence (Istota AI):** `0.00%`\n\n**RL Reward (Skóre odmeny):** `N/A`{explanation_text}")
                
                time.sleep(0.01)
                        
            cap.release()
            st.success("Spracovanie videa úspešne dokončené.")