import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import urllib.request
import os
import math
from ultralytics import YOLO

# ==============================================================================
# NASTAVENIE STRÁNKY
# ==============================================================================
st.set_page_config(page_title="UAV Tracking - Bakalárska práca", layout="wide")
st.title("UAV Autonómne Sledovacie Rozhranie")

# ==============================================================================
# WEBRTC ENGINE
# ==============================================================================
VIDEO_PATH = "vtest.avi"
VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
if not os.path.exists(VIDEO_PATH):
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
model = load_model()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 360))
    results = model(img, imgsz=160, verbose=False)
    
    hud = np.zeros((360, 300, 3), dtype=np.uint8)
    CENTER_X, CENTER_Y = 320, 180
    
    person_detected = False
    for box in results[0].boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            err_x, err_y = cx - CENTER_X, cy - CENTER_Y
            dist = math.sqrt(err_x**2 + err_y**2)
            reward = (conf * 2.5) - (dist * 0.0015)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(img, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
            cv2.putText(hud, "STAV: ZAMERANE", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(hud, f"ISTOTA: {conf*100:.1f}%", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"ODCHYLKA: X:{err_x} Y:{err_y}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"SKORE: {reward:+.3f}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            break
            
    if not person_detected:
        cv2.putText(hud, "HLADAM...", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    final_frame = np.hstack((img, hud))
    return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

webrtc_streamer(key="uav-stream", mode=WebRtcMode.RECVONLY, 
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_frame_callback=video_frame_callback)

# ==============================================================================
# ODBORNÝ TEXT BAKALÁRSKEJ PRÁCE
# ==============================================================================
st.markdown("---")
st.markdown("""
Tento projekt predstavuje moderný prístup k automatizovanému sledovaniu cieľov pomocou bezpilotných prostriedkov (UAV). V kontexte bakalárskej práce nejde len o "ukazovanie videa", ale o implementáciu systému pre **vizuálnu servovú slučku**.

### 🧠 Čo kód robí (Logika systému)
1. **Vstupný stream:** Pomocou knižnice `aiortc` a `WebRTC` prijímame video v reálnom čase. Toto je kritické, pretože bežné metódy prenosu videa na webe majú vysoké oneskorenie (latenciu).
2. **Spracovanie obrazu (YOLO):** Model YOLOv8n (YOLO - *You Only Look Once*) analyzuje každú snímku. Jeho úlohou je v reálnom čase lokalizovať človeka a vrátiť súradnice ohraničujúceho rámčeka (Bounding Box).
3. **Matematická analýza (HUD):** Program vypočíta, ako ďaleko je cieľ od stredu záberu kamery.
4. **Vizualizácia:** Všetky informácie sa v reálnom čase vykresľujú do tzv. HUD (Heads-Up Display) – virtuálneho prístrojového panela, ktorý simuluje ovládacie rozhranie skutočného dronu.

### 📊 Vysvetlenie kľúčových veličín
* **Odchýlka (Error $e_x, e_y$):** Toto je vzdialenosť cieľa od stredu obrazu v pixeloch. Ak je $e_x = 0$ a $e_y = 0$, cieľ je presne v strede (na optickej osi kamery). Tieto hodnoty by v reálnom systéme slúžili ako vstup pre PID regulátor na natočenie dronu.
* **Vzdialenosť (Distance):** Ide o Euklidovskú vzdialenosť v 2D priestore obrazu definovanú vzorcom $d = \\sqrt{e_x^2 + e_y^2}$. Hovorí nám, ako ďaleko je cieľ od ideálnej pozície.
* **Istota (Confidence):** Hodnota od 0 do 1, ktorú vracia YOLO model. Vyjadruje pravdepodobnosť, že detegovaný objekt je skutočne človek.
* **RL Skóre (Reward):** V bakalárskej práci simulujeme "odmenu" (Reward) pre agenta. Vychádza zo vzorca: $R = (conf \\cdot 2.5) - (distance \\cdot 0.0015)$. Vysoká istota zvyšuje skóre, zatiaľ čo veľká vzdialenosť od stredu skóre znižuje, čím penalizuje agenta za to, že cieľ "uteká" zo záberu.

### 🎓 Cieľ bakalárskej práce
V bakalárskej práci týmto demonštruješ schopnosť prepojiť počítačové videnie s robotikou. Cieľom je navrhnúť systém, ktorý dokáže autonómne identifikovať človeka a udržať ho v strede zorného poľa kamery bez nutnosti manuálneho pilotovania. Ďalším dôležitým aspektom je optimalizácia pre edge computing, kde ukazuješ, že dokážeš prispôsobiť komplexnú neurónovú sieť tak, aby bežala na obmedzenom hardvéri v reálnom čase s vysokou snímkovou frekvenciou. Celý projekt tak slúži ako dôkaz, že rozumieš tomu, ako sa surové dáta z kamery transformujú na reálne fyzikálne hodnoty, ktoré môže autonómny stroj použiť na svoje riadenie.
""")
