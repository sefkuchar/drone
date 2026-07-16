import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import urllib.request
import os
import math
from ultralytics import YOLO

# ==============================================================================
# KONFIGURÁCIA A UI STRÁNKY
# ==============================================================================
st.set_page_config(page_title="Proof of concept", layout="wide")
st.title("Proof of concept - bakalárska práca")
st.markdown("---")

VIDEO_PATH = "vtest.avi"
VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"

if not os.path.exists(VIDEO_PATH):
    with st.spinner("Sťahujem vtest.avi..."):
        urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ==============================================================================
# ENGINE PRE WEBRTC
# ==============================================================================
def create_player():
    from aiortc.contrib.media import MediaPlayer
    return MediaPlayer(VIDEO_PATH)

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
            err_x = cx - CENTER_X
            err_y = cy - CENTER_Y
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
        cv2.putText(hud, "HLADAM CIEL...", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.drawMarker(img, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        
    final_frame = np.hstack((img, hud))
    return av.VideoFrame.from_ndarray(final_frame, format="bgr24")

# ==============================================================================
# STREAMER & DOKUMENTÁCIA
# ==============================================================================
webrtc_streamer(
    key="uav-stream",
    mode=WebRtcMode.RECVONLY,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    player_factory=create_player,
    video_frame_callback=video_frame_callback
)

st.markdown("---")
st.markdown("""
### Status projektu: Proof of Concept
Aktuálne zobrazená aplikácia je funkčný prototyp, ktorý overuje schopnosť neurónovej siete YOLOv8n v reálnom čase detekovať objekt a vypočítať jeho odchýlku od stredu záberu. 

### Logika
1. Vstupný stream: Pomocou knižnice aiortc a WebRTC prijímame video v reálnom čase.
2. Spracovanie obrazu: Model YOLOv8n analyzuje každú snímku. Našou úlohou je v reálnom čase lokalizovať človeka a vrátiť súradnice ohraničujúceho rámčeka.
3. Matematická analýza: Program vypočíta, ako ďaleko je cieľ od stredu záberu kamery.
4. Vizualizácia: Všetky informácie vykresľujeme v reálnom čase do HUD panelu, ktorý simuluje ovládacie rozhranie skutočného dronu.

### Vysvetlenie veličín
V kóde sledujeme metriky, ktoré určujú kvalitu a presnosť sledovania:

* **Odchýlka ($e_x, e_y$):** Predstavuje vzdialenosť cieľa od stredu obrazu v pixeloch. Ak sú hodnoty nulové, cieľ sa nachádza presne na optickej osi kamery. Tieto hodnoty slúžia ako vstup pre PID regulátor na natočenie dronu.
* **Vzdialenosť ($d$):** Ide o Euklidovskú vzdialenosť v dvojrozmernom priestore obrazu definovanú vzorcom: 
$$d = \\sqrt{e_x^2 + e_y^2}$$
* **Istota ($conf$):** Ide o hodnotu od 0 do 1, ktorú vracia model YOLO. Vyjadruje pravdepodobnosť, že detegovaný objekt je skutočne človek.
* **RL Skóre ($R$):** V práci simulujeme odmenu pre agenta, ktorú vypočítavame podľa vzorca: 
$$R = (conf \\cdot 2.5) - (d \\cdot 0.0015)$$
Vysoká istota zvyšuje skóre, zatiaľ čo veľká vzdialenosť od stredu skóre znižuje, čím penalizujeme agenta za to, že cieľ uniká zo záberu.

### Cieľ a zameranie bakalárskej práce
Bakalárska práca nadväzuje na tento prototyp a zameriava sa na tri piliere:

1. **Optimalizácia spracovania videa:** Cieľom je implementovať techniky pre zníženie latencie a efektívne využitie hardvérových prostriedkov, aby systém bežal s čo najvyššou snímkovou frekvenciou aj na palubnom počítači drona.
2. **Implementácia riadiacej slučky (PID regulácia):** Cieľom je navrhnúť systém, ktorý vypočítanú odchýlku premení na riadiace povely.
3. **Autonómna správa stavov:** Cieľom je vytvoriť logiku, ktorá definuje správanie drona v prípade straty cieľa. Systém bude schopný autonómne zahájiť vyhľadávací manéver, cieľ znovu lokalizovať a vrátiť sa do módu sledovania.

### Metodika dosiahnutia cieľov

* **Matematické modelovanie:** Zadefinovanie prenosovej funkcie, ktorá popíše vzťah medzi vizuálnou odchýlkou a potrebným náklonom dronu.
* **Simulácia:** Overenie stabilitu riadiacej slučky v simulovanom prostredí, kde môžeme bezpečne ladiť parametre regulátora pred nasadením na reálny hardvér.
* **Hardvérová implementácia:** Nasadenie systemu na palubný počítač, ktorý zabezpečí spracovanie obrazu a riadenie drona v reálnom čase.

Celkovým cieľom práce je transformovať tento prototyp na ucelený autonómny systém, ktorý dokáže sledovať človeka v dynamických prostredi.
""")
