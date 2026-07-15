import streamlit as st
import cv2
import os
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
    model.to("cpu")  # Stabilné CPU pre cloud servery
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
    # R = R_conf - P_dist (Vzorec posilňovaného učenia pre UAV agenta)
    return (confidence * 2.5) - (np.sqrt(error_x**2 + error_y**2) * 0.0015)

# Inicializácia reálneho videa z internetu (Intel Benchmark)
video_source = "intel_people_detection.mp4"
output_video = "analyzed_output.mp4"

if not os.path.exists(video_source):
    st.info("Inicializujem reálne testovacie video z internetu (Intel Benchmark). Prosím, počkajte chvíľu...")
    try:
        url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
        urllib.request.urlretrieve(url, video_source)
        st.success("Reálne video úspešne stiahnuté a pripravené na analýzu!")
        st.rerun()
    except Exception as e:
        st.error(f"Nepodarilo sa stiahnuť video: {e}")

# ==============================================================================
# ROZLOZENIE STRÁNKY (LAYOUT)
# ==============================================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("UAV Optický Stream & Analýza")
    # Hlavné okno na prehrávanie finálneho stabilného videa
    video_placeholder = st.empty()
    
    # Úvodný statický náhľad pred spustením analýzy
    if not os.path.exists(output_video) and os.path.exists(video_source):
        cap_preview = cv2.VideoCapture(video_source)
        ret, frame_preview = cap_preview.read()
        if ret:
            video_placeholder.image(cv2.resize(frame_preview, (640, 360)), channels="BGR", use_container_width=True)
        cap_preview.release()

with col2:
    st.subheader("Telemetrické Parametre & Výpočty")
    telemetry_placeholder = st.empty()
    
    # Statický popis algoritmu
    st.markdown("""
    ### Použité matematické modely:
    
    1. **Smerový Vektor Odchýlky (Error $e_x, e_y$):**
       $$e_x = x_{target} - x_{center}, \\quad e_y = y_{target} - y_{center}$$
       Tento vektor definuje vzdialenosť zameraného objektu od stredu kamery. Využíva ho PID regulátor na korekciu náklonu drona.
       
    2. **Lineárny Prediktor (Smerová extrapolácia):**
       $$\hat{x}_{t+1} = x_t + \\frac{1}{N} \sum_{i=1}^{N} (x_i - x_{i-1})$$
       Odhaduje nasledujúcu polohu objektu pri jeho čiastočnej oklúzii (keď prejde za prekážku).
       
    3. **RL Funkcia Odmeny (Reward Function):**
       $$R = 2.5 \\cdot C - 0.0015 \\cdot \\sqrt{e_x^2 + e_y^2}$$
       *Kde $C$ je Confidence (istota detekcie).* Agent v modeli posilňovaného učenia je motivovaný držať objekt v strede záberu a minimalizovať stratu kontaktu.
    """)

start_button = st.button("Spustiť analýzu a vygenerovať plynulé 30 FPS video")

# ==============================================================================
# SPRACOVANIE A GENEROVANIE PLYNULÉHO MP4 VIDEOSTREAMU
# ==============================================================================
if start_button:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Chyba pri otváraní zdrojového videa.")
    else:
        # Nastavenie výstupného video súboru
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Použitie štandardného kodeku
        out = cv2.VideoWriter(output_video, fourcc, 30.0, (640, 360))
        
        CENTER_X, CENTER_Y = 640 // 2, 360 // 2
        predictor = TrajectoryPredictor()
        occlusion_counter = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Telemetrická štatistika, ktorú vypíšeme na konci
        telemetry_logs = []
        frame_idx = 0
        
        status_text.warning("AI spracováva video, kreslí zameriavacie boxy a prepočítava telemetriu...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            
            web_frame = cv2.resize(frame, (640, 360))
            found = False
            
            # Detekcia YOLOv8
            results = model(web_frame, imgsz=320, verbose=False)
            
            for box in results[0].boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    predictor.update(cx, cy)
                    err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                    reward = evaluate_reward(err_x, err_y, conf)
                    
                    # Vykreslenie priamo do videa, ktoré sa ukladá
                    cv2.rectangle(web_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(web_frame, f"TRACKING PERSON {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    # Vykreslenie kríža uprostred kamery (zameriavač UAV)
                    cv2.drawMarker(web_frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                    cv2.line(web_frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
                    
                    telemetry_logs.append({
                        "Čas (Snímka)": f"{frame_idx}",
                        "Stav UAV": "TRACKING ACTIVE",
                        "Confidence": f"{conf:.2%}",
                        "Odchýlka [X, Y]":f"[{err_x}px, {err_y}px]",
                        "RL Reward (Odmena)": f"{reward:+.4f}"
                    })
                    found = True
                    occlusion_counter = 0
                    break

            # Lineárna predikcia pri výpadku zamerania (oklúzia)
            if not found:
                prediction = predictor.predict_next()
                if prediction and occlusion_counter < 30:
                    occlusion_counter += 1
                    px, py = prediction
                    predictor.update(px, py)
                    
                    cv2.circle(web_frame, (px, py), 10, (0, 255, 255), 2)
                    cv2.putText(web_frame, f"PREDIKCIA ({occlusion_counter})", (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.drawMarker(web_frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                    
                    telemetry_logs.append({
                        "Čas (Snímka)": f"{frame_idx}",
                        "Stav UAV": f"PREDIKCIA ({occlusion_counter}/30)",
                        "Confidence": "0.00%",
                        "Odchýlka [X, Y]": f"[{px - CENTER_X}px, {py - CENTER_Y}px]",
                        "RL Reward (Odmena)": "0.0000"
                    })
                else:
                    predictor.reset()
                    cv2.putText(web_frame, "VYHLADAVANIE...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.drawMarker(web_frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                    
                    telemetry_logs.append({
                        "Čas (Snímka)": f"{frame_idx}",
                        "Stav UAV": "SEARCHING",
                        "Confidence": "0.00%",
                        "Odchýlka [X, Y]": "N/A",
                        "RL Reward (Odmena)": "N/A"
                    })
            
            # Zapíšeme upravenú snímku do nového MP4 videa
            out.write(web_frame)
            
        cap.release()
        out.release()
        
        progress_bar.empty()
        status_text.empty()
        st.success("Analýza úspešne dokončená. Video je pripravené v 30 FPS!")
        
        # HTML5 prehrávač, ktorý prehrá video v perfektnej plynulosti 30 FPS
        video_placeholder.video(output_video)
        
        # Zobrazenie celej telemetrickej tabuľky s presnými prepočtami pre každú sekundu videa
        with telemetry_placeholder.container():
            st.write("#### Nametraná telemetria pre každú snímku:")
            st.dataframe(telemetry_logs, use_container_width=True, height=300)
