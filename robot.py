import streamlit as st
import cv2
import numpy as np
import time

# ==============================================================================
# KONFIGURÁCIA STRÁNKY
# ==============================================================================
st.set_page_config(
    page_title="UAV Active Tracking Platform",
    layout="wide"
)

st.title("UAV Active Tracking Platform (Zero-Dependency IR Mode)")
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
        deltas_x = [self.history[i][0] - self.history[i-1][0] for i in range(1, len(self.history))]
        deltas_y = [self.history[i][1] - self.history[i-1][1] for i in range(1, len(self.history))]
        return int(self.history[-1][0] + sum(deltas_x)/len(deltas_x)), int(self.history[-1][1] + sum(deltas_y)/len(deltas_y))

    def reset(self):
        self.history.clear()

def evaluate_reward(error_x, error_y, confidence):
    # RL funkcia odmeny: R = (C * w_conf) - (dist * w_dist)
    return (confidence * 2.5) - (np.sqrt(error_x**2 + error_y**2) * 0.0015)

# ==============================================================================
# ROZLOZENIE STRÁNKY (LAYOUT)
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hlavný Optický Stream (Simulovaný IR Senzor)")
    main_placeholder = st.empty()

with col2:
    st.subheader("Taktický Mikro-Výrez (ROI)")
    zoom_placeholder = st.empty()
    
    st.subheader("Telemetria a Výpočty (LIVE)")
    telemetry_placeholder = st.empty()

# Úvodné obrazovky pred spustením
SEARCH_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
cv2.putText(SEARCH_SCREEN, "STANDBY...", (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")
main_placeholder.image(np.zeros((360, 640, 3), dtype=np.uint8), channels="BGR")

start_button = st.button("Spustiť garantovaný 30FPS Stream")

# ==============================================================================
# ŽIVÝ CYKLUS - GENERÁTOR A VYKRESLENIE (BEZ EXTERNÉHO VIDEA)
# ==============================================================================
if start_button:
    CENTER_X, CENTER_Y = 640 // 2, 360 // 2
    predictor = TrajectoryPredictor()
    occlusion_counter = 0
    
    PRED_SCREEN = np.zeros((150, 150, 3), dtype=np.uint8)
    cv2.putText(PRED_SCREEN, "PREDICTING...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Generujeme 300 snímok (cca 10 sekúnd plynulého videa)
    for i in range(300):
        # 1. Vytvorenie syntetického obrazu (Termovízny šum)
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 40
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # 2. Výpočet pohybu cieľa (kombinácia lineárneho posunu a sínusoidy pre reálny pohyb)
        cx = int(80 + i * 1.8)
        cy = int(180 + np.sin(i * 0.1) * 40)
        
        # 3. Simulácia oklúzie (prekážky vo výhľade drona od 120. do 170. snímky)
        is_occluded = (120 <= i <= 170)
        
        if is_occluded:
            # Nakreslenie prekážky (napr. budova)
            cv2.rectangle(frame, (280, 0), (400, 360), (20, 20, 20), -1)
            cv2.putText(frame, "OBSTACLE DETECTED", (285, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            # Spustenie prediktívneho filtra
            prediction = predictor.predict_next()
            if prediction and occlusion_counter < 50:
                occlusion_counter += 1
                px, py = prediction
                predictor.update(px, py)
                
                # Zobrazenie predpovede
                cv2.circle(frame, (px, py), 12, (0, 255, 255), 2)
                cv2.putText(frame, f"PRED ({occlusion_counter})", (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                
                zoom_placeholder.image(PRED_SCREEN, channels="BGR")
                telemetry_placeholder.markdown(f"""
* **Stav UAV:** `PREDIKCIA (ZÁKRYT)` 
* **Snímky naslepo:** `{occlusion_counter} / 50`
* **Odhadovaná poloha:** `X: {px}px | Y: {py}px`
* **RL Reward (Odmena):** `0.0000` (Zrážka pre agenta)
""")
            else:
                predictor.reset()
                zoom_placeholder.image(SEARCH_SCREEN, channels="BGR")
                telemetry_placeholder.markdown("* **Stav UAV:** `STRATA KONTAKTU`")

        else:
            # Nakreslenie cieľa (Tepelná stopa)
            if cx < 640:
                cv2.circle(frame, (cx, cy), 12, (200, 200, 255), -1)
                cv2.rectangle(frame, (cx-20, cy-20), (cx+20, cy+20), (0, 255, 0), 2)
                cv2.putText(frame, "IR_TARGET_01", (cx-35, cy-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.drawMarker(frame, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
                cv2.line(frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
                
                # Matematika
                predictor.update(cx, cy)
                err_x, err_y = cx - CENTER_X, cy - CENTER_Y
                reward = evaluate_reward(err_x, err_y, 0.96)
                
                # Taktický výrez
                crop = frame[max(0, cy-30):min(360, cy+30), max(0, cx-30):min(640, cx+30)]
                if crop.size > 0:
                    zoom_placeholder.image(cv2.resize(crop, (150, 150)), channels="BGR")
                
                # Telemetria
                telemetry_placeholder.markdown(f"""
* **Stav UAV:** `TRACKING ACTIVE (IR)` 
* **Confidence (Istota):** `96.00%` 
* **Regulačná odchýlka $e_x, e_y$:** `X: {err_x}px | Y: {err_y}px`
* **Euklidovská vzdialenosť:** `{np.sqrt(err_x**2 + err_y**2):.2f}px`
* **RL Reward (Odmena):** `{reward:+.4f}`
***
**Výpočet funkcie odmeny pre RL Agenta:**
$$R = (2.5 \\cdot 0.96) - (0.0015 \\cdot {np.sqrt(err_x**2 + err_y**2):.2f}) = {reward:+.4f}$$
""")
                occlusion_counter = 0

        # Vykreslenie priamo na obrazovku bez latencie sťahovania
        main_placeholder.image(frame, channels="BGR", use_container_width=True)
        time.sleep(0.033)  # Simulácia stabilných 30 FPS

    st.success("Testovacia misia úspešne dokončená.")
