import cv2
import numpy as np
import urllib.request
import os
import math
from ultralytics import YOLO

print("[INFO] Štartujem UAV Tracking Systém...")

# ==============================================================================
# 1. PRÍPRAVA VIDEA A UMELEJ INTELIGENCIE
# ==============================================================================
VIDEO_PATH = "test_video.mp4"
VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"

if not os.path.exists(VIDEO_PATH):
    print("[INFO] Sťahujem testovacie video...")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)

print("[INFO] Načítavam YOLO model...")
model = YOLO("yolov8n.pt")
model.to("cpu")

# ==============================================================================
# 2. SPUSTENIE VIDEO STREAMU
# ==============================================================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[CHYBA] Nepodarilo sa načítať video.")
    exit()

print("[INFO] Video spustené. Pre ukončenie stlač klávesu 'q'.")

CENTER_X, CENTER_Y = 640 // 2, 360 // 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Koniec videa.")
        break
        
    # Zmenšenie hlavného obrazu pre rýchlosť
    img = cv2.resize(frame, (640, 360))
    
    # Príprava bočného panela (HUD - Heads Up Display)
    hud = np.zeros((360, 300, 3), dtype=np.uint8)
    
    # YOLO Detekcia
    results = model(img, imgsz=160, verbose=False)
    person_detected = False
    
    for box in results[0].boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # --- MATEMATIKA ---
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            err_x = cx - CENTER_X
            err_y = cy - CENTER_Y
            distance = math.sqrt(err_x**2 + err_y**2)
            reward = (conf * 2.5) - (distance * 0.0015)
            
            # --- HLAVNÁ KAMERA (Kreslenie) ---
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.drawMarker(img, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.line(img, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 1)
            
            # --- BOČNÝ PANEL (Výrez / ROI) ---
            roi = img[max(0, y1-20):min(360, y2+20), max(0, x1-20):min(640, x2+20)]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, (150, 150))
                hud[20:170, 75:225] = roi_resized
                cv2.rectangle(hud, (75, 20), (225, 170), (0, 255, 0), 2)
                cv2.putText(hud, "TAKTICKY VYREZ", (75, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                
            # --- BOČNÝ PANEL (Telemetria text) ---
            cv2.putText(hud, "STAV: ZAMERANE", (15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(hud, f"ISTOTA AI:  {conf*100:.1f} %", (15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"ODCHYLKA:   X:{err_x}px | Y:{err_y}px", (15, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"VZDIALEN.:  {distance:.1f}px", (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"RL SKORE:   {reward:+.4f}", (15, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            break # Sledujeme iba prvý cieľ
            
    if not person_detected:
        cv2.drawMarker(img, (CENTER_X, CENTER_Y), (255, 0, 0), cv2.MARKER_CROSS, 20, 1)
        cv2.putText(hud, "HLADAM CIEL...", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    # ==============================================================================
    # 3. SPOJENIE OBRAZOV A ZOBRAZENIE
    # ==============================================================================
    # Spojíme kameru a čierny panel vedľa seba
    final_frame = np.hstack((img, hud))
    
    # Zobrazenie v natívnom okne operačného systému
    cv2.imshow("UAV Active Tracking Platform", final_frame)
    
    # Ak stlačíš klávesu 'q', program sa vypne
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
