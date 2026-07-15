import cv2
import math
from ultralytics import YOLO

# 1. Načítanie modelu a videa
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("test_video.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 2. Nastavenie VideoWriteru (toto vytvorí súbor)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('final_uav_analysis.mp4', fourcc, fps, (width, height))

print("Generujem video... prosím čakaj.")

CENTER_X, CENTER_Y = width // 2, height // 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, imgsz=320, verbose=False)
    
    for box in results[0].boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Výpočty
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            err_x = cx - CENTER_X
            err_y = cy - CENTER_Y
            dist = math.sqrt(err_x**2 + err_y**2)
            reward = (conf * 2.5) - (dist * 0.0015)
            
            # Kreslenie priamo do snímky (to, čo sa uloží)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 2)
            
            # Zápis textu s výpočtami priamo do obrazu
            cv2.putText(frame, f"AI Conf: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Error: X:{err_x} Y:{err_y}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"RL Score: {reward:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            break
            
    out.write(frame)

cap.release()
out.release()
print("Hotovo! Video 'final_uav_analysis.mp4' bolo vytvorené.")
