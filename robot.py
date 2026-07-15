import cv2
import numpy as np
import math
from ultralytics import YOLO

# 1. Načítanie modelu a videa
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("test_video.mp4")

# Nastavenie výstupného videa
width = 640
height = 360
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v je najkompatibilnejší formát
out = cv2.VideoWriter('vysledok_uav.mp4', fourcc, fps, (width, height))

print("Generujem video... Prosím čakaj, kým sa dokončí proces.")

CENTER_X, CENTER_Y = width // 2, height // 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Zmenšenie pre rýchlosť a štandardizáciu
    frame = cv2.resize(frame, (width, height))
    
    results = model(frame, imgsz=160, verbose=False)
    person_detected = False
    
    for box in results[0].boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.4:
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Výpočty
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            err_x = cx - CENTER_X
            err_y = cy - CENTER_Y
            dist = math.sqrt(err_x**2 + err_y**2)
            reward = (conf * 2.5) - (dist * 0.0015)
            
            # Kreslenie
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (CENTER_X, CENTER_Y), (cx, cy), (0, 0, 255), 2)
            
            # Zápis do videa
            cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"E_x: {err_x} E_y: {err_y}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Score: {reward:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            break
            
    out.write(frame)

cap.release()
out.release()
print("Hotovo! Súbor 'vysledok_uav.mp4' je pripravený na odoslanie.")
