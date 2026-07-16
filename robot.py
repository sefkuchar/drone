import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
import math
from ultralytics import YOLO

st.set_page_config(page_title="UAV Tracking", layout="wide")
st.title("UAV Autonómne Sledovacie Rozhranie")

# 1. Príprava videa
VIDEO_PATH = "vtest.avi"
if not os.path.exists(VIDEO_PATH):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
    urllib.request.urlretrieve(url, VIDEO_PATH)

# 2. Načítanie modelu
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
model = load_model()

# 3. UI
col1, col2 = st.columns([2, 1])
with col1:
    video_placeholder = st.empty()
with col2:
    st.info("Systém pripravený na spracovanie.")

# 4. Stabilný cyklus
if st.button("SPUSTIŤ ANALÝZU"):
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 360))
        results = model(frame, imgsz=160, verbose=False)
        
        # Detekcia a kreslenie
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Konverzia pre zobrazenie
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(img_rgb, channels="RGB")
        
    cap.release()
