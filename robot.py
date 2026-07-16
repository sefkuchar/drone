import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import os
import urllib.request
from ultralytics import YOLO

st.set_page_config(page_title="UAV Tracking - Diagnostika", layout="wide")
st.title("Diagnostika UAV Streamu")

VIDEO_PATH = "vtest.avi"
VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"

# 1. Diagnostika súboru
if not os.path.exists(VIDEO_PATH):
    st.warning(f"Súbor {VIDEO_PATH} neexistuje, sťahujem...")
    try:
        urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
        st.success("Súbor úspešne stiahnutý!")
    except Exception as e:
        st.error(f"Chyba pri sťahovaní: {e}")

# 2. Skúška otvorenia cez OpenCV
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    st.error("CHYBA: OpenCV nedokáže otvoriť video. Skontroluj, či je súbor poškodený.")
else:
    st.success("OpenCV dokáže otvoriť video - Stream by mal bežať.")
    cap.release()

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
model = load_model()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    # Zjednodušené spracovanie pre maximálnu kompatibilitu
    results = model(img, imgsz=160, verbose=False)
    
    # Kreslenie boxov
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamer
webrtc_streamer(key="uav-stream", mode=WebRtcMode.RECVONLY, 
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_frame_callback=video_frame_callback)
