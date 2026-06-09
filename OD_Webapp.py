import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import cv2
import av
import asyncio
import sys

# 🔥 FIX: Prevent "RuntimeError: Event loop is closed" on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set page layout to wide
st.set_page_config(page_title="Real-Time YOLOv8 Detector", layout="wide")
st.title("🎯 Real-Time Object Detection WebApp")
st.write("An elegant dashboard powered by Ultralytics YOLOv8 and Streamlit.")

# 1. Sidebar Configurations
st.sidebar.header("⚙️ Model Configuration")

# Select YOLOv8 Model Size
model_size = st.sidebar.selectbox(
    "Choose YOLOv8 Model Size",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
)

# Confidence Threshold Slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.50, 
    step=0.05
)

# 2. Load the YOLOv8 Model Safely (Using session state to avoid reload errors)
if "yolo_model" not in st.session_state or st.session_state.get("current_model_size") != model_size:
    with st.spinner(f"Loading {model_size} model... Please wait."):
        st.session_state["yolo_model"] = YOLO(model_size)
        st.session_state["current_model_size"] = model_size

model = st.session_state["yolo_model"]

# 3. Video Frame Processing Callback Function
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLOv8 inference
    results = model.predict(img, conf=conf_threshold, verbose=False)
    
    # Render bounding boxes onto the frame image
    annotated_img = results[0].plot()
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 4. Secure WebRTC Streamer Setup with Enhanced Fallback STUN Configurations
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]}
        ]
    }
)

# Render the WebRTC layout interface cleanly
st.subheader("📺 Live Feed Window")
st.caption("Click 'START' below to initialize your browser's webcam stream with real-time AI bounding boxes.")

webrtc_streamer(
    key="yolov8-live-detection-dashboard-v2", # Altered key to break older cache conflicts
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)