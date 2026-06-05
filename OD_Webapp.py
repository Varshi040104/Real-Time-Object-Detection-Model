import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import cv2
import av

# Set page layout
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

# 2. Load the YOLOv8 Model (Cached so it doesn't reload constantly)
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

try:
    model = load_model(model_size)
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. Video Frame Processing Callback Function
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert image frame to OpenCV standard format (BGR)
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLOv8 inference on the frame using the slider's confidence value
    results = model.predict(img, conf=conf_threshold, verbose=False)
    
    # Extract the image with drawn bounding boxes overlaid on top
    annotated_img = results[0].plot()
    
    # Return the processed frame back to the browser interface
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 4. WebRTC Streamer setup (Enhanced STUN servers for robust fallback connectivity)
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

webrtc_streamer(
    key="yolov8-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

webrtc_streamer(
    key="yolov8-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)