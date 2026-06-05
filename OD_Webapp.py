import cv2
import numpy as np
import streamlit as st
import time
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time YOLOv8 Detector",
    page_icon="🎯",
    layout="wide",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎯 Real-Time Object Detection WebApp</div>', unsafe_allow_html=True)
st.write("An elegant dashboard powered by Ultralytics YOLOv8 and Streamlit.")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Model Configuration")

# Model selection
model_type = st.sidebar.selectbox("Choose YOLOv8 Model Size", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])

# Confidence Slider (Replaces hardcoded 0.5)
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Start/Stop Buttons
st.sidebar.markdown("---")
start_button = st.sidebar.button("▶️ Start Webcam", type="primary")
stop_button = st.sidebar.button("⏹️ Stop Webcam")

# --- Initialize Model ---
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    model = load_model(model_type)
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- UI Layout ---
# Creating columns: Left for video, Right for metrics/status
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Feed")
    # This empty placeholder will dynamically display our video frames
    frame_placeholder = st.empty() 

with col2:
    st.subheader("Performance Metrics")
    fps_metric = st.empty()
    status_indicator = st.empty()

# --- Main Processing Loop ---
if start_button:
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is accessible
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
    else:
        status_indicator.success("Webcam Active")
        
        prev_frame_time = 0
        
        # Keep running until the 'Stop' button is clicked
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            
            # Update FPS metric in the sidebar/col2
            fps_metric.metric(label="Frames Per Second (FPS)", value=f"{int(fps)}")

            # YOLO Object Detection
            results = model(frame, stream=True, conf=conf_threshold)

            # Plot bounding boxes
            for r in results:
                frame = r.plot()

            # Streamlit requires RGB images (OpenCV reads in BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the image placeholder with the new frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Cleanup when loop exits
        cap.release()
        status_indicator.warning("Webcam Stopped")
        frame_placeholder.empty()