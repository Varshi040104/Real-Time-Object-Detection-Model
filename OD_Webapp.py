from streamlit_webrtc import webrtc_streamer
import av

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Put your YOLO model prediction logic right here!
    # results = model(img)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# This creates an interactive web component with Start/Stop buttons automatically
webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)