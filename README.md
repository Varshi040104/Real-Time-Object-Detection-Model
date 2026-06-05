# 🎯 Real-Time Object Detection Dashboard (YOLOv8 + Streamlit)

Welcome to the **Real-Time Object Detection Web Application**! This project combines the blistering speed of **Ultralytics YOLOv8** with the beautiful frontend capabilities of **Streamlit** to create an interactive, live-streaming computer vision dashboard.

---

## 🕹️ Interactive Features & WebApp Walkthrough

When you launch the web application, you will see a beautifully structured dashboard split into an interactive sidebar and a main content area. Here is how to use it:

### ⚙️ 1. Control Center (The Sidebar)
* **Model Size Selection (`Selectbox`):** Choose your AI model size on the fly! You can switch between `yolov8n.pt` (Nano - ultra-fast, lightweight) and larger variants depending on your hardware capability.
* **Confidence Threshold (`Slider`):** Dynamically control how confident the model needs to be before it draws a box around an object. Move it up (e.g., `0.70`) to reduce false positives, or down (e.g., `0.30`) to detect more objects.
* **▶️ Start Webcam Button:** Initializes your local webcam feed and starts streaming individual frames directly into the AI pipeline.
* **⏹️ Stop Webcam Button:** Gracefully disconnects from your webcam source and halts the video processing loop safely.

### 📺 2. Workspace Layout (The Main Dashboard)
* **🖼️ Live Feed Window:** This is where the magic happens. Your webcam feed is displayed here in high definition, with YOLOv8 dynamically overlaying colorful bounding boxes and percentage labels around detected objects in real-time.
* **📊 Performance Metrics Box:** Located neatly to the side, this column dynamically updates to show your live **Frames Per Second (FPS)**. It monitors exactly how fast the AI model is processing your video frames.

---

## 💻 How to Run the WebApp Locally

Follow these quick steps to launch the app on your local machine:

### 1. Activate Your Environment
Open your terminal inside the project folder and activate your virtual environment:
```bash
.venv\Scripts\activate

#### 2. To stop stop the camera directly in the Webapp
or you can use the CTRL + C to stop the camera directly from terminal  