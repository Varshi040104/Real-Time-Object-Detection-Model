from ultralytics import YOLO
import cv2
import time  # Import time for FPS calculation

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)

    # Only show objects if YOLO is more than 50% sure
    results = model(frame, stream=True, conf=0.5)

    for r in results:
        frame = r.plot()

    # Put the FPS text on the frame
    cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("YOLO Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
