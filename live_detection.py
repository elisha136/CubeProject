import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load the custom-trained YOLO model
MODEL_PATH = "C:/Users/maggi/Desktop/elisha/Cubeproject/Dataset/results/cube_detection_exp3/weights/best.pt"
model = YOLO(MODEL_PATH)

# Initialize Intel RealSense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # Get frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to NumPy array
        frame = np.asanyarray(color_frame.get_data())

        # Run inference using YOLO
        results = model(frame)

        # Draw detections on the frame
        for result in results:
            for box in result.boxes.xyxy:  # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the live frame with detections
        cv2.imshow("YOLO RealSense Detection", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    pipeline.stop()
    cv2.destroyAllWindows()
