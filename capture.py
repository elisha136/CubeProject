import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# Directories
BASE_DIR = r'C:\Users\adimalaa\OneDrive - NTNU\Desktop\Dataset'
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'Annotations')

def get_next_image_number(directory):
    """
    Get the next image number based on existing files in directory.
    Looks for files named 'cube_XXXX.jpg' and returns the highest XXXX found.
    """
    max_num = 0
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.startswith('cube_') and filename.endswith('.jpg'):
                # Extract the integer part, e.g., from 'cube_0012.jpg' -> '0012'
                num_str = filename[5:-4]
                try:
                    num = int(num_str)
                    max_num = max(max_num, num)
                except ValueError:
                    pass
    return max_num

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

print("Directories created (or already exist):")
print(f"Images directory: {IMAGES_DIR}")
print(f"Annotations directory: {ANNOTATIONS_DIR}\n")

# Initialize camera
print("Initializing Intel RealSense D435i camera...")
pipeline = rs.pipeline()
config = rs.config()

try:
    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start pipeline
    pipeline_profile = pipeline.start(config)
    print("Camera initialized successfully.\n")

except Exception as e:
    print(f"Failed to initialize camera: {e}")
    exit(1)

# Get the current highest image number so we can continue counting
current_max = get_next_image_number(IMAGES_DIR)
print(f"Found {current_max} existing images. Next capture will be numbered {current_max + 1}.\n")

print("""Instructions:
- Press 's' to capture and save the current frame
- Press 'q' to quit
- Captured images will be briefly displayed for verification
""")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RealSense Live Feed', color_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            current_max += 1
            image_filename = f"cube_{current_max:04d}.jpg"
            image_path = os.path.join(IMAGES_DIR, image_filename)

            # Save the image
            cv2.imwrite(image_path, color_image)
            print(f"Saved image {image_filename}")

            # Create a corresponding annotation file (empty by default)
            annotation_filename = f"cube_{current_max:04d}.txt"
            annotation_path = os.path.join(ANNOTATIONS_DIR, annotation_filename)
            open(annotation_path, 'a').close()
            print(f"Created annotation file {annotation_filename}")

            # Brief preview
            preview = cv2.resize(color_image, (320, 240))
            cv2.imshow('CAPTURED IMAGE', preview)
            cv2.waitKey(500)  # Show preview for 500ms
            cv2.destroyWindow('CAPTURED IMAGE')

        elif key == ord('q'):
            print("\nCapture session ended by user.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print("\nAll done. Check your Dataset\\Images and Dataset\\Annotations folders!")
