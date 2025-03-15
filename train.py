from ultralytics import YOLO

def main():
    # 1. Load the most accurate YOLOv8 model (yolov8x)
    #    This is a pretrained checkpoint that we'll fine-tune on our dataset.
    model = YOLO('yolov8x.pt')

    # 2. Train the model on your custom dataset
    #    Adjust 'data' path to point to your data.yaml
    #    Adjust epochs, batch, imgsz, etc. as needed.
    results = model.train(
        data=r"C:\Users\adimalaa\OneDrive - NTNU\Desktop\Dataset\data.yaml",
        epochs=50,         # Increase epochs for better results
        batch=8,            # Adjust based on your GPU RAM
        imgsz=640,          # Image size
        name='cube_detection_exp',  # Results folder name
        project=r"C:\Users\adimalaa\OneDrive - NTNU\Desktop\Dataset\results",
        device='device'           # '0' for first GPU, 'cpu' if no GPU
    )

    # 3. Print or inspect training results
    print("Training complete. Results saved to:", results.project_dir)

if __name__ == "__main__":
    main()
