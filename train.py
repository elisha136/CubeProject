from ultralytics import YOLO

def main():
    model = YOLO('yolov8x.pt')

    results = model.train(
        data=r"C:\Users\maggi\Desktop\elisha\Cubeproject\Dataset\data.yaml",
        epochs=50,        
        batch=8,           
        imgsz=640,          
        name='cube_detection_exp',  
        project=r"C:\Users\maggi\Desktop\elisha\Cubeproject\Dataset\results",
        device='cuda'  # Change this if using CPU
    )

    print("Training complete. Results saved to:", results.save_dir)  # FIXED

if __name__ == "__main__":
    main()
