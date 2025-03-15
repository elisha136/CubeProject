import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8x.pt')

    # Train the model on the custom dataset
    results = model.train(
        data=r"C:\Users\maggi\Desktop\elisha\Cubeproject\Dataset\data.yaml",
        epochs=50,        
        batch=8,           
        imgsz=640,          
        name='cube_detection_exp',  
        project=r"C:\Users\maggi\Desktop\elisha\Cubeproject\Dataset\results",
        device='device'           
    )

    # Save training plots
    save_training_plots(results)

    # Print training completion message
    print("Training complete. Results saved to:", results.project_dir)

def save_training_plots(results):
    """Save training loss and mAP plots to the results directory."""
    output_dir = os.path.join(results.project_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Extract training metrics
    metrics = results.metrics
    epochs = list(range(1, len(metrics['train/loss']) + 1))
    
    if not epochs:
        print("No training metrics found. Skipping plot generation.")
        return

    train_loss = metrics['train/loss']
    val_loss = metrics['val/loss']
    map50 = metrics['metrics/mAP_50']

    # Plot training & validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", color='blue')
    plt.plot(epochs, val_loss, label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    # Plot mAP@50
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, map50, label="mAP@50", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("mAP@50")
    plt.title("Mean Average Precision (mAP@50)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "map_plot.png"))
    plt.close()

    print(f"Plots saved in {output_dir}")

if __name__ == "__main__":
    main()
