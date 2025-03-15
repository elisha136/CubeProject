import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def save_validation_plots(results):
    output_dir = os.path.join(results.save_dir, "plots")  
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Extract validation metrics correctly (without calling as functions)
        map50 = results.box.map50  # ✅ Corrected: Access as attribute
        map50_95 = results.box.map  # ✅ Corrected: Access as attribute
        epochs = [1]  # Only one validation run

        if map50 is None or map50_95 is None:
            print("Validation metrics not found. Skipping plot generation.")
            return

        # Plot mAP@50
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [map50], label="mAP@50", color='green', marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("mAP@50")
        plt.title("Mean Average Precision (mAP@50)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "map50_plot.png"))
        plt.close()

        # Plot mAP@50-95
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [map50_95], label="mAP@50-95", color='blue', marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("mAP@50-95")
        plt.title("Mean Average Precision (mAP@50-95)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "map50_95_plot.png"))
        plt.close()

        print(f"✅ Validation plots saved in {output_dir}")

    except AttributeError as e:
        print(f"❌ AttributeError: {e}. Ensure the validation results contain expected metrics.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    # ✅ Windows multiprocessing fix
    results_path = r"C:\Users\maggi\Desktop\elisha\Cubeproject\Dataset\results\cube_detection_exp3\weights\best.pt"
    model = YOLO(results_path)

    # Run validation to generate metrics
    results = model.val(data=r"C:\Users\maggi\Desktop\elisha\Cubeproject\Dataset\data.yaml")

    # Generate validation plots
    save_validation_plots(results)
