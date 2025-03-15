# CubeProject

## Overview
CubeProject is a deep learning-based object detection system designed to detect and track cubes using a custom-trained YOLOv8 model. The system is optimized to run on an Intel RealSense camera, enabling real-time detection and analysis.

## Features
- **Real-time Object Detection**: Uses YOLOv8 for accurate and fast cube detection.
- **Intel RealSense Integration**: Captures live depth and RGB data for enhanced tracking.
- **Custom-Trained Model**: A specialized model fine-tuned for cube detection.
- **GitHub Integration**: Code and scripts are version-controlled and regularly updated.

## Model Weights
Due to file size restrictions, the trained model weights are hosted externally on OneDrive. You can download them from the links below:

- **Best Model Weights:** [best.pt](https://studntnu-my.sharepoint.com/:u:/r/personal/adimalaa_ntnu_no/Documents/CubeProject_Models/weights/best.pt?csf=1&web=1&e=iIN08Y)
- **Last Model Weights:** [last.pt](https://studntnu-my.sharepoint.com/:u:/r/personal/adimalaa_ntnu_no/Documents/CubeProject_Models/weights/last.pt?csf=1&web=1&e=fNr3RW)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/elisha136/CubeProject.git
   cd CubeProject
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the model weights from the links above and place them in:
   ```
   CubeProject/weights/
   ```

## Running the Model on Live Feed
```bash
python live_detection.py --model weights/best.pt
```

## Contributing
Feel free to submit issues or contribute to the project through pull requests.

## License
This project is licensed under the MIT License.

