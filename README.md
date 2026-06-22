# Number Plate Detection - R-CNN 🚗
Real-time number plate detection using R-CNN, Selective Search, CNN, 
and Non-Maximum Suppression built with TensorFlow and OpenCV.
## How It Works
1. Selective Search generates Region of Interest (ROI)
2. CNN model classifies each ROI as plate or no plate
3. IoU filters correct bounding boxes
4. NMS removes duplicate boxes
5. Best box is selected and drawn
## Technologies Used
- Python 3
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
## Dataset
- Number plate images with annotations (CSV)
- Binary classification: Plate / No Plate
## Files
- `nms_iou.py` — Non-Maximum Suppression and IoU functions
- `read_data.py` — Data loading and preprocessing
- `RCNN.py` — Main model training and detection
## Author
Laraib Hafeez - BS Artificial Intelligence Student
