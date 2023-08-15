# Coral Tracking Pipeline 
On-device object detection, tracking, and behavioral classification

### Additional documentation
[Status, Challenges, Next Steps](https://docs.google.com/document/d/1DR68MHHCU__imOpU9BR54bHQwtUggSJWpgb94p7BdeM/edit?usp=sharing)

## Usage

### main.py (main branch)
Current status of integrating behavioral pipeline on device. 
- Able to detect and track objects (~0.05s per frame)
- Able to generate tracklets for behavioral classification
- Runs out of memory when running inference

### run_inference.py (main branch)
Simple script to run inference on video of previously generated tracklets
- Currently just splits video into sequences of 11 frames and feeds them to behavioral model

### main.py (tracking branch)
Script to run object detection and tracking and generate video with bounding boxes
- Note: utils are not documented here, see main.py in main branch for docstrings

## References
[SORT](https://github.com/abewley/sort)
YOLO utils from sentinel-image-processing
