# Airplane Tracking System

⚠️ IMPORTANT NOTICE

The dataset used in this project (videos and labeled images) is private and cannot be shared publicly due to its proprietary nature. Please note that:

Training videos are private
Labeled images are proprietary
Dataset access requires authorization

## Overview
A sophisticated computer vision system leveraging YOLO (You Only Look Once) object detection to track and monitor aircraft movements and passenger bridge operations at airports. The system provides real-time analytics for aircraft parking duration, bridge connection times, and overall runway occupancy.

## Table of Contents
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Detection Parameters](#detection-parameters)
- [Output Specifications](#output-specifications)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities
- Real-time object detection using YOLO v8
- Multi-object tracking system
- Automated time measurements
- Video processing and analysis
- Custom model training support

### Tracking Metrics
- Total runway occupation time
- Aircraft parking duration
- Passenger bridge connection time
- Movement pattern analysis
- Position stability monitoring

### Visual Output
- Real-time video annotations
- On-screen time displays
- Status indicators
- Movement tracking visualization
- Bridge connection status

## Technical Architecture

### Components
1. **Object Detection Module**
   - YOLO model integration
   - Custom-trained weights
   - Real-time frame processing
   - Object classification

2. **Tracking System**
   - Position monitoring
   - Movement calculation
   - State management
   - Time tracking

3. **Video Processing**
   - Frame capture
   - Annotation overlay
   - Output generation
   - Performance optimization

### Data Flow
```
Video Input → Frame Extraction → YOLO Detection → Position Analysis → State Updates → Video Output
```

## Requirements

### Software Dependencies
```
Python >= 3.8
opencv-python >= 4.5
ultralytics >= 8.0
numpy >= 1.19
roboflow >= 1.0
```

### Hardware Recommendations
- CPU: Multi-core processor (i5/i7 or equivalent)
- RAM: 8GB minimum, 16GB recommended
- GPU: NVIDIA GPU with CUDA support (optional, but recommended)
- Storage: 1GB free space for model and video processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/airplane-tracking-system.git
cd airplane-tracking-system
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO weights:
```bash
python frame.py
```

## Configuration

### Model Configuration
```python
# frame.py
ROBOFLOW_API_KEY = "your_api_key"
PROJECT_WORKSPACE = "your_workspace"
PROJECT_NAME = "your_project"
VERSION = "your_version"
```

### Video Processing Settings
```python
# airplane.py
VIDEO_PATH = "path/to/input/video"
OUTPUT_PATH = "path/to/output/video"
MODEL_PATH = "path/to/model/weights"
```

### Detection Parameters
```python
CONFIDENCE_THRESHOLD = 0.7
FPS_THRESHOLD = 40
WINDOW_DETECTION_INTERVAL = 10
BRIDGE_MOVEMENT_THRESHOLD = 6
```

## Usage

### Basic Operation
1. Configure your environment variables:
```bash
export ROBOFLOW_API_KEY="your_api_key"
```

2. Initialize the model:
```bash
python frame.py
```

3. Run the tracking system:
```bash
python airplane.py
```

### Custom Model Training
1. Prepare your dataset
2. Upload to Roboflow
3. Train model
4. Update weights file

## Detection Parameters

### Aircraft Detection
- Window tracking interval: 10 frames
- Position stability threshold: 3 pixels
- Minimum detection confidence: 0.7
- Movement calculation window: 10 samples

### Bridge Detection
- Movement threshold: 6 pixels
- Connection stability period: 15 seconds
- Minimum confidence score: 0.7
- Sample window size: 10 frames

## Output Specifications

### Video Output
- Format: AVI (XVID codec)
- Resolution: Matches input video
- Frame rate: Matches input video
- Annotations: Real-time overlay

### Console Output
```
Pistteki Toplam Süre: XX.XX saniye
Park Süresi: XX.XX saniye
Köprü Duruş Süresi: XX.XX saniye
```

### Status Messages
- "Uçak iniyor" - Aircraft landing detected
- "Köprü durdu ve bağlandı" - Bridge connected

## Performance Optimization

### Memory Usage
- Frame buffer management
- Efficient list operations
- Regular garbage collection

### Processing Speed
- Batch processing implementation
- GPU acceleration support
- Optimized frame sampling

## Troubleshooting

### Common Issues
1. Model Loading Errors
   - Verify model path
   - Check CUDA installation
   - Validate weights file

2. Video Processing Issues
   - Check codec compatibility
   - Verify file permissions
   - Monitor memory usage

3. Detection Accuracy
   - Adjust confidence thresholds
   - Update model weights
   - Check lighting conditions

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request
