🚘 PlateTrack-AI

AI-Powered Vehicle Number Plate Recognition and Monitoring System

PlateTrack AI is an intelligent vehicle monitoring system that uses deep learning and computer vision to detect vehicles, recognize Indian vehicle registration plates, track vehicle movement, and automatically maintain entry and exit records with timestamps.

The system integrates YOLO11, ByteTrack, PaddleOCR, and OpenCV to provide automated vehicle identification and monitoring through video footage.

✨ Features

- Number Plate Detection: Detects vehicle number plates using a custom-trained YOLO11 model.
- Optical Character Recognition: Extracts registration numbers from detected plates using PaddleOCR.
- Vehicle Tracking: Uses ByteTrack to maintain vehicle identities across video frames.
- Entry/Exit Monitoring: Identifies vehicle entry and exit events based on movement across a predefined virtual line.
- OCR Verification: Uses repeated recognition and confidence-based selection to improve plate-reading reliability.
- Automatic Logging: Records recognized registration numbers, entry/exit status, and timestamps in a CSV file.
- Video Processing: Processes video footage and generates an annotated output video.

🛠️ Technologies Used

Technology| Purpose
Python| Core programming language
YOLO11| Number plate detection
PaddleOCR| Text recognition
ByteTrack| Object tracking
OpenCV| Image and video processing
CSV| Vehicle event logging

⚙️ System Workflow

1. Input video is loaded and processed frame by frame.
2. YOLO11 detects vehicle number plates.
3. ByteTrack assigns and maintains tracking identities.
4. Detected plate regions are passed to PaddleOCR.
5. OCR results are verified using confidence-based selection.
6. Vehicle movement is monitored using a predefined virtual line.
7. Entry and exit events are recorded in a CSV file.
8. An annotated video is generated as output.

📁 Project Structure

PlateTrack-AI
│
├── main.py              # Main vehicle detection pipeline
├── img.py               # Extracts and saves manually selected plate images
├── ocr_helper.py        # Number plate text recognition using PaddleOCR
├── tracker_helper.py    # Vehicle tracking and CSV logging
│
├── best.pt              # Trained YOLO11 model
│
├── detected_video.mp4   # Processed output video
├── vehicle_log.csv      # Vehicle entry/exit records
│
├── plate_samples/       # Extracted number plate images
│
└── README.md

🚀 Installation and Setup

1. Clone the repository

git clone https://github.com/kumarisakshi26xvw/PlateTrackAI.git
cd PlateTrack-AI

2. Install dependencies

pip install ultralytics opencv-python paddleocr paddlepaddle

Install any additional dependencies required by your environment.

3. Configure the project

- Place your trained YOLO11 weights ("best.pt") in the project directory.
- Place the input video in the project directory.
- Update the model and video paths in "main.py" if required.

4. Run the application

python main.py

📊 Output

The system generates:

- Detected Video: An annotated video showing detected number plates and tracking information.
- Vehicle Log (CSV): A structured record containing:

License Plate| Status| Timestamp
KA02MN1826| IN| Recorded timestamp
JH05AB1234| OUT| Recorded timestamp



👩‍💻 Project Information

Project: PlateTrack AI
Domain: Artificial Intelligence | Computer Vision | Deep Learning
Application: Automated Vehicle Monitoring and Number Plate Recognition

---

PlateTrack AI aims to simplify vehicle monitoring through automated number plate recognition, object tracking, and event-based logging.
