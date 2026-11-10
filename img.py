import cv2
import os

video_path = "2.mp4"

output_folder = "plate_samples"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    raise SystemExit

# Enter the frame number you want
frame_number = int(input("Enter frame number: "))

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()

cap.release()

if not ret:
    print("Could not read frame.")
    raise SystemExit

# Select the plate manually
bbox = cv2.selectROI(
    "Select Number Plate",
    frame,
    fromCenter=False,
    showCrosshair=True
)

x, y, w, h = map(int, bbox)

if w == 0 or h == 0:
    print("No selection made.")
    cv2.destroyAllWindows()
    raise SystemExit

crop = frame[y:y+h, x:x+w]

filename = os.path.join(
    output_folder,
    f"plate_{frame_number}.jpg"
)

cv2.imwrite(filename, crop)

cv2.destroyAllWindows()

print("Plate saved:", filename)