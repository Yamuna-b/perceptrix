import cv2
import pyttsx3
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# Important classes for navigation
important_classes = {"person", "chair", "door", "car", "bus", "bicycle", "apple", "dog", "cat", "bottle"}

def estimate_distance(box_height, frame_height):
    if box_height == 0:
        return None
    K = 2.5 * frame_height
    distance = K / box_height
    return round(distance, 1)

def get_position(x_center, frame_width):
    if x_center < frame_width / 3:
        return "on your left"
    elif x_center > 2 * frame_width / 3:
        return "on your right"
    else:
        return "ahead"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if conf > 0.6 and label in important_classes:
                x1, y1, x2, y2 = box.xyxy[0]
                box_height = (y2 - y1).item()
                frame_height, frame_width = frame.shape[:2]
                x_center = (x1 + x2).item() / 2

                distance = estimate_distance(box_height, frame_height)
                direction = get_position(x_center, frame_width)

                if distance:
                    detections.append((distance, f"{label} {direction}, about {distance} meters"))
                else:
                    detections.append((999, f"{label} {direction}"))

    if detections:
        # Sort by closest object
        detections.sort(key=lambda x: x[0])
        # Speak only top 1 or 2 closest objects
        top_detections = [desc for _, desc in detections[:2]]

        for sentence in top_detections:
            print("🗣️", sentence)
            engine.say(sentence)

        engine.runAndWait()

    cv2.imshow("Perceptrix", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
