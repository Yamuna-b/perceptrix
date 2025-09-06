import cv2
import torch
import pyttsx3
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load YOLOv8 for hazard alerts
yolo_model = YOLO("yolov8n.pt")

# Load BLIP for natural narration
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty("rate", 150)   # speed
engine.setProperty("volume", 1.0) # volume

# Open webcam
cap = cv2.VideoCapture(0)

print("✅ Perceptrix Hybrid Mode Started (Press 'q' to quit)")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- YOLO: Fast Hazard Detection --------
    results = yolo_model.predict(source=frame, verbose=False)
    hazards = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            if label in ["person", "car", "bus", "truck", "dog", "chair"]:  # key hazards
                hazards.append(label)

    if hazards:
        hazard_text = f"⚠️ Careful, {', '.join(set(hazards))} ahead."
        print("Hazard:", hazard_text)
        engine.say(hazard_text)

    # -------- BLIP: Natural Narration (every 20th frame to save speed) --------
    if frame_count % 20 == 0:  
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb_frame, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)

        narration = f"I see {caption}."
        print("Narration:", narration)
        engine.say(narration)

    engine.runAndWait()

    # Show webcam feed
    cv2.imshow("Perceptrix - Hybrid Guide", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
