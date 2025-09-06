import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3

# Load BLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Start webcam
cap = cv2.VideoCapture(0)

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (as PIL expects)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare input
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)

    # Generate caption
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Speak caption
    print("🗣️", caption)
    engine.say(caption)
    engine.runAndWait()

    # Show webcam
    cv2.imshow("Perceptrix - Scene Narration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
