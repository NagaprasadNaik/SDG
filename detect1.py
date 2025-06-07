import cv2
import time
import RPi.GPIO as GPIO
from ultralytics import YOLO

# Define GPIO pin numbers
PLASTIC_PIN = 17
OTHER_PIN = 27

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(PLASTIC_PIN, GPIO.OUT)
GPIO.setup(OTHER_PIN, GPIO.OUT)

# Load YOLOv8 model
model = YOLO('/home/pi/best.pt')  # Adjust path if needed

# Define class names (update this to match your model)
class_names = ['plastic', 'paper', 'metal', 'glass']

# Image capture and inference loop
try:
    while True:
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not found!")
            time.sleep(60)
            continue

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to capture image")
            time.sleep(60)
            continue

        # Save the captured image
        img_path = "/home/pi/captured_image.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Image saved to {img_path}")

        # Run detection on the saved image
        results = model(img_path)[0]
        detected_classes = [class_names[int(cls)] for cls in results.boxes.cls]
        print("Detected:", detected_classes)

        # Control GPIO based on detection
        if 'plastic' in detected_classes:
            GPIO.output(PLASTIC_PIN, GPIO.HIGH)
            GPIO.output(OTHER_PIN, GPIO.LOW)
        elif len(detected_classes) > 0:
            GPIO.output(PLASTIC_PIN, GPIO.LOW)
            GPIO.output(OTHER_PIN, GPIO.HIGH)
        else:
            GPIO.output(PLASTIC_PIN, GPIO.LOW)
            GPIO.output(OTHER_PIN, GPIO.LOW)

        # Sleep for 1 minute
        print("Sleeping for 60 seconds...\n")
        time.sleep(60)

except KeyboardInterrupt:
    print("Terminated by user")

finally:
    GPIO.cleanup()
