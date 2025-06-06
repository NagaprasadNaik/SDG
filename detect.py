import cv2
import RPi.GPIO as GPIO
from ultralytics import YOLO
import time

# Define GPIO pin numbers
PLASTIC_PIN = 17
OTHER_PIN = 27

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(PLASTIC_PIN, GPIO.OUT)
GPIO.setup(OTHER_PIN, GPIO.OUT)

# Load the YOLOv8 model
model = YOLO('/home/pi/best.pt')  # Change path if needed

# Define class names (edit this list to match your model's class names)
class_names = ['plastic', 'paper', 'metal', 'glass']

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run detection
        results = model(frame)[0]

        # Get class names from results
        detected_classes = [class_names[int(cls)] for cls in results.boxes.cls]

        print("Detected:", detected_classes)

        # Set GPIO based on detection
        if 'plastic' in detected_classes:
            GPIO.output(PLASTIC_PIN, GPIO.HIGH)
            GPIO.output(OTHER_PIN, GPIO.LOW)
        elif len(detected_classes) > 0:
            GPIO.output(PLASTIC_PIN, GPIO.LOW)
            GPIO.output(OTHER_PIN, GPIO.HIGH)
        else:
            GPIO.output(PLASTIC_PIN, GPIO.LOW)
            GPIO.output(OTHER_PIN, GPIO.LOW)

        # Optional: display the video
        cv2.imshow("Waste Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
