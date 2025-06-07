import cv2
import time
import serial
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('/home/pi/best.pt')  # Adjust if needed

# Define class names (update according to your model)
class_names = ['plastic', 'paper', 'metal', 'glass']

# Setup serial communication with Arduino
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)  # Allow time for Arduino to reset
except serial.SerialException:
    print("Failed to connect to Arduino. Check USB cable and port.")
    exit(1)

# Image capture and detection loop
try:
    while True:
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            time.sleep(60)
            continue

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Failed to capture image")
            time.sleep(60)
            continue

        # Save image
        img_path = '/home/pi/captured_image.jpg'
        cv2.imwrite(img_path, frame)
        print(f"Image saved to {img_path}")

        # Run YOLO detection
        results = model(img_path)[0]
        detected_classes = [class_names[int(cls)] for cls in results.boxes.cls]
        print("Detected:", detected_classes)

        # Send label to Arduino
        if detected_classes:
            label_to_send = detected_classes[0]  # Send first detected class
        else:
            label_to_send = 'none'  # Send 'none' if nothing detected

        try:
            arduino.write((label_to_send + '\n').encode())
            print(f"Sent to Arduino: {label_to_send}")
        except Exception as e:
            print(f"Error sending to Arduino: {e}")

        # Wait 1 minute
        time.sleep(60)

except KeyboardInterrupt:
    print("Terminated by user")

finally:
    arduino.close()


pip install pyserial