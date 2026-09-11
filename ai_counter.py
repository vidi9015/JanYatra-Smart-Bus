from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Ask for image/video file
file_name = input("Enter image/video file name: ")

# Run YOLO detection
results = model(file_name, conf=0.35)

# Count people
total_people = 0

for result in results:
    if result.boxes is not None:
        for cls in result.boxes.cls:
            if int(cls) == 0:   # Class 0 = person
                total_people += 1

# Bus capacity
capacity = 10

# Calculate occupancy
occupancy = (total_people / capacity) * 100

# Crowd classification
if occupancy <= 40:
    crowd_status = "LOW"
elif occupancy <= 70:
    crowd_status = "MODERATE"
elif occupancy <= 100:
    crowd_status = "HIGH"
else:
    crowd_status = "OVER-CAPACITY"

# Display JanYatra result
print("--------------------------------")
print("JANyatra AI Passenger Counter")
print("--------------------------------")
print("Passengers detected:", total_people)
print("Bus Capacity:", capacity)
print(f"Occupancy: {occupancy:.1f}%")
print("Crowd Status:", crowd_status)
print("--------------------------------")