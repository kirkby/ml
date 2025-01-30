""""
This script demonstrates how to crop an image using a trained YOLO model.
"""
import cv2
from ultralytics import YOLO

# Indlæs trænet model
model = YOLO("runs/detect/train6/weights/best.pt")

# Load the image
image_path = "datasets/images/train/4-15.jpg"
frame = cv2.imread(image_path)

# Inference
results = model(frame)
box = results[0].boxes
x1, y1, x2, y2 = box.xyxy[0]
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

# Crop the image by the bounding box
cropped_image = frame[y1:y2, x1:x2]

# Save the output image
output_path_cropped = "output_cropped.jpg"
cv2.imwrite(output_path_cropped, cropped_image)
