"""
This script demonstrates how to use a trained YOLO model to draw bounding boxes on an image.
"""

import cv2
from ultralytics import YOLO

# Indlæs trænet model
model = YOLO("runs/detect/train6/weights/best.pt")

# Load the image
image_path = "datasets/images/train/3-12.jpg"
frame = cv2.imread(image_path)

# Inference
results = model(frame)
box = results[0].boxes
x1, y1, x2, y2 = box.xyxy[0]
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

# Draw the bounding box
conf = box.conf[0]
cls = int(box.cls[0])
label = f"{cls + 1}: {conf:.2f}"
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Save the output image
output_path = "output.jpg"
cv2.imwrite(output_path, frame)

# Display the image
cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
