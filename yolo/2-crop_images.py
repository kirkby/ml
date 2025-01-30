import cv2
from ultralytics import YOLO
import os
import glob

# Indlæs trænet model
model = YOLO("runs/detect/train6/weights/best.pt")

# Load the image
# Directory containing images
image_dir = "datasets/images/train"

# Loop through all images in the directory
# Loop through all images in the directory
for image_path in glob.glob(os.path.join(image_dir, "*.jpg")):
    image_name = os.path.basename(image_path)
    frame = cv2.imread(image_path)
    print(f"Processing image: {image_name}")
    # Inference
    results = model(frame)
    box = results[0].boxes
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Crop the image by the bounding box
    cropped_image = frame[y1:y2, x1:x2]

    # Save the output image
    output_path_cropped = os.path.join("output", f"cropped_{image_name}")
    folder = image_name[0]
    output_path_cropped = os.path.join("output", folder, f"cropped_{image_name}")
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path_cropped), exist_ok=True)
    cv2.imwrite(output_path_cropped, cropped_image)
