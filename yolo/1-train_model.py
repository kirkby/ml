from ultralytics import YOLO

# Indlæs YOLO-model
model = YOLO("yolov8n.pt")  # Erstat med den korrekte modelsti

# Start træning
model.train(data="dice.yaml", epochs=50, batch=8, imgsz=224)
