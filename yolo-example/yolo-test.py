from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# results = model("dice.png")
# results = model("test.webp")
results = model("dice-on-table.jpg")
results[0].show()
