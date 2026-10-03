from ultralytics import YOLO
import os

# Load trained YOLO model
model = YOLO("../model/best.pt")

# Folder containing multiple test images
image_folder = "../images"

# Run detection on all images
results = model.predict(
    source=image_folder,
    conf=0.25,
    save=True
)

# Print detected classes
for result in results:
    print("\nImage:", os.path.basename(result.path))

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            print(
                f"Detected: {class_name} | "
                f"Confidence: {confidence:.2f}"
            )