from ultralytics import YOLO

# 1. Load pre-trained nano classification model
model = YOLO("yolov8n-cls.pt")

# Path to the structured dataset created in Step 1
dataset_path = r"C:\Users\Nirmal Rajasekaran\Desktop\Farm ML\dataset"

# 2. Train with anti-overfitting parameters & heavy augmentation
results = model.train(
    data=dataset_path,
    epochs=15,  # Keep epochs low for a very small dataset
    imgsz=224,
    batch=2,  # Small batch size for small dataset

    # Heavy Augmentations (creates artificial variations of your 22 images)
    degrees=20.0,  # Rotation
    fliplr=0.5,  # Horizontal flip
    hsv_h=0.02,  # Hue variation
    hsv_s=0.7,  # Saturation variation
    hsv_v=0.5  # Brightness variation
)

print("\n--- Training Complete ---")
print("Model saved to:", results.save_dir)