from ultralytics import YOLO

# 1. Load your trained model weights (adjust train run folder path if needed)
model_path = r"runs\classify\train\weights\best.pt"
model = YOLO(model_path)

# 2. Path to the test image
test_image_path = r"C:\Users\Nirmal Rajasekaran\Desktop\Farm ML\dataset\train\unclean\images (3).jpg"

# 3. Run prediction
results = model.predict(test_image_path)

# 4. Find the class index for 'unclean'
class_names = results[0].names  # Dictionary mapping IDs to names, e.g., {0: 'clean', 1: 'unclean'}
unclean_class_id = None

for idx, name in class_names.items():
    if name.lower() == "unclean":
        unclean_class_id = idx
        break

# 5. Extract confidence for the 'unclean' class
if unclean_class_id is not None:
    # Get probabilities tensor and extract unclean confidence score
    probs = results[0].probs.data.tolist()
    unclean_confidence = probs[unclean_class_id]

    # 6. Apply 60% threshold rule
    if unclean_confidence >= 0.70:
        final_prediction = "unclean"
    else:
        final_prediction = "clean"

    print(f"Unclean Confidence Score: {unclean_confidence * 100:.2f}%")
    print(f"Final Classification: {final_prediction}")
else:
    print("Error: 'unclean' class label not found in model output.")