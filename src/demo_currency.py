import cv2
import os
from pathlib import Path

# -------------------------------
# PROJECT PATHS
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
TRAIN_PATH = DATASET_PATH / "Train"
OUTPUT_PATH = PROJECT_ROOT / "Outputs" / "demo"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------
# SETTINGS
# -------------------------------
TEMPLATES_PER_CLASS = 10
RATIO = 0.7
IMAGE_SIZE = (224, 224)
UNKNOWN_THRESHOLD = 0.08   # normalized score threshold

# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocess_image(image_path, size=IMAGE_SIZE):
    img = cv2.imread(str(image_path))

    if img is None:
        return None, None

    original = img.copy()

    # Resize
    img = cv2.resize(img, size)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    return original, enhanced

# -------------------------------
# ORB FEATURE EXTRACTION
# -------------------------------
def extract_orb(image):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# -------------------------------
# TEMPLATE SELECTION
# -------------------------------
def get_template_paths(train_folder, templates_per_class=TEMPLATES_PER_CLASS):
    template_paths = {}

    for label in os.listdir(train_folder):
        class_path = train_folder / label

        if not class_path.is_dir():
            continue

        images = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        selected_images = images[:templates_per_class]
        template_paths[label] = [class_path / img for img in selected_images]

    return template_paths

# -------------------------------
# NORMALIZED MATCH SCORE
# -------------------------------
def get_good_match_score(img1, img2, ratio=RATIO):
    kp1, desc1 = extract_orb(img1)
    kp2, desc2 = extract_orb(img2)

    if desc1 is None or desc2 is None:
        return 0.0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    good_count = len(good_matches)
    denom = max(len(desc1), len(desc2), 1)
    normalized_score = good_count / denom

    return normalized_score, good_count

# -------------------------------
# PREDICTION
# -------------------------------
def predict_currency(input_image_path, template_paths):
    original, test_img = preprocess_image(input_image_path)

    if test_img is None:
        return None, None, {}, {}

    scores = {}
    raw_match_counts = {}

    for label, template_list in template_paths.items():
        best_score = 0.0
        best_raw_count = 0

        for template_path in template_list:
            _, template_img = preprocess_image(template_path)

            if template_img is None:
                continue

            score, raw_count = get_good_match_score(template_img, test_img)

            if score > best_score:
                best_score = score
                best_raw_count = raw_count

        scores[label] = best_score
        raw_match_counts[label] = best_raw_count

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score < UNKNOWN_THRESHOLD:
        predicted_label = "Unknown"
    else:
        predicted_label = best_label

    return original, predicted_label, scores, raw_match_counts

# -------------------------------
# DISPLAY + SAVE RESULT
# -------------------------------
def show_result(original_image, predicted_label, input_path, scores, raw_match_counts):
    display_img = original_image.copy()
    display_img = cv2.resize(display_img, (700, 400))

    text1 = f"Detected: {predicted_label}"
    text2 = f"Input: {Path(input_path).name}"

    cv2.putText(display_img, text1, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_img, text2, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    output_file = OUTPUT_PATH / "demo_result.jpg"
    cv2.imwrite(str(output_file), display_img)

    print("\nPrediction scores (normalized):")
    for label, score in scores.items():
        print(f"{label}: {score:.4f}")

    print("\nGood match counts:")
    for label, count in raw_match_counts.items():
        print(f"{label}: {count}")

    print(f"\nPredicted denomination: {predicted_label}")
    print(f"Result image saved at: {output_file}")

    cv2.imshow("Indian Currency Recognition - Demo", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Train folder exists:", TRAIN_PATH.exists())

    template_paths = get_template_paths(TRAIN_PATH, templates_per_class=TEMPLATES_PER_CLASS)

    print("\nAvailable sample input examples:")
    print("1. dataset/Test/Tennote/1.jpg")
    print("2. dataset/Test/2Hundrednote/1.jpg")
    print("3. dataset/Test/Fiftynote/1.jpg")
    print("4. dataset/Test/Twentynote/1.jpg")

    user_input = input(
        "\nEnter input image path relative to project folder\n"
        "(or press Enter to use dataset/Test/Tennote/1.jpg): "
    ).strip()

    if user_input == "":
        input_image_path = PROJECT_ROOT / "dataset" / "Test" / "Tennote" / "1.jpg"
    else:
        input_image_path = PROJECT_ROOT / user_input

    print("\nInput image path:", input_image_path)
    print("Input exists:", input_image_path.exists())

    if not input_image_path.exists():
        print("\nError: Input image path not found.")
    else:
        original, predicted_label, scores, raw_match_counts = predict_currency(
            input_image_path, template_paths
        )

        if original is None:
            print("\nError: Could not read the input image.")
        else:
            show_result(original, predicted_label, input_image_path, scores, raw_match_counts)