import cv2
import os

def preprocess_image(image_path, size=(224, 224)):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return None

    # 1. Resize
    img = cv2.resize(img, size)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoise (Gaussian Blur)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Enhance (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    return enhanced


if __name__ == "__main__":
    # Pick ONE sample image
    sample_path = "dataset/Train/Tennote/1.jpg"

    output_folder = "Outputs/preprocessed"
    os.makedirs(output_folder, exist_ok=True)

    result = preprocess_image(sample_path)

    if result is not None:
        output_path = os.path.join(output_folder, "sample_preprocessed.jpg")
        cv2.imwrite(output_path, result)
        print("Preprocessed image saved at:", output_path)