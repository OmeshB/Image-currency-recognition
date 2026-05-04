import cv2
import os
from preprocess import preprocess_image

def extract_orb(image):
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(
        image, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )


if __name__ == "__main__":
    sample_path = "dataset/Train/Tennote/1.jpg"

    output_folder = "Outputs/keypoints"
    os.makedirs(output_folder, exist_ok=True)

    # Preprocess first
    processed = preprocess_image(sample_path)

    if processed is None:
        print("Error loading image")
    else:
        # ORB extraction
        kp, desc = extract_orb(processed)

        print("Number of keypoints:", len(kp))

        # Draw keypoints
        kp_image = draw_keypoints(processed, kp)

        output_path = os.path.join(output_folder, "orb_keypoints.jpg")
        cv2.imwrite(output_path, kp_image)

        print("Keypoints image saved at:", output_path)