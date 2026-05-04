import cv2
import os
from preprocess import preprocess_image
from feature_extract import extract_orb


def get_template_paths(train_folder, templates_per_class=3):
    template_paths = {}

    for label in os.listdir(train_folder):
        class_path = os.path.join(train_folder, label)

        if not os.path.isdir(class_path):
            continue

        images = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        selected_images = images[:templates_per_class]
        template_paths[label] = [os.path.join(class_path, img) for img in selected_images]

    return template_paths


def get_good_match_score(img1, img2, ratio=0.75):
    kp1, desc1 = extract_orb(img1)
    kp2, desc2 = extract_orb(img2)

    if desc1 is None or desc2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    return len(good_matches)


def predict_denomination(test_image_path, template_paths, ratio=0.75):
    test_img = preprocess_image(test_image_path)

    if test_img is None:
        return None, {}

    scores = {}

    for label, template_list in template_paths.items():
        best_score = 0

        for template_path in template_list:
            template_img = preprocess_image(template_path)

            if template_img is None:
                continue

            score = get_good_match_score(template_img, test_img, ratio=ratio)

            if score > best_score:
                best_score = score

        scores[label] = best_score

    predicted_label = max(scores, key=scores.get)
    return predicted_label, scores


if __name__ == "__main__":
    train_folder = "dataset/Train"
    template_paths = get_template_paths(train_folder, templates_per_class=3)

    test_image_path = "dataset/Test/Tennote/1.jpg"

    predicted_label, scores = predict_denomination(test_image_path, template_paths, ratio=0.75)

    print("Template images selected:")
    for label, paths in template_paths.items():
        print(f"{label}:")
        for p in paths:
            print(f"   {p}")

    print("\nBest match scores:")
    for label, score in scores.items():
        print(f"{label}: {score}")

    print("\nPredicted denomination:", predicted_label)