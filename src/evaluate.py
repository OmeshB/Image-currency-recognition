import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

from preprocess import preprocess_image
from feature_extract import extract_orb
import cv2


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


def evaluate_dataset(test_folder, template_paths, ratio=0.75):
    y_true = []
    y_pred = []
    results = []

    for true_label in os.listdir(test_folder):
        class_path = os.path.join(test_folder, true_label)

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):
            if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(class_path, file_name)

            predicted_label, scores = predict_denomination(image_path, template_paths, ratio=ratio)

            if predicted_label is None:
                continue

            y_true.append(true_label)
            y_pred.append(predicted_label)

            row = {
                "filename": file_name,
                "actual": true_label,
                "predicted": predicted_label
            }

            for label, score in scores.items():
                row[f"score_{label}"] = score

            results.append(row)

    return y_true, y_pred, results


if __name__ == "__main__":
    train_folder = "dataset/Train"
    test_folder = "dataset/Test"
    output_folder = "Outputs/metrics"
    os.makedirs(output_folder, exist_ok=True)

    templates_per_class = 5
    ratio = 0.7

    template_paths = get_template_paths(train_folder, templates_per_class=templates_per_class)

    y_true, y_pred, results = evaluate_dataset(test_folder, template_paths, ratio=ratio)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("Accuracy:", accuracy)
    print("Weighted F1-score:", f1)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, "prediction_results_improved.csv")
    df.to_csv(csv_path, index=False)

    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix (templates={templates_per_class}, ratio={ratio})")
    plt.tight_layout()

    cm_path = os.path.join(output_folder, "confusion_matrix_improved.png")
    plt.savefig(cm_path)
    plt.show()

    print(f"\nImproved results CSV saved at: {csv_path}")
    print(f"Improved confusion matrix image saved at: {cm_path}")