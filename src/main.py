import os

dataset_path = "dataset"

for split in ["Train", "Test"]:
    split_path = os.path.join(dataset_path, split)
    print(f"\n{split} classes:")
    
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            print(f"{class_name}: {count} images")