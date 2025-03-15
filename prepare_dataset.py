import os
import random
import shutil

# 1. Paths to your current dataset
dataset_dir = r"C:\Users\adimalaa\OneDrive - NTNU\Desktop\Dataset"
images_dir = os.path.join(dataset_dir, "Images")
annotations_dir = os.path.join(dataset_dir, "Annotations")

# 2. Paths for the new YOLOv8 structure
images_output_dir = os.path.join(dataset_dir, "images")
labels_output_dir = os.path.join(dataset_dir, "labels")

train_images_dir = os.path.join(images_output_dir, "train")
val_images_dir = os.path.join(images_output_dir, "val")
train_labels_dir = os.path.join(labels_output_dir, "train")
val_labels_dir = os.path.join(labels_output_dir, "val")

# 3. Create the subfolders if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 4. Gather all .jpg files in the Images folder
all_images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

# 5. Shuffle and split into train/val
train_split_ratio = 0.8  # 80% train, 20% val
random.seed(42)  # For reproducible splits
random.shuffle(all_images)
split_index = int(train_split_ratio * len(all_images))
train_files = all_images[:split_index]
val_files = all_images[split_index:]


# 6. Function to copy image & corresponding annotation
def copy_data(image_file, dst_img_folder, dst_label_folder):
    # image_file = 'cube_0001.jpg'
    base_name = os.path.splitext(image_file)[0]  # 'cube_0001'

    # Source image path
    src_img_path = os.path.join(images_dir, image_file)
    # Destination image path
    dst_img_path = os.path.join(dst_img_folder, image_file)

    # Copy image
    shutil.copy(src_img_path, dst_img_path)

    # Annotation file name (txt)
    annotation_file = base_name + ".txt"
    src_label_path = os.path.join(annotations_dir, annotation_file)
    dst_label_path = os.path.join(dst_label_folder, annotation_file)

    # Copy annotation if it exists
    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dst_label_path)
    else:
        print(f"Warning: Annotation not found for {image_file}")


# 7. Copy train files
for img_file in train_files:
    copy_data(img_file, train_images_dir, train_labels_dir)

# 8. Copy val files
for img_file in val_files:
    copy_data(img_file, val_images_dir, val_labels_dir)

print("Dataset successfully split into train and val sets!")
print(f"Train set: {len(train_files)} images")
print(f"Val set:   {len(val_files)} images")
