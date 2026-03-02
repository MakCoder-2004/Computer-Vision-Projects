"""
Prepare dataset for YOLO training by splitting into train and validation sets.
"""

import os
import shutil
import random
from pathlib import Path

def prepare_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    """
    Split dataset into train and validation sets for YOLO.

    Args:
        images_dir: directory containing images
        labels_dir: directory containing YOLO labels
        output_dir: output directory for organized dataset
        train_ratio: ratio of training data (default 0.8)
    """
    # Create output directories
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get all image files
    image_files = list(Path(images_dir).glob('*.png')) + \
                  list(Path(images_dir).glob('*.jpg')) + \
                  list(Path(images_dir).glob('*.jpeg'))

    print(f"Found {len(image_files)} images")

    # Filter images that have corresponding labels
    valid_images = []
    for img_file in image_files:
        label_file = Path(labels_dir) / (img_file.stem + '.txt')
        if label_file.exists():
            valid_images.append(img_file)

    print(f"Found {len(valid_images)} images with corresponding labels")

    # Split into train and validation
    random.seed(42)
    random.shuffle(valid_images)
    split_idx = int(len(valid_images) * train_ratio)
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]

    print(f"Train set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")

    # Copy train files
    print("\nCopying train files...")
    for img_file in train_images:
        # Copy image
        shutil.copy2(img_file, os.path.join(train_images_dir, img_file.name))

        # Copy label
        label_file = Path(labels_dir) / (img_file.stem + '.txt')
        shutil.copy2(label_file, os.path.join(train_labels_dir, label_file.name))

    # Copy validation files
    print("Copying validation files...")
    for img_file in val_images:
        # Copy image
        shutil.copy2(img_file, os.path.join(val_images_dir, img_file.name))

        # Copy label
        label_file = Path(labels_dir) / (img_file.stem + '.txt')
        shutil.copy2(label_file, os.path.join(val_labels_dir, label_file.name))

    print("\nDataset preparation complete!")
    print(f"Train images: {train_images_dir}")
    print(f"Train labels: {train_labels_dir}")
    print(f"Val images: {val_images_dir}")
    print(f"Val labels: {val_labels_dir}")


if __name__ == "__main__":
    IMAGES_DIR = "../data/images"
    LABELS_DIR = "../data/labels"
    OUTPUT_DIR = "../dataset"

    prepare_yolo_dataset(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR, train_ratio=0.8)