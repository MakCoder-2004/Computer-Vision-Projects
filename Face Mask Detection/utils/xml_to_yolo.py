"""
XML to YOLO Annotation Converter

Converts Pascal VOC XML annotation files to YOLO format.
YOLO format: <class_id> <x_center> <y_center> <width> <height>
All coordinates are normalized to [0, 1].
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_bbox_to_yolo(size, box):
    """
    Convert bounding box from Pascal VOC format to YOLO format.

    Args:
        size: tuple (width, height) of the image
        box: tuple (xmin, ymin, xmax, ymax) in pixels

    Returns:
        tuple (x_center, y_center, width, height) normalized to [0, 1]
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    # Calculate center coordinates
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0

    # Calculate width and height
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Normalize
    x_center = x_center * dw
    y_center = y_center * dh
    width = width * dw
    height = height * dh

    return (x_center, y_center, width, height)


def parse_xml_annotation(xml_file, class_mapping):
    """
    Parse XML annotation file and extract bounding boxes.

    Args:
        xml_file: path to XML annotation file
        class_mapping: dictionary mapping class names to class IDs

    Returns:
        list of tuples (class_id, x_center, y_center, width, height)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    yolo_annotations = []

    # Parse all objects in the image
    for obj in root.findall('object'):
        class_name = obj.find('name').text

        # Skip if class name not in mapping
        if class_name not in class_mapping:
            print(f"Warning: Class '{class_name}' not in class mapping. Skipping.")
            continue

        class_id = class_mapping[class_name]

        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Convert to YOLO format
        yolo_bbox = convert_bbox_to_yolo(
            (img_width, img_height),
            (xmin, ymin, xmax, ymax)
        )

        yolo_annotations.append((class_id, *yolo_bbox))

    return yolo_annotations


def convert_xml_to_yolo(xml_dir, output_dir, class_mapping):
    """
    Convert all XML files in a directory to YOLO format.

    Args:
        xml_dir: directory containing XML annotation files
        output_dir: directory to save YOLO annotation files
        class_mapping: dictionary mapping class names to class IDs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all XML files
    xml_files = list(Path(xml_dir).glob('*.xml'))

    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return

    print(f"Found {len(xml_files)} XML files")
    print(f"Converting to YOLO format...")

    converted_count = 0

    for xml_file in xml_files:
        try:
            # Parse XML file
            yolo_annotations = parse_xml_annotation(xml_file, class_mapping)

            # Create output filename (same name as XML but with .txt extension)
            output_filename = xml_file.stem + '.txt'
            output_path = os.path.join(output_dir, output_filename)

            # Write YOLO annotations to file
            with open(output_path, 'w') as f:
                for annotation in yolo_annotations:
                    class_id, x_center, y_center, width, height = annotation
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            converted_count += 1

            if converted_count % 100 == 0:
                print(f"Converted {converted_count}/{len(xml_files)} files...")

        except Exception as e:
            print(f"Error processing {xml_file.name}: {e}")
            continue

    print(f"\nConversion complete!")
    print(f"Successfully converted {converted_count}/{len(xml_files)} files")
    print(f"Output saved to: {output_dir}")


def get_classes_from_xml(xml_dir):
    """
    Scan all XML files and extract unique class names.

    Args:
        xml_dir: directory containing XML annotation files

    Returns:
        set of unique class names
    """
    xml_files = list(Path(xml_dir).glob('*.xml'))
    classes = set()

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                classes.add(class_name)
        except Exception as e:
            print(f"Error reading {xml_file.name}: {e}")
            continue

    return classes


def create_classes_file(classes, output_path):
    """
    Create a classes.txt file with class names.

    Args:
        classes: list of class names
        output_path: path to save classes.txt file
    """
    with open(output_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    print(f"Classes file saved to: {output_path}")


if __name__ == "__main__":
    # Configuration
    XML_DIR = "../data/annotations"
    OUTPUT_DIR = "../data/labels"

    # First, scan all XML files to get unique classes
    print("Scanning XML files to find all classes...")
    classes = get_classes_from_xml(XML_DIR)
    classes_list = sorted(list(classes))

    print(f"\nFound {len(classes_list)} unique classes:")
    for i, class_name in enumerate(classes_list):
        print(f"  {i}: {class_name}")

    # Create class mapping (class_name -> class_id)
    class_mapping = {class_name: idx for idx, class_name in enumerate(classes_list)}

    # Create classes.txt file
    create_classes_file(classes_list, "../classes.txt")

    # Convert XML files to YOLO format
    print("\n" + "="*50)
    convert_xml_to_yolo(XML_DIR, OUTPUT_DIR, class_mapping)

    print("\n" + "="*50)
    print("Class mapping:")
    for class_name, class_id in class_mapping.items():
        print(f"  {class_id}: {class_name}")

