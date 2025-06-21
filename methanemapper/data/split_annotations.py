import json
import argparse
import random
from pathlib import Path

def split_annotations(input_json_path, output_dir, split_ratio=0.8, seed=42):
    """
    Splits a custom annotation file from annotation_generator.py into training and validation sets.

    Args:
        input_json_path (str): Path to the input annotation JSON file.
        output_dir (str): Directory to save the output train.json and val.json files.
        split_ratio (float): The ratio of training data (default: 0.8).
        seed (int): Random seed for reproducible splits (default: 42).
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from {input_json_path}")
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Extract data components from the custom format
    # In this format, each element in 'annotations' corresponds to one image and its labels.
    all_annotated_images = data.get('annotations', [])
    
    # Preserve the metadata
    info = data.get('info', {})
    licenses = data.get('licenses', [])
    categories = data.get('categories', [])

    print(f"Total annotated images found: {len(all_annotated_images)}")

    # Shuffle the list of annotated images for a random split
    random.shuffle(all_annotated_images)

    # Split the list based on the ratio
    split_index = int(len(all_annotated_images) * split_ratio)
    train_annotations = all_annotated_images[:split_index]
    val_annotations = all_annotated_images[split_index:]

    # Create train and val data structures, preserving the original metadata
    train_data = {
        'info': info,
        'licenses': licenses,
        'annotations': train_annotations,
        'categories': categories
    }

    val_data = {
        'info': info,
        'licenses': licenses,
        'annotations': val_annotations,
        'categories': categories
    }

    # Save to new json files
    train_json_path = output_dir / 'train.json'
    val_json_path = output_dir / 'val.json'

    print(f"Saving training annotations to {train_json_path}")
    with open(train_json_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"Saving validation annotations to {val_json_path}")
    with open(val_json_path, 'w') as f:
        json.dump(val_data, f, indent=2)

    print("\nSplitting complete!")
    print(f"Training set: {len(train_annotations)} annotated images")
    print(f"Validation set: {len(val_annotations)} annotated images")
    print(f"Split ratio: {split_ratio:.1%} train / {(1-split_ratio):.1%} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split custom annotation file into training and validation sets."
    )
    parser.add_argument(
        '--input_json', 
        type=str, 
        required=True, 
        help='Path to the input annotation JSON file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help='Directory to save the output train.json and val.json files'
    )
    parser.add_argument(
        '--split_ratio', 
        type=float, 
        default=0.8, 
        help='The ratio of training data (default: 0.8)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Random seed for reproducible splits (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate split ratio
    if not 0.0 < args.split_ratio < 1.0:
        raise ValueError("Split ratio must be between 0 and 1")
    
    split_annotations(args.input_json, args.output_dir, args.split_ratio, args.seed)