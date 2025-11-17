import os
from pathlib import Path

def verify_dataset(dataset_path):
    """Verify the merged dataset structure and files"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        return False
    
    splits = ['train', 'valid', 'test']
    issues = []
    
    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            issues.append(f"Missing split directory: {split}")
            continue
        
        # Check for image and annotation files
        jpg_files = list(split_path.glob('*.jpg'))
        xml_files = list(split_path.glob('*.xml'))
        
        print(f"{split}: {len(jpg_files)} images, {len(xml_files)} annotations")
        
        # Check if every image has a corresponding XML file
        jpg_names = {f.stem for f in jpg_files}
        xml_names = {f.stem for f in xml_files}
        
        missing_annotations = jpg_names - xml_names
        extra_annotations = xml_names - jpg_names
        
        if missing_annotations:
            issues.append(f"{split}: {len(missing_annotations)} images missing annotations")
        
        if extra_annotations:
            issues.append(f"{split}: {len(extra_annotations)} annotations without images")
    
    # Check class distribution
    print("\nClass distribution in merged dataset:")
    class_counts = {}
    for split in splits:
        split_path = dataset_path / split
        if split_path.exists():
            for img_file in split_path.glob('*.jpg'):
                class_name = img_file.name.split('_')[0]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\nDataset verification passed!")
        return True

# Usage
if __name__ == "__main__":
    verify_dataset("/home/atharva/quanser_ws/merged_fruit_detection_dataset")