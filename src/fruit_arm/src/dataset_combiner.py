import os
import shutil
from pathlib import Path
import random
import hashlib

def get_image_hash(image_path):
    """Generate MD5 hash of image file to detect actual duplicates"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def merge_datasets(source_dirs, target_dir, split_ratios=(0.7, 0.2, 0.1)):
    """
    Merge multiple VOC format datasets into one unified dataset
    
    Args:
        source_dirs: List of source dataset directories
        target_dir: Target directory for merged dataset
        split_ratios: Tuple of (train, valid, test) ratios
    """
    
    # Create target directory structure
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'valid', 'test']
    for split in splits:
        (target_path / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all files from all source datasets
    all_files = []
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        
        for split in splits:
            split_path = source_path / split
            if split_path.exists():
                # Find all image files
                for img_file in split_path.glob('*.jpg'):
                    xml_file = img_file.with_suffix('.xml')
                    if xml_file.exists():
                        all_files.append({
                            'img': img_file,
                            'xml': xml_file,
                            'split': split,
                            'source': source_dir,
                            'original_name': img_file.name
                        })
    
    print(f"Found {len(all_files)} total image-annotation pairs")
    
    # Remove ACTUAL duplicate images (same content) using MD5 hash
    print("Checking for duplicate images (by content)...")
    unique_images = {}
    duplicate_count = 0
    
    for file_info in all_files:
        img_hash = get_image_hash(file_info['img'])
        if img_hash is None:
            continue
            
        if img_hash not in unique_images:
            unique_images[img_hash] = file_info
        else:
            duplicate_count += 1
            print(f"Found duplicate image: {file_info['img']}")
    
    if duplicate_count > 0:
        print(f"Removed {duplicate_count} duplicate images (same content)")
        all_files = list(unique_images.values())
    else:
        print("No duplicate images found by content")
    
    # Handle filename conflicts by renaming
    print("Handling filename conflicts...")
    name_counter = {}
    processed_files = []
    
    for file_info in all_files:
        original_name = file_info['original_name']
        source_name = Path(file_info['source']).name
        
        if original_name not in name_counter:
            name_counter[original_name] = 1
            new_name = original_name
        else:
            name_counter[original_name] += 1
            # Rename duplicate filename: "image.jpg" -> "image_source_2.jpg"
            name_parts = original_name.split('.')
            new_name = f"{name_parts[0]}_{source_name}_{name_counter[original_name]}.{name_parts[1]}"
            print(f"Renaming duplicate: {original_name} -> {new_name}")
        
        file_info['new_name'] = new_name
        processed_files.append(file_info)
    
    print(f"Final dataset size: {len(processed_files)} images")
    
    # Shuffle the files
    random.shuffle(processed_files)
    
    # Split files according to ratios
    total_files = len(processed_files)
    train_count = int(total_files * split_ratios[0])
    valid_count = int(total_files * split_ratios[1])
    
    train_files = processed_files[:train_count]
    valid_files = processed_files[train_count:train_count + valid_count]
    test_files = processed_files[train_count + valid_count:]
    
    print(f"Final split: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test")
    
    # Copy files to target directory
    split_mapping = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }
    
    for split_name, files in split_mapping.items():
        print(f"Copying {split_name} files...")
        for file_info in files:
            # Use new name to avoid conflicts
            new_img_name = file_info['new_name']
            new_xml_name = file_info['new_name'].replace('.jpg', '.xml')
            
            target_img = target_path / split_name / new_img_name
            target_xml = target_path / split_name / new_xml_name
            
            # Copy image file
            shutil.copy2(file_info['img'], target_img)
            # Copy XML annotation file
            shutil.copy2(file_info['xml'], target_xml)
    
    # Create dataset info file
    create_dataset_info(target_path, processed_files)
    
    print(f"Dataset merged successfully to: {target_dir}")

def create_dataset_info(target_path, all_files):
    """Create a dataset information file"""
    info_file = target_path / "dataset_info.txt"
    
    class_counts = {}
    split_counts = {'train': 0, 'valid': 0, 'test': 0}
    source_counts = {}
    renamed_files = []
    
    for file_info in all_files:
        # Extract class from filename (e.g., "apple_10_jpg" -> "apple")
        filename = file_info['new_name']
        class_name = filename.split('_')[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        split_counts[file_info['split']] += 1
        source_name = Path(file_info['source']).name
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
        
        # Track renamed files
        if file_info['original_name'] != file_info['new_name']:
            renamed_files.append(f"{file_info['original_name']} -> {file_info['new_name']}")
    
    with open(info_file, 'w') as f:
        f.write("Merged Fruit Detection Dataset Info\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Class Distribution:\n")
        for class_name, count in class_counts.items():
            percentage = (count / len(all_files)) * 100
            f.write(f"  {class_name}: {count} images ({percentage:.1f}%)\n")
        
        f.write(f"\nTotal Images: {len(all_files)}\n\n")
        
        f.write("Split Distribution:\n")
        for split_name, count in split_counts.items():
            percentage = (count / len(all_files)) * 100
            f.write(f"  {split_name}: {count} images ({percentage:.1f}%)\n")
        
        f.write("\nSource Datasets:\n")
        for source_name, count in source_counts.items():
            percentage = (count / len(all_files)) * 100
            f.write(f"  {source_name}: {count} images ({percentage:.1f}%)\n")
        
        if renamed_files:
            f.write(f"\nRenamed Files ({len(renamed_files)} files):\n")
            for rename in renamed_files[:10]:  # Show first 10
                f.write(f"  {rename}\n")
            if len(renamed_files) > 10:
                f.write(f"  ... and {len(renamed_files) - 10} more\n")

def main():
    # Define source datasets (adjust paths as needed)
    source_datasets = [
        "/home/atharva/quanser_ws/src/fruits_dataset/fruit detection.v1i.voc",
        "/home/atharva/quanser_ws/src/fruits_dataset/fruit detection.v1i1.voc",
        "/home/atharva/quanser_ws/src/fruits_dataset/fruit detection.v2i.voc",
        "/home/atharva/quanser_ws/src/fruits_dataset/fruit.v3i.voc"
    ]
    
    # Target directory for merged dataset
    target_dataset = "merged_fruit_detection_dataset"
    
    # Split ratios: (train, valid, test)
    split_ratios = (0.7, 0.2, 0.1)
    
    # Verify source datasets exist
    missing_sources = []
    for source in source_datasets:
        if not os.path.exists(source):
            missing_sources.append(source)
    
    if missing_sources:
        print(f"Error: The following source datasets were not found:")
        for missing in missing_sources:
            print(f"  - {missing}")
        print("\nPlease check the paths and try again.")
        return
    
    print("Starting dataset merge...")
    print(f"Source datasets: {source_datasets}")
    print(f"Target: {target_dataset}")
    print(f"Split ratios: {split_ratios}")
    print("-" * 50)
    
    # Merge datasets
    merge_datasets(source_datasets, target_dataset, split_ratios)
    
    print("\nMerge completed successfully!")
    print(f"Check '{target_dataset}/dataset_info.txt' for detailed statistics")

if __name__ == "__main__":
    main()