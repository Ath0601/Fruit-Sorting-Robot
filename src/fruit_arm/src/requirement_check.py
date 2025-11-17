import importlib
import sys

def check_package(package_name, min_version=None):
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown version')
        
        if min_version:
            from packaging import version as packaging_version
            if packaging_version.parse(version) >= packaging_version.parse(min_version):
                status = "✅ SATISFIED"
            else:
                status = "❌ VERSION TOO LOW"
        else:
            status = "✅ INSTALLED"
        
        print(f"{package_name:15} {version:15} {status}")
        return True
        
    except ImportError:
        print(f"{package_name:15} {'Not installed':15} ❌ MISSING")
        return False

# Required packages with minimum versions
requirements = {
    'torch': '1.9.0',
    'torchvision': '0.10.0', 
    'matplotlib': '3.3.0',
    'numpy': '1.19.0',
    'cv2': '4.5.0',  # opencv-python
    'PIL': '8.0.0',   # Pillow
}

print("Checking SSD Training Requirements:")
print("=" * 50)

all_satisfied = True
for package, min_version in requirements.items():
    if package == 'cv2':
        package_name = 'opencv-python'
    elif package == 'PIL':
        package_name = 'Pillow'
    else:
        package_name = package
    
    satisfied = check_package(package, min_version)
    if not satisfied:
        all_satisfied = False

print("=" * 50)
if all_satisfied:
    print("🎉 All requirements satisfied! You can start training.")
else:
    print("❌ Some requirements are missing. Please install them with:")
    print("pip install torch>=1.9.0 torchvision>=0.10.0 matplotlib>=3.3.0 numpy>=1.19.0 opencv-python>=4.5.0 Pillow>=8.0.0")