import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import cv2

class SSDFruitDetector:
    def __init__(self, model_path, num_classes=5, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.classes = ['background', 'apple', 'banana', 'orange', 'mixed']
        self.colors = ['red', 'yellow', 'orange', 'purple']
        
        # Load model
        self.model = ssd300_vgg16(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            print("🔍 Loading model_state_dict from checkpoint...")
            state_dict = checkpoint["model_state_dict"]
        else:
            print("✅ Loading raw state_dict directly...")
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        keep = scores >= self.confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        return boxes, scores, labels, original_image
    
    def visualize_detection(self, image_path, output_path=None):
        boxes, scores, labels, image = self.detect(image_path)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw bounding boxes
        for box, score, label in zip(boxes, scores, labels):
            if label == 0:  # Skip background
                continue
                
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            
            # Create rectangle
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor=self.colors[label-1], 
                facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f'{self.classes[label]}: {score:.2f}'
            ax.text(
                xmin, ymin - 10, label_text,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors[label-1], alpha=0.7),
                fontsize=10, color='white', weight='bold'
            )
        
        ax.set_title('Fruit Detection Results')
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Result saved to: {output_path}")
        
        plt.show()
        
        return len(boxes)

def main():
    # Initialize detector
    detector = SSDFruitDetector('/home/atharva/quanser_ws/ssd_fruit_checkpoint_best.pth',confidence_threshold=0.5)
    
    # Test on a sample image
    test_image = "/home/atharva/quanser_ws/merged_fruit_detection_dataset/test/tog132_jpg.rf.6883ab6308bd09a73187357e0543bb92.jpg"  # Replace with actual path
    num_detections = detector.visualize_detection(test_image, "detection_result.png")
    print(f"Detected {num_detections} fruits")

if __name__ == "__main__":
    main()