#!/usr/bin/env python3
# fruit_vision_node.py

import rclpy, os
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge
import torch
import numpy as np
from typing import List, Tuple, Optional
import json
from threading import Lock

# Imports needed to build the model (copied from training_script.py)
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead

# ============================================================
# Model creation helper function
# (Copied from training_script.py)
# ============================================================
def create_ssd_model(num_classes):
    """
    Helper function to create the SSD model
    (Copied from training_script.py)
    """
    model = ssd300_vgg16(weights="DEFAULT")
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    return model

# ============================================================
# ROS Node
# ============================================================
class FruitVisionNode(Node):
    def check_camera_topics(self):
        """Check available camera topics"""
        try:
            import subprocess
            result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True)
            topics = result.stdout.split('\n')
            camera_topics = [t for t in topics if 'camera' in t or 'image' in t]
            self.get_logger().info("Available camera/image topics:")
            for topic in camera_topics:
                self.get_logger().info(f"  {topic}")
        except Exception as e:
            self.get_logger().error(f"Failed to list topics: {e}")
    
    def test_with_sample_image(self):
        """Test the model with a sample image to verify it's working"""
        try:
            # Load a test image (you can replace this with a path to an image with fruits)
            test_image_path = "/home/atharva/quanser_ws/merged_fruit_detection_dataset/test/mixed_5_jpg.rf.7c93d7da82cdcdd5a4c199dfe19ad615.jpg"  # Change this path
            if os.path.exists(test_image_path):
                test_image = cv2.imread(test_image_path)
                if test_image is not None:
                    input_tensors = self.preprocess_image(test_image)
                    with torch.no_grad():
                        outputs = self.model(input_tensors)
                    
                    detections = self.postprocess_detections(outputs, test_image.shape[:2])
                    self.get_logger().info(f"Test image detections: {len(detections)}")
                    
                    # Draw detections on test image
                    for detection in detections:
                        bbox = detection['bbox']
                        score = detection['score']
                        class_id = detection['class_id']
                        cv2.rectangle(test_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(test_image, f"Fruit {class_id}: {score:.2f}", 
                                (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save the result
                    cv2.imwrite("/home/atharva/quanser_ws/test_result.jpg", test_image)
                    self.get_logger().info("Saved test result to test_result.jpg")
        except Exception as e:
            self.get_logger().error(f"Test image error: {e}")

    def __init__(self):
        super().__init__('fruit_vision_node')
        
        # Parameters
        # IMPORTANT: Make sure this path points to your 'ssd_fruit_detector_final.pth' file
        self.declare_parameter('model_path', '/home/atharva/quanser_ws/ssd_fruit_checkpoint_best.pth')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('fruit_real_width', 0.07)  # 7cm average fruit diameter
        
        model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.fruit_real_width = self.get_parameter('fruit_real_width').value
        
        # === MODIFIED MODEL LOADING ===
        # Load SSD model - Standard PyTorch method
        self.device = torch.device('cpu')
        
        # This MUST match the num_classes from your training script
        num_classes = 5  # (background, apple, banana, orange, mixed)
        
        # 1. Create a new instance of the model architecture
        self.model = create_ssd_model(num_classes)
        
        # 2. Load the saved weights (the .pth file) from model_path
        try:
            # Load the file first. Add weights_only=True to fix warning and improve safety.
            loaded_data = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Check if this is a checkpoint dictionary or just a state_dict
            if "model_state_dict" in loaded_data:
                # It's a checkpoint dictionary, extract the state_dict
                state_dict = loaded_data["model_state_dict"]
            else:
                # It's a raw state_dict
                state_dict = loaded_data
            
            # Load the state_dict into the model
            self.model.load_state_dict(state_dict)

        except FileNotFoundError:
            self.get_logger().error(f"Model file not found at {model_path}. Exiting.")
            rclpy.shutdown()
            return
        except Exception as e:
            self.get_logger().error(f"Error loading model state_dict: {e}. Exiting.")
            rclpy.shutdown()
            return

        # 3. Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        self.get_logger().info(f"SSD model (state_dict) loaded on {self.device}")
        # === END OF MODIFIED SECTION ===

        # CV bridge
        self.bridge = CvBridge()
        
        # Camera calibration data
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_calibrated = False
        self.calibration_lock = Lock()
        
        # Latest depth image
        self.latest_depth = None
        self.depth_lock = Lock()
        
        # Publishers
        self.detection_2d_pub = self.create_publisher(String, '/fruit_detections_2d', 10)
        self.detection_3d_pub = self.create_publisher(String, '/fruit_detections_3d', 10)
        self.position_pub = self.create_publisher(PointStamped, '/fruit_position', 10)
        self.annotated_img_pub = self.create_publisher(Image, '/fruit_detection_image', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.image_callback, 
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.get_logger().info("Fruit Vision Node (Python) initialized")

    def camera_info_callback(self, msg: CameraInfo):
        with self.calibration_lock:
            if not self.camera_calibrated:
                # Extract camera matrix (3x3)
                self.camera_matrix = np.array(msg.k).reshape(3, 3)
                
                # Extract distortion coefficients
                self.dist_coeffs = np.array(msg.d)
                
                self.camera_calibrated = True
                self.get_logger().info("Camera calibration parameters received")

    def depth_callback(self, msg: Image):
        with self.depth_lock:
            try:
                # Check if this is actually a depth image
                if 'depth' in msg.header.frame_id or 'depth' in msg.encoding:
                    # Convert depth image to OpenCV format
                    if msg.encoding == '32FC1':
                        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
                    elif msg.encoding == '16UC1':
                        depth_16uc1 = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                        self.latest_depth = depth_16uc1.astype(np.float32) / 1000.0  # Convert mm to meters
                    else:
                        self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
                else:
                    # This might be a color image mistakenly sent to depth topic
                    self.get_logger().warn(f"Received non-depth image on depth topic: {msg.encoding}")
                    self.latest_depth = None
            except Exception as e:
                self.get_logger().error(f"Depth conversion error: {str(e)}")
                self.latest_depth = None

    def get_depth_at_pixel(self, u: int, v: int) -> Optional[float]:
        """Get depth value at specified pixel coordinates"""
        with self.depth_lock:
            if self.latest_depth is None:
                return None
            
            height, width = self.latest_depth.shape
            if u < 0 or u >= width or v < 0 or v >= height:
                return None
            
            depth = self.latest_depth[v, u]
            if np.isnan(depth) or depth <= 0:
                return None
            
            return float(depth)

    def find_valid_depth_near(self, u: int, v: int, radius: int = 5) -> Optional[float]:
        """Search nearby pixels for valid depth value"""
        for r in range(radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx == 0 and dy == 0:
                        continue
                    uu = u + dx
                    vv = v + dy
                    depth = self.get_depth_at_pixel(uu, vv)
                    if depth is not None:
                        return depth
        return None

    def calculate_3d_position(self, bbox, depth: float) -> Tuple[float, float, float]:
        """Calculate 3D position from 2D bounding box and depth"""
        if not self.camera_calibrated or self.camera_matrix is None:
            return float('nan'), float('nan'), float('nan')
        
        # Calculate center of bounding box
        center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
        center_y = bbox[1] + (bbox[3] - bbox[1]) / 2
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Convert to 3D coordinates
        x = (center_x - cx) * depth / fx
        y = (center_y - cy) * depth / fy
        z = depth
        
        return x, y, z

    def preprocess_image(self, cv_image: np.ndarray) -> List[torch.Tensor]:
        """Preprocess image for SSD model - returns list of tensors"""
        # Resize to SSD input size (300x300)
        resized = cv2.resize(cv_image, (300, 300))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        # NOTE: We do NOT apply normalization here, because the training script
        # uses a custom transform that applies normalization.
        # Standard torchvision models expect ToTensor (0-1) then normalization.
        # Let's match the training script's ToTensor + Normalize logic
        
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Ensure tensor stays on CPU to match model
        return [tensor]  # No .to(self.device) needed since we're forcing CPU

    # === MODIFIED POST-PROCESSING FUNCTION ===
    def postprocess_detections(self, outputs, original_shape, model_input_size=(300, 300)):
        """Postprocess model outputs - standard PyTorch output"""
        detections = []
        
        # 'outputs' is now a list of dicts (one per image)
        # Since we process one image, we just need outputs[0]
        if not isinstance(outputs, list) or len(outputs) == 0:
            self.get_logger().warn("Model output was not a list or was empty.", throttle_duration_sec=5.0)
            return []
            
        output = outputs[0] # Get detections for the first (and only) image
        
        if 'boxes' in output and 'scores' in output and 'labels' in output:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            # Filter by confidence
            mask = scores > self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Scale boxes to original image size
            scale_x = original_shape[1] / model_input_size[0]
            scale_y = original_shape[0] / model_input_size[1]
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Ensure valid coordinates
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_shape[1], x2)
                y2 = min(original_shape[0], y2)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(score),
                    'class_id': int(label),
                    'class_name': f'fruit_{label}'
                })
        else:
            self.get_logger().warn(f"Output dict missing expected keys: {output.keys()}", throttle_duration_sec=5.0)

        self.get_logger().info(f"Found {len(detections)} detections", throttle_duration_sec=1.0)
        return detections
    # === END OF MODIFIED SECTION ===

    def image_callback(self, msg: Image):
        """Main image processing callback"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            original_shape = cv_image.shape[:2]
            
            # Preprocess image - now returns list of tensors
            input_tensors = self.preprocess_image(cv_image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensors)

            # Postprocess detections
            detections = self.postprocess_detections(outputs, original_shape)
            
            # Process each detection
            detection_results_2d = []
            detection_results_3d = []
            
            for detection in detections:
                bbox = detection['bbox']
                score = detection['score']
                class_id = detection['class_id']
                
                # Calculate center point for depth lookup
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Get depth
                depth = self.get_depth_at_pixel(center_x, center_y)
                if depth is None:
                    depth = self.find_valid_depth_near(center_x, center_y)
                
                # Calculate 3D position
                if depth is not None:
                    x, y, z = self.calculate_3d_position(bbox, depth)
                else:
                    x, y, z = float('nan'), float('nan'), float('nan')
                
                # Create detection messages
                detection_2d = {
                    'bbox': bbox,
                    'score': score,
                    'class_id': class_id,
                    'timestamp': str(self.get_clock().now().nanoseconds)
                }
                
                detection_3d = {
                    'position': [x, y, z],
                    'score': score,
                    'class_id': class_id,
                    'timestamp': str(self.get_clock().now().nanoseconds)
                }
                
                detection_results_2d.append(detection_2d)
                detection_results_3d.append(detection_3d)
                
                # Publish individual position
                if not np.isnan(x):
                    position_msg = PointStamped()
                    position_msg.header = msg.header
                    position_msg.point.x = x
                    position_msg.point.y = y
                    position_msg.point.z = z
                    self.position_pub.publish(position_msg)
                
                # Draw on image
                self.draw_detection(cv_image, bbox, score, class_id, x, y, z)
            
            # Publish detection results
            if detection_results_2d:
                detection_2d_msg = String()
                detection_2d_msg.data = json.dumps(detection_results_2d)
                self.detection_2d_pub.publish(detection_2d_msg)
            
            if detection_results_3d:
                detection_3d_msg = String()
                detection_3d_msg.data = json.dumps(detection_results_3d)
                self.detection_3d_pub.publish(detection_3d_msg)
            
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            annotated_msg.header = msg.header
            self.annotated_img_pub.publish(annotated_msg)
            
            self.get_logger().info(f"Processed frame: {len(detections)} detections", 
                                throttle_duration_sec=1.0)
            
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    def draw_detection(self, image: np.ndarray, bbox: List[int], score: float, 
                      class_id: int, x: float, y: float, z: float):
        """Draw detection bounding box and information on image"""
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create label
        label = f"Fruit {class_id}: {score:.2f}"
        if not np.isnan(z):
            label += f" ({x:.2f}, {y:.2f}, {z:.2f})"
        
        # Put text
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = FruitVisionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals() and rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()