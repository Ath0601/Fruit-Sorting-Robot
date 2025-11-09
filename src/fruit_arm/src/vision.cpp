#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

class FruitVisionNode : public rclcpp::Node
{
public:
    FruitVisionNode() : Node("fruit_vision_node")
    {
        // Parameters
        this->declare_parameter<std::string>("model_path", "/home/atharva/quanser_ws/ssd_fruit_best.pt");
        this->declare_parameter<double>("confidence_threshold", 0.5);
        this->declare_parameter<double>("fruit_real_width", 0.07); // 7cm average fruit diameter
        this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");

        // Load parameters
        std::string model_path;
        this->get_parameter("model_path", model_path);
        this->get_parameter("confidence_threshold", confidence_threshold_);
        this->get_parameter("fruit_real_width", fruit_real_width_);

        // Load model
        try {
            module_ = torch::jit::load(model_path);
            module_.eval();
            RCLCPP_INFO(this->get_logger(), "SSD model loaded successfully from: %s", model_path.c_str());
        } catch (const c10::Error &e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading the model: %s", e.what());
            rclcpp::shutdown();
        }

        // Publishers
        detection_2d_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/fruit_detections_2d", 10);
        detection_3d_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("/fruit_detections_3d", 10);

        // Subscribers
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&FruitVisionNode::image_callback, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            this->get_parameter("camera_info_topic").as_string(), 10,
            std::bind(&FruitVisionNode::camera_info_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Fruit Vision Node initialized");
    }

private:
    torch::jit::script::Module module_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_2d_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_3d_pub_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    bool camera_calibrated_ = false;
    double confidence_threshold_;
    double fruit_real_width_;

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (!camera_calibrated_) {
            // Extract camera matrix
            camera_matrix_ = cv::Mat(3, 3, CV_64F);
            for (int i = 0; i < 9; i++) {
                camera_matrix_.at<double>(i/3, i%3) = msg->k[i];
            }

            // Extract distortion coefficients
            dist_coeffs_ = cv::Mat(1, 5, CV_64F);
            for (size_t i = 0; i < msg->d.size(); i++) {
                dist_coeffs_.at<double>(0, i) = msg->d[i];
            }

            camera_calibrated_ = true;
            RCLCPP_INFO(this->get_logger(), "Camera calibration parameters received");
        }
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!camera_calibrated_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                                "Waiting for camera calibration data");
            return;
        }

        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(300, 300));  // SSD expects 300x300

        // Convert OpenCV image to Torch tensor
        torch::Tensor img_tensor = torch::from_blob(
            resized_frame.data, {1, resized_frame.rows, resized_frame.cols, 3}, torch::kByte);
        img_tensor = img_tensor.permute({0, 3, 1, 2});  // NHWC -> NCHW
        img_tensor = img_tensor.toType(torch::kFloat) / 255.0;

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_tensor);

        try {
            torch::NoGradGuard no_grad;
            auto output = module_.forward(inputs).toTuple();

            auto detections = output->elements()[0].toTensor();  // bounding boxes [x1, y1, x2, y2]
            auto labels = output->elements()[1].toTensor();      // class labels
            auto scores = output->elements()[2].toTensor();      // confidence scores

            process_detections(detections, labels, scores, frame, resized_frame.size());

        } catch (const c10::Error &e) {
            RCLCPP_ERROR(this->get_logger(), "Inference failed: %s", e.what());
        }
    }

    void process_detections(const torch::Tensor& detections, const torch::Tensor& labels, 
                           const torch::Tensor& scores, cv::Mat& original_frame, cv::Size model_input_size)
    {
        auto scores_accessor = scores.accessor<float, 1>();
        auto detections_accessor = detections.accessor<float, 2>();
        auto labels_accessor = labels.accessor<int64_t, 1>();

        vision_msgs::msg::Detection2DArray detection_2d_array;
        vision_msgs::msg::Detection3DArray detection_3d_array;

        detection_2d_array.header.stamp = this->now();
        detection_3d_array.header.stamp = this->now();

        float scale_x = static_cast<float>(original_frame.cols) / model_input_size.width;
        float scale_y = static_cast<float>(original_frame.rows) / model_input_size.height;

        for (int i = 0; i < scores.size(0); i++) {
            if (scores_accessor[i] < confidence_threshold_) continue;

            // Get bounding box in original image coordinates
            float x1 = detections_accessor[i][0] * scale_x;
            float y1 = detections_accessor[i][1] * scale_y;
            float x2 = detections_accessor[i][2] * scale_x;
            float y2 = detections_accessor[i][3] * scale_y;

            int label = labels_accessor[i];
            float confidence = scores_accessor[i];

            // Calculate 3D position using pinhole camera model
            geometry_msgs::msg::Point3d fruit_3d_position = calculate_3d_position(x1, y1, x2, y2);

            // Publish 2D detection
            auto detection_2d = create_2d_detection(x1, y1, x2, y2, label, confidence);
            detection_2d_array.detections.push_back(detection_2d);

            // Publish 3D detection
            auto detection_3d = create_3d_detection(fruit_3d_position, label, confidence);
            detection_3d_array.detections.push_back(detection_3d);

            // Visualize on image
            draw_detection(original_frame, x1, y1, x2, y2, label, confidence, fruit_3d_position);
        }

        // Publish all messages
        detection_2d_pub_->publish(detection_2d_array);
        detection_3d_pub_->publish(detection_3d_array);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                            "Detected %zu fruits", detection_2d_array.detections.size());
    }

    geometry_msgs::msg::Point3d calculate_3d_position(float x1, float y1, float x2, float y2)
    {
        geometry_msgs::msg::Point3d position;

        // Calculate center of bounding box
        float center_x = (x1 + x2) / 2.0f;
        float center_y = (y1 + y2) / 2.0f;
        
        // Calculate width in pixels
        float bbox_width = x2 - x1;
        
        // Simple pinhole camera model for depth estimation
        // Using the known real size of fruit and apparent size in image
        double fx = camera_matrix_.at<double>(0, 0);  // focal length in x
        double z = (fruit_real_width_ * fx) / bbox_width;  // depth estimation
        
        // Calculate x and y in camera coordinates
        double x = (center_x - camera_matrix_.at<double>(0, 2)) * z / fx;
        double y = (center_y - camera_matrix_.at<double>(1, 2)) * z / camera_matrix_.at<double>(1, 1);

        position.x = x;
        position.y = y;
        position.z = z;

        return position;
    }

    vision_msgs::msg::Detection2D create_2d_detection(float x1, float y1, float x2, float y2, int label, float confidence)
    {
        vision_msgs::msg::Detection2D detection;
        
        // Bounding box
        detection.bbox.center.position.x = (x1 + x2) / 2.0;
        detection.bbox.center.position.y = (y1 + y2) / 2.0;
        detection.bbox.size_x = x2 - x1;
        detection.bbox.size_y = y2 - y1;
        
        // Results
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        hypothesis.hypothesis.class_id = std::to_string(label);
        hypothesis.hypothesis.score = confidence;
        detection.results.push_back(hypothesis);

        return detection;
    }

    vision_msgs::msg::Detection3D create_3d_detection(const geometry_msgs::msg::Point3d& position, int label, float confidence)
    {
        vision_msgs::msg::Detection3D detection;
        
        detection.bbox.center.position = position;
        detection.bbox.size.x = fruit_real_width_;
        detection.bbox.size.y = fruit_real_width_;
        detection.bbox.size.z = fruit_real_width_;
        
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        hypothesis.hypothesis.class_id = std::to_string(label);
        hypothesis.hypothesis.score = confidence;
        detection.results.push_back(hypothesis);

        return detection;
    }

    void draw_detection(cv::Mat& image, float x1, float y1, float x2, float y2, int label, float confidence,
                       const geometry_msgs::msg::Point3d& position)
    {
        // Draw bounding box
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        
        // Create label text
        std::string label_text = "Class: " + std::to_string(label) + " Conf: " + std::to_string(confidence).substr(0, 4);
        std::string position_text = "Pos: (" + std::to_string(position.x).substr(0, 4) + ", " +
                                   std::to_string(position.y).substr(0, 4) + ", " +
                                   std::to_string(position.z).substr(0, 4) + ")";
        
        // Put text on image
        cv::putText(image, label_text, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, position_text, cv::Point(x1, y1 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FruitVisionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}