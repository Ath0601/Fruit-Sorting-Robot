// new_vision.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <mutex>
#include <vector>
#include <string>
#include <optional>
#include <sstream>
#include <iomanip>

using std::placeholders::_1;

class FruitVisionNode : public rclcpp::Node
{
public:
  FruitVisionNode()
  : Node("fruit_vision_node")
  {
    // model path parameter
    this->declare_parameter<std::string>("model_path", "/home/atharva/quanser_ws/ssd_fruit_best.pt");
    this->get_parameter("model_path", model_path_);

    // load TorchScript model
    try {
      module_ = torch::jit::load(model_path_);
      module_.eval();
      RCLCPP_INFO(this->get_logger(), "Loaded TorchScript model: %s", model_path_.c_str());
    } catch (const c10::Error &e) {
      RCLCPP_FATAL(this->get_logger(), "Failed to load model: %s", e.what());
      throw;
    }

    // Publishers
    annotated_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/ssd_detection/image", 1);
    position_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/ssd_detection/position", 10);
    info_pub_ = this->create_publisher<std_msgs::msg::String>("/ssd_detection/info", 10);

    // Subscribers: color image, depth image, camera_info
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/color/image_raw", 10, std::bind(&FruitVisionNode::image_callback, this, _1));
    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/depth/image_raw", 10, std::bind(&FruitVisionNode::depth_callback, this, _1));
    caminfo_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/color/camera_info", 10, std::bind(&FruitVisionNode::caminfo_callback, this, _1));

    RCLCPP_INFO(this->get_logger(), "FruitVisionNode initialized and subscriptions created.");
  }

private:
  // ---------- ROS interfaces ----------
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_img_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr position_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr info_pub_;

  // ---------- Torch model ----------
  torch::jit::script::Module module_;
  std::string model_path_;

  // ---------- Camera intrinsics + latest depth ----------
  std::mutex mutex_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_caminfo_;
  sensor_msgs::msg::Image::SharedPtr latest_depth_msg_;

  // ---------- helpers ----------
  void caminfo_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_caminfo_ = msg;
  }

  void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_depth_msg_ = msg;
  }

  // Read depth from image message at integer pixel (u,v). Handles 16UC1 (mm) and 32FC1 (m).
  std::optional<float> get_depth_at(int u, int v)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!latest_depth_msg_) return std::nullopt;

    const auto &dmsg = *latest_depth_msg_;
    if (u < 0 || v < 0 || u >= static_cast<int>(dmsg.width) || v >= static_cast<int>(dmsg.height))
      return std::nullopt;

    try {
      if (dmsg.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        // float32 meters
        const float *data = reinterpret_cast<const float *>(&dmsg.data[0]);
        float z = data[v * dmsg.width + u];
        if (!std::isfinite(z) || z <= 0.0f) return std::nullopt;
        return z;
      } else if (dmsg.encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
        // uint16 mm -> convert to meters
        const uint16_t *data = reinterpret_cast<const uint16_t *>(&dmsg.data[0]);
        uint16_t d = data[v * dmsg.width + u];
        if (d == 0) return std::nullopt;
        return static_cast<float>(d) / 1000.0f;
      } else if (dmsg.encoding == sensor_msgs::image_encodings::MONO16) {
        const uint16_t *data = reinterpret_cast<const uint16_t *>(&dmsg.data[0]);
        uint16_t d = data[v * dmsg.width + u];
        if (d == 0) return std::nullopt;
        return static_cast<float>(d) / 1000.0f;
      } else {
        // Try to interpret as 32F anyway
        RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 5000, "Unknown depth encoding: %s", dmsg.encoding.c_str());
        return std::nullopt;
      }
    } catch (...) {
      return std::nullopt;
    }
  }

  // If center pixel depth invalid, search small window for valid depth
  std::optional<float> find_valid_depth_near(int u, int v, int radius = 5)
  {
    for (int r = 0; r <= radius; ++r) {
      for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
          int uu = u + dx;
          int vv = v + dy;
          auto d = get_depth_at(uu, vv);
          if (d.has_value()) return d;
        }
      }
    }
    return std::nullopt;
  }

  // Helper to parse camera intrinsics
  bool get_intrinsics(double &fx, double &fy, double &cx, double &cy)
  {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!latest_caminfo_) return false;
    fx = latest_caminfo_->k[0];
    fy = latest_caminfo_->k[4];
    cx = latest_caminfo_->k[2];
    cy = latest_caminfo_->k[5];
    return true;
  }

  // Format detection metadata into a simple JSON-like string
  std::string detection_to_string(int class_id, float score, const cv::Rect &bbox, float depth_m)
  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "{ \"class_id\": " << class_id
       << ", \"score\": " << score
       << ", \"bbox\": [" << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height << "]"
       << ", \"depth_m\": " << depth_m << " }";
    return ss.str();
  }

  // Main image callback: do inference, compute 3D position, publish results
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // Convert ROS->OpenCV
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat image = cv_ptr->image; // original size
    const int orig_w = image.cols;
    const int orig_h = image.rows;

    // Preprocess copy: resize to 300x300 (SSD trained on 300x300)
    const int net_w = 300;
    const int net_h = 300;
    cv::Mat input_img;
    cv::resize(image, input_img, cv::Size(net_w, net_h));

    // BGR -> RGB if model expects RGB. Many torchvision models expect RGB.
    cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);

    // Convert to tensor [1,3,H,W], float, normalized 0..1
    torch::Tensor img_tensor = torch::from_blob(
      input_img.data, {1, net_h, net_w, 3}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // NHWC -> NCHW
    img_tensor = img_tensor.to(torch::kFloat).div(255.0);

    // If model expects normalization (imagenet mean/std) you can apply here.
    // For many SSD models trained without special normalization, this is OK.

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor);

    // Run model
    torch::NoGradGuard no_grad;
    torch::jit::IValue out;
    try {
      out = module_.forward(inputs);
    } catch (const c10::Error &e) {
      RCLCPP_ERROR(this->get_logger(), "Model forward error: %s", e.what());
      return;
    }

    // Parse output: torchvision scripted detectors typically return a list of dicts
    // Each dict: "boxes": Tensor[N,4], "labels": Tensor[N], "scores": Tensor[N]
    std::vector<cv::Rect> boxes2d;
    std::vector<int> class_ids;
    std::vector<float> scores;

    if (out.isTensor()) {
      // If model returned a single tensor, try to interpret (less likely)
      RCLCPP_WARN(this->get_logger(), "Model returned single tensor; expected list of dicts.");
    } else if (out.isTuple()) {
      // some custom script outputs tuple; try to read common pattern
      auto t = out.toTuple();
      // Try to find boxes/scores in tuple (best-effort)
      RCLCPP_WARN(this->get_logger(), "Model returned tuple output; attempting to parse.");
      // fallback: not implemented in detail here
    } else if (out.isList()) {
      auto list = out.toList();
      if (list.size() > 0) {
        auto elem = list.get(0);
        if (elem.isObject()) {
          // sometimes TorchScript returns custom object; try dict
          RCLCPP_WARN(this->get_logger(), "Model returned object type in list (unsupported parsing).");
        } else if (elem.isGenericDict()) {
          // Typical torchvision detection: dict with "boxes","labels","scores"
          auto dict = elem.toGenericDict();
          // get boxes
          if (dict.contains(torch::jit::IValue(std::string("boxes")))) {
            auto boxes_iv = dict.at(torch::jit::IValue(std::string("boxes")));
            auto labels_iv = dict.at(torch::jit::IValue(std::string("labels")));
            auto scores_iv = dict.at(torch::jit::IValue(std::string("scores")));
            torch::Tensor boxes_t = boxes_iv.toTensor(); // Nx4
            torch::Tensor labels_t = labels_iv.toTensor().to(torch::kInt32);
            torch::Tensor scores_t = scores_iv.toTensor();

            auto boxes_acc = boxes_t.accessor<float,2>();
            auto labels_acc = labels_t.accessor<int,1>();
            auto scores_acc = scores_t.accessor<float,1>();

            int N = boxes_t.size(0);
            double scale_x = static_cast<double>(orig_w) / net_w;
            double scale_y = static_cast<double>(orig_h) / net_h;

            for (int i = 0; i < N; ++i) {
              float x1 = boxes_acc[i][0] * scale_x;
              float y1 = boxes_acc[i][1] * scale_y;
              float x2 = boxes_acc[i][2] * scale_x;
              float y2 = boxes_acc[i][3] * scale_y;

              int bx = std::max(0, static_cast<int>(std::round(x1)));
              int by = std::max(0, static_cast<int>(std::round(y1)));
              int bw = std::min(orig_w-1, static_cast<int>(std::round(x2))) - bx;
              int bh = std::min(orig_h-1, static_cast<int>(std::round(y2))) - by;
              if (bw <= 0 || bh <= 0) continue;

              boxes2d.emplace_back(bx, by, bw, bh);
              class_ids.push_back(labels_acc[i]);
              scores.push_back(scores_acc[i]);
            }
          } else {
            RCLCPP_WARN(this->get_logger(), "Model dict doesn't contain 'boxes' key.");
          }
        } else {
          RCLCPP_WARN(this->get_logger(), "Unexpected element type in model output list.");
        }
      }
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unsupported model output type.");
      return;
    }

    // If no detections, publish nothing (or annotated image same as input)
    cv::Mat annotated = image.clone();

    // Get intrinsics
    double fx, fy, cx, cy;
    bool intr_ok = get_intrinsics(fx, fy, cx, cy);
    if (!intr_ok) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *get_clock(), 5000, "No camera_info yet, cannot compute 3D positions.");
    }

    // Iterate detections
    for (size_t i = 0; i < boxes2d.size(); ++i) {
      const cv::Rect &r = boxes2d[i];
      int class_id = class_ids[i];
      float score = scores[i];

      // compute bbox center in image coordinates
      int cx_pix = r.x + r.width / 2;
      int cy_pix = r.y + r.height / 2;

      // get depth at center (m)
      auto depth_opt = get_depth_at(cx_pix, cy_pix);
      if (!depth_opt) depth_opt = find_valid_depth_near(cx_pix, cy_pix, 6);

      float depth_m = depth_opt.value_or(std::numeric_limits<float>::quiet_NaN());

      // compute 3D point in camera frame
      geometry_msgs::msg::PointStamped pt_msg;
      pt_msg.header = msg->header; // same timestamp, frame_id
      if (intr_ok && std::isfinite(depth_m)) {
        double X = (static_cast<double>(cx_pix) - cx) * depth_m / fx;
        double Y = (static_cast<double>(cy_pix) - cy) * depth_m / fy;
        double Z = depth_m;
        pt_msg.point.x = X;
        pt_msg.point.y = Y;
        pt_msg.point.z = Z;
      } else {
        pt_msg.point.x = std::numeric_limits<double>::quiet_NaN();
        pt_msg.point.y = std::numeric_limits<double>::quiet_NaN();
        pt_msg.point.z = std::numeric_limits<double>::quiet_NaN();
      }

      // Publish point
      position_pub_->publish(pt_msg);

      // Publish detection info as JSON-like string
      std_msgs::msg::String info;
      info.data = detection_to_string(class_id, score, r, depth_opt.value_or(-1.0f));
      info_pub_->publish(info);

      // Draw bbox & label on annotated image
      cv::rectangle(annotated, r, cv::Scalar(0, 255, 0), 2);
      std::ostringstream label_ss;
      label_ss << "id:" << class_id << " s:" << std::fixed << std::setprecision(2) << score;
      if (std::isfinite(depth_m)) {
        label_ss << " z:" << std::fixed << std::setprecision(2) << depth_m << "m";
      } else {
        label_ss << " z:NaN";
      }
      std::string label = label_ss.str();
      int baseLine = 0;
      cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
      int ty = std::max(0, r.y - textSize.height - 4);
      cv::rectangle(annotated, cv::Point(r.x, ty), cv::Point(r.x + textSize.width, ty + textSize.height + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
      cv::putText(annotated, label, cv::Point(r.x, ty + textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
    }

    // publish annotated image
    auto out_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, annotated).toImageMsg();
    annotated_img_pub_->publish(*out_msg);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FruitVisionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
