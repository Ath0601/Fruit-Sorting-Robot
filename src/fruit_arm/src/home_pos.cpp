#include <rclcpp/rclcpp.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <memory>

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;

class QArmMotionNode : public rclcpp::Node
{
public:
    QArmMotionNode() : Node("qarm_motion_node")
    {
        client_ = rclcpp_action::create_client<FollowJointTrajectory>(
            this, "/qarm_controller/follow_joint_trajectory");
            
        // Send trajectory immediately
        send_trajectory();
    }

private:
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr client_;

    void send_trajectory()
    {
        if (!client_->wait_for_action_server(std::chrono::seconds(5))) {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            rclcpp::shutdown();
            return;
        }

        auto goal_msg = FollowJointTrajectory::Goal();
        goal_msg.trajectory.joint_names = {"YAW_joint", "SHOULDER_joint", "ELBOW_joint", "WRIST_joint"};
        
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = {0.0, 0.0, M_PI/6, 0.0};
        point.time_from_start = rclcpp::Duration(5, 0);
        goal_msg.trajectory.points.push_back(point);

        auto send_goal_options = rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();
        
        send_goal_options.result_callback = [this](const auto & result) {
            if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                RCLCPP_INFO(this->get_logger(), "Trajectory execution succeeded!");
            } else {
                RCLCPP_ERROR(this->get_logger(), "Trajectory execution failed");
            }
            rclcpp::shutdown();
        };

        RCLCPP_INFO(this->get_logger(), "Sending trajectory to home position");
        client_->async_send_goal(goal_msg, send_goal_options);
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<QArmMotionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}