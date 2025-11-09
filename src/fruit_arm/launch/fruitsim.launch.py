from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # --- Package paths ---
    pkg_share = FindPackageShare('fruit_arm')
    moveit_pkg_share = FindPackageShare('moveit_pkg')

    # --- Configurations ---
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_path = PathJoinSubstitution([pkg_share, 'world', 'empty.world'])
    xacro_file = PathJoinSubstitution([pkg_share, 'urdf', 'robot_gripper.urdf.xacro'])

    # --- Generate robot description from xacro ---
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ', xacro_file
    ])
    robot_description = ParameterValue(robot_description_content, value_type=str)

    # --------------------------------------------------------------------
    # 1. Launch Gazebo with the specified world
    # --------------------------------------------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'gz_args': ['-r ', world_path],
            'on_exit_shutdown': 'true'
        }.items()
    )

    # --------------------------------------------------------------------
    # 2. Publish robot state to TF
    # --------------------------------------------------------------------
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description
        }]
    )

    # --------------------------------------------------------------------
    # 3. Spawn the robot entity in Gazebo
    # --------------------------------------------------------------------
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'QARM_with_Gripper', '-topic', 'robot_description'],
        output='screen'
    )

    # --------------------------------------------------------------------
    # 4. Spawn controllers sequentially
    # --------------------------------------------------------------------
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    qarm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['qarm_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    # --------------------------------------------------------------------
    # 5. Launch MoveIt (move_group + RViz)
    # --------------------------------------------------------------------

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([moveit_pkg_share, 'launch', 'move_group.launch.py'])
        ])
    )

    # --------------------------------------------------------------------
    # 6. Bridge Gazebo camera topics → ROS2 standard names (CORRECTED)
    # --------------------------------------------------------------------
    bridges = [
        # RGB image
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/rgb_camera/image'
                '@sensor_msgs/msg/Image[gz.msgs.Image',
                '--ros-args', '-r',
                '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/rgb_camera/image'
                ':=/camera/color/image_raw'
            ],
            output='screen'
        ),
        # RGB camera info
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/rgb_camera/camera_info'
                '@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '--ros-args', '-r',
                '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/rgb_camera/camera_info'
                ':=/camera/color/camera_info'
            ],
            output='screen'
        ),
        # Depth image
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/rgbd_camera/image'
                '@sensor_msgs/msg/Image[gz.msgs.Image',
                '--ros-args', '-r',
                '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/rgbd_camera/image'
                ':=/camera/depth/image_raw'
            ],
            output='screen'
        ),
        # Depth camera info
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/rgbd_camera/camera_info'
                '@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '--ros-args', '-r',
                '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/rgbd_camera/camera_info'
                ':=/camera/depth/camera_info'
            ],
            output='screen'
        )
    ]

    # --------------------------------------------------------------------
    # 7. Define the launch sequence (timed like the UR5e)
    # --------------------------------------------------------------------
    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='true'))

    # Step 1: Gazebo and robot state publisher
    ld.add_action(gazebo)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_robot)

    # Step 2: Controllers (delayed to allow Gazebo to spawn)
    ld.add_action(TimerAction(period=5.0, actions=[joint_state_broadcaster]))
    ld.add_action(TimerAction(period=10.0, actions=[qarm_controller]))
    ld.add_action(TimerAction(period=15.0, actions=[gripper_controller]))

    # Step 3: MoveIt (launch once controllers are active)
    ld.add_action(TimerAction(period=20.0, actions=[moveit_launch]))

    # Step 4: Bridges (start after everything else)
    ld.add_action(TimerAction(period=25.0, actions=bridges))

    return ld
