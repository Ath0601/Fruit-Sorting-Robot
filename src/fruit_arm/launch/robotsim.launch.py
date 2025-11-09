from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, RegisterEventHandler, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg_share = FindPackageShare('fruit_arm')
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_path = PathJoinSubstitution([pkg_share, 'world', 'empty.world'])
    xacro_file = PathJoinSubstitution([pkg_share, 'urdf', 'robot_gripper.urdf.xacro'])

    # Generate robot_description from xacro
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ', xacro_file
    ])

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': ParameterValue(robot_description_content, value_type=str)
        }],
        output='screen'
    )

    # ------------------- Gazebo -------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'gz_args': ['-r ', world_path],
            'on_exit_shutdown': 'true'
        }.items()
    )
    # ------------------- Spawn Robot -------------------
    create_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'QARM_with_Gripper', '-topic', 'robot_description'],
        output='screen'
    )

    # ------------------- Controllers -------------------
    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    # Spawn ARM controller
    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['qarm_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    # Spawn GRIPPER controller  
    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    # ------------------- Camera Bridges -------------------
    bridge_rgb = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/forearm_camera_link/rgb_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '--ros-args', '-r',
            '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/forearm_camera_link/rgb_camera/image:=/camera/color/image_raw'
        ],
        output='screen'
    )

    # RGB camera_info - CORRECTED PATH
    bridge_rgb_caminfo = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/forearm_camera_link/rgb_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '--ros-args', '-r',
            '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/forearm_camera_link/rgb_camera/camera_info:=/camera/color/camera_info'
        ],
        output='screen'
    )

    # Depth image - CORRECTED PATH
    bridge_depth = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/forearm_camera_link/rgbd_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '--ros-args', '-r',
            '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/forearm_camera_link/rgbd_camera/depth_image:=/camera/depth/image_raw'
        ],
        output='screen'
    )

    # Depth camera_info - CORRECTED PATH
    bridge_depth_caminfo = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/forearm_camera_link/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '--ros-args', '-r',
            '/world/empty/model/QARM_with_Gripper/link/FOREARM/sensor/forearm_camera_link/rgbd_camera/camera_info:=/camera/depth/camera_info'
        ],
        output='screen'
    )

    # Delayed start to allow Gazebo to spawn
    delayed_bridges = TimerAction(
        period=7.0,
        actions=[bridge_rgb, bridge_rgb_caminfo, bridge_depth, bridge_depth_caminfo]
    )

    # ------------------- Launch Sequence -------------------
    return LaunchDescription([
        DeclareLaunchArgument(
            name='use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),

        gazebo,
        robot_state_publisher_node,
        create_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=create_node,
                on_exit=[
                   TimerAction(period=4.0, actions=[joint_state_spawner])
                ]
           )
        ),

        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_spawner,
                on_exit=[
                    TimerAction(period=2.0, actions=[arm_controller_spawner])
                ]
            )
        ),

        RegisterEventHandler(
            OnProcessExit(
                target_action=arm_controller_spawner,
                on_exit=[
                    TimerAction(period=2.0, actions=[gripper_controller_spawner])
                ]
            )
        ),
        
        delayed_bridges
    ])
