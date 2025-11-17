cmake_minimum_required(VERSION 3.8)
project(fruit_arm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =============================
# Torch + CUDA configuration
# =============================
set(CMAKE_PREFIX_PATH "/home/atharva/quanser_ws/libtorch")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/lib/nvidia-cuda-toolkit")
set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")

# =============================
# ROS2 dependencies
# =============================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

# >>> ADDED for C++ MoveIt node
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)

# =============================
# Python executables
# =============================
file(GLOB PYTHON_SCRIPTS "src/*.py")
foreach(script ${PYTHON_SCRIPTS})
  execute_process(
    COMMAND chmod +x ${script}
  )
endforeach()

install(PROGRAMS
  src/new_vision.py
  DESTINATION lib/${PROJECT_NAME}
)

# =============================
# >>> ADDED for C++ MoveIt node
# =============================
add_executable(home_pos src/home_pos.cpp)
ament_target_dependencies(home_pos
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  control_msgs
  trajectory_msgs
)

install(TARGETS home_pos
  DESTINATION lib/${PROJECT_NAME}
)

# =============================
# Install assets (URDF, meshes, config, etc.)
# =============================
install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY world/
  DESTINATION share/${PROJECT_NAME}/world
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
