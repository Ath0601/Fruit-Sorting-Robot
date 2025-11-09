cmake_minimum_required(VERSION 3.8)
project(fruit_arm)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =============================
# Torch + CUDA configuration
# =============================
# Path to LibTorch (you already have this)
set(CMAKE_PREFIX_PATH "/home/atharva/quanser_ws/libtorch")

# Explicitly point to CUDA (for system-installed toolkit)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/lib/nvidia-cuda-toolkit")
set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")

# Optional: suppress harmless CMake warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dev")

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# =============================
# ROS2 dependencies
# =============================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# =============================
# Executable
# =============================
add_executable(fruit_vision_node src/vision.cpp)

ament_target_dependencies(fruit_vision_node
  rclcpp
  cv_bridge
  OpenCV
  sensor_msgs
  geometry_msgs
)

# =============================
# Linking
# =============================
target_link_libraries(fruit_vision_node
  "${TORCH_LIBRARIES}"
  CUDA::cudart
)

set_property(TARGET fruit_vision_node PROPERTY CXX_STANDARD 17)

# =============================
# Install assets (URDF, meshes, etc.)
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

# =============================
# Install the executable
# =============================
install(TARGETS fruit_vision_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()