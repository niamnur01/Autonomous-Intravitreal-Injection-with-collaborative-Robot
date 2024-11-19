# Autonomous Intravitreal Injection with collaborative Robot
This repository provide a ROS2 package, which implement the trajectory planning for an UR3e performing an autonomous injection, following the eye motion in real-time.

## Installation

### Setup
- Ubuntu 22.04
- [ROS2 Humble](https://docs.ros.org/en/humble/index.html)

#### 1. Required dependencies
Install the following dependendencies:
```
pip install setuptools==version wheel==version matplotlib==3.3.4 numpy==1.19.5 opencv-python==4.5.5 pandas==1.1.5
Pillow==8.4.0 scipy==1.5.4 torch==1.10.1 torchvision==0.11.2 
```
```
pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
```
```
pip3 install roboticstoolbox-python
```

#### 2. Nvida CUDA
Verify if CUDA is installed on your system
```
nvidia-smi
```
If CUDA is installed, it will display the version of the driver and the supported CUDA version in the top-right corner.
If CUDA is not present, follow the guide that can be found at this [link](https://developer.nvidia.com/cuda-downloads)

### L2CS-Net
- Official git project [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)
- It only need a common RGB camera, a webcam
#### 1. Clone the repository:
```
cd 
git clone https://github.com/Ahmednull/L2CS-Net
```
#### 2. Add pretrained model:

Dowload L2CSNet_gaze360.pkl located in Gaze360 at this [link](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing)

Copy the folder in ~/L2CS-Net/models

#### 3. Create python library:
```
cd ~/L2CS-Net
pip install .
```
Now you should be able to import the package with the following command:
```
$ python3
>>> import l2cs
```

### Prerequisites
#### 1. Install required packages:
ROS 2 Dependencies
```
sudo apt update
sudo apt install xsltproc ros-humble-map-msgs ros-humble-pendulum-msgs ros-humble-example-interfaces ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-hardware-interface
```  
#### 2. If you do not have one, create a new ROS2 workspace:
```
mkdir -p ~/ros2_ws/src
```
#### 3. Clone this repo in your workspace src folder:  
```
cd ~/ros2_ws/src
git clone https://github.com/niamnur01/Autonomous-Intravitreal-Injection-with-collaborative-Robot.git ur3_injection_controller
```
#### 4. Create the interface used in the project:  
- Create a new ROS2 package:
```
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake --license Apache-2.0 project_interfaces
cd project_interfaces
```
- Open CMakeLists.txt and append the following lines:
```
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Pose.srv"
)
```
- Open package.xml and append the following lines:
```
<buildtool_depend>rosidl_default_generators</buildtool_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```
- Create the service:
```
mkdir srv
cd srv
touch Pose.srv
```
- Open Pose.srv file and copy inside the following text:
```
int64 r
---
float64 x
float64 y
float64 z
float64 w
```
#### 5. Build workspace:  
```
cd $HOME/ros2_ws/
colcon build --symlink-install
```
#### 6. Controllers
The controllers used in this project make use of a GitHub repository which is no longer available.
This part will be updated in the future

## Running the planning    
#### 0. Source terminals
Remember to always source both the ros2_ws workspaces:
```
source $HOME/ros2_ws/install/setup.bash
```
#### 2. Run the nodes:
Assuming that a controller is already running you can run the two nodes in separate terminal:
```
ros2 run ur3_injection_controller eye_tracking
```
```
ros2 run ur3_injection_controller ur3_injection
```
