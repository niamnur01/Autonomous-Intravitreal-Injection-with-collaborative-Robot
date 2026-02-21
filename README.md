# Autonomous Intravitreal Injection with collaborative Robot
This repository provides a ROS2 package, which implements a UR3e performing an autonomous injection, following the eye motion in real-time.

## Installation

### Setup
- Ubuntu 22.04
- [ROS2 Humble](https://docs.ros.org/en/humble/index.html)

#### 1. Required dependencies
Install the following dependencies:
```
pip install setuptools==version wheel==version matplotlib==3.3.4 numpy==1.19.5 opencv-python==4.5.5 pandas==1.1.5 Pillow==8.4.0 scipy==1.5.4 torch==1.10.1 torchvision==0.11.2 
```
```
pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
```
```
pip3 install roboticstoolbox-python
```

#### 2. NVIDIA CUDA
Verify if CUDA is installed on your system
```
nvidia-smi
```
If CUDA is installed, it will display the version of the driver and the supported CUDA version in the top-right corner.
If CUDA is not present, follow the guide that can be found at this [link](https://developer.nvidia.com/cuda-downloads)

### L2CS-Net
- Official git project [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)
- It only requires a common RGB camera (or a webcam)
#### 1. Clone the repository:
```
cd 
git clone https://github.com/Ahmednull/L2CS-Net
```
#### 2. Add pretrained model:

Download L2CSNet_gaze360.pkl located in Gaze360 at this [link](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing)

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

### Robot Controller
The robot is controlled using the [ur_coppeliasim](https://github.com/niamnur01/Universal_Robot_ROS2_CoppeliaSim.git) package.
Clone the repository and follow its README to set up the package, completing all steps up to and including **"8. (Optional) Install controller"**. Once done, resume this installation procedure.

### Prerequisites
#### 1. Install required packages:
ROS 2 Dependencies
```
sudo apt update
sudo apt install ros-humble-map-msgs ros-humble-pendulum-msgs ros-humble-example-interfaces 
```  

#### 2. Clone this repo in your workspace src folder:  
```
cd ~/ros2_ws/src
git clone https://github.com/niamnur01/Autonomous-Intravitreal-Injection-with-collaborative-Robot.git ur3_injection_controller
```
#### 3. Create the interface used in the project:  
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
#### 4. Build workspace:  
```
cd $HOME/ros2_ws/
colcon build --symlink-install
```

## Running the simulation    
#### 0. Source terminals
To run a simulation you will have to open 5 separate terminals.
Remember to always source both the controller_ws and ros2_ws workspaces, in all the terminals:
```
source $HOME/ros2_ws/install/setup.bash
source $HOME/controller_ws/install/setup.bash
```
You can automate this procedure copying these two lines in the .bashrc so that they are executed as soon as you start a terminal. This can be done with:
```
echo $'source $HOME/ros2_ws/install/setup.bash  \nsource $HOME/controller_ws/install/setup.bash' >> $HOME/.bashrc
```

#### 1. Load the world
Open CoppeliaSim
```
cd ~/CoppeliaSim/
./coppeliaSim.sh
```
and load ``coppelia_world.ttt``, located under ``~/ros2_ws/src/ur_coppeliasim``, via ``File->Open_Scene``. 
Before clicking the play button, make sure to select **MuJoCo** as the simulation engine. 
Then click the play button; the robot will move to a predefined HOMING joint configuration.

#### 2. Ocular mapping:
In a new terminal localize the eyes while they are directed forward (repeat step 0.):
```
ros2 run ur3_injection_controller eye_sphere_tracker --calib-frames 25
```
You can change the number after "--calib-frames" or omit it to use the default number of frames 
```
ros2 run ur3_injection_controller eye_sphere_tracker
```

#### 3. Run the hardware interface + cartesian motion controller:
In a new terminal start the controller (repeat step 0.)
```
ros2 launch ur_coppeliasim ur_coppelia_controllers.launch.py
```

#### 4. Gaze tracking node:
Start the gaze tracker while your face is clearly visible to the webcam
```
ros2 run ur3_injection_controller eye_tracking
```
A small window should appear with your webcam view

#### 5. Injection node
Assuming that the controller is running you can run the remaining node in the last terminal (repeat step 0.):
```
ros2 run ur3_injection_controller ur3_injection
```
The robot will start moving in CoppeliaSim

> **Important — Left/Right Eye Convention:** Throughout the scripts, "left" and "right" refer to the perspective of the RGB-D sensor, not the patient. For example, the "left eye" in this codebase is the patient's right eye as seen from the camera's point of view.

## Simulation Support

Two additional scripts are available to extend the simulation with extra functionalities.

### Gaze Replayer

This node substitutes the `eye_tracking` node by publishing pre-recorded gaze angles on the gaze topic. This allows the simulation to run without requiring the user to physically rotate their eyes to the angle needed for the procedure.

The node reads and publishes the orientations found in:
```
ros2_ws/src/ur3_injection_controller/test/eye_movement.txt
```

The `ros2_ws/src/ur3_injection_controller/test` folder contains the following files:
- **`eye_movement.txt`** — currently contains the orientation required to target the left eye.
- **`stable_right.txt`** — contains the orientation for the right eye. To use it, rename it to `eye_movement.txt`.
- **`eye_movement_left.txt`** — contains a sequence of free eye movements. In the middle of the sequence, the eyes pause at the orientation required to target the left eye, then resume moving. Rename it to `eye_movement.txt` to use this movement pattern.

To use this node, replace step **4. Gaze tracking node** with the following (remember to source):
```
ros2 run ur3_injection_controller gaze_replayer
```

### RViz Visualization and Error Measuring

This node allows you to observe the procedure in RViz, visualizing the needle penetrating the eye along the injection vector. It also collects error measurements, saved in:
*File: `ros2_ws/src/ur3_injection_controller/test/E_E error logs`*

#### Setup

First, enable RViz in the controller launch script by removing the `#` on the following line:
*`ros2_ws\src\ur_coppeliasim\launch\ur_coppelia_controllers.launch.py`*
```python
134        rviz, #get rid of the hashtag
```
 
Since the workspace was built with `--symlink-install`, there is no need to rebuild. Simply re-launch:
```
ros2 launch ur_coppeliasim ur_coppelia_controllers.launch.py
```
An RViz window should appear with the robot model.

Make sure the pitch and yaw angles for the injection vector are consistent — see **Injection vector** under both [`ur3_eye_motion`](#ur3_eye_motion) and [`ur3_injection`](#ur3_injection) in the [Customization](#customization) section.

#### Running

Once you have followed all the steps in [Running the simulation](#running-the-simulation), open a new terminal and execute (remember to source):
```
ros2 run ur3_injection_controller ur3_eye_motion
```
In RViz you should see the eye appear along with a small green marker representing the needle.

---

## Customization

This section provides a quick reference for the most common settings you may want to adjust across the workspace.

### `eye_sphere_tracker`
*File: `ros2_ws/src/ur3_injection_controller/ur3_injection_controller/eye_sphere_tracker.py`*

- **Default calibration frames:** Change the value after `default=` to set a different number of calibration frames:
```python
213    parser.add_argument("--calib-frames", type=int, default=1, help="Number of frames to collect for calibration")
```

- **RGB-D sensor clipping planes:** If you modify the near/far clipping planes of the RGB-D sensor in CoppeliaSim, update the corresponding values:
```python
18        self.near = 0.50  # m
19        self.far  = 0.80  # m
```

- **Vision sensor intrinsic parameters:** If you change the intrinsic parameters of the vision sensor, update:
```python
54        self.W = self.H = 1080
55        vfov_rad = math.radians(33.78)
```

- **Vision sensor position:** If you move the RGB-D sensor, update its position with the new coordinates of the vision sensor (not the base of the kinect):
```python
63        self.T = np.array([-0.1301, -0.24275, 0.60139])
```

---

### `ur3_injection`
*File: `ros2_ws/src/ur3_injection_controller/ur3_injection_controller/ur3_injection.py`*

- **Eye selection:** To target the right eye instead of the left, change `"left"` to `"right"`:
```python
60        chosen = centers['left_center']
```

- **Injection vector:** Update the yaw and pitch to your desired injection angle:
```python
91        yaw_deg = 50    # any value within [-80, 80] for left eye, [100, 260] for right eye
92        pitch_deg = 45.5  # within [44.5, 46.5]
```

---

### `ur3_eye_motion`
*File: `ros2_ws/src/ur3_injection_controller/ur3_injection_controller/ur3_eye_motion.py`*

- **Target eye position:** Adjust the coordinates to match the position of the target eye in CoppeliaSim:
```python
36    EYE_POSITION = (-0.164, 0.35274, 0.629)  # [m]
```

- **Injection vector:** Same as above, update yaw and pitch as needed:
```python
65        yaw_deg = 50    # any value within [-80, 80] for left eye, [100, 260] for right eye
66        pitch_deg = 45.5  # within [44.5, 46.5]
```

---

### Robot Controller — Initial Configuration
If you change the robot's starting configuration in CoppeliaSim, make sure to update the following two files with the same joint values:

*`ros2_ws/src/ur_coppeliasim/config/initial_positions.yaml`*
```yaml
shoulder_pan_joint: 1.55
shoulder_lift_joint: -0.65
elbow_joint: -1.8
wrist_1_joint: -1.0
wrist_2_joint: -1.55
wrist_3_joint: 3.14
```

*`ros2_ws/src/ur_coppeliasim/hardware/HWInterface.cpp`*
```cpp
143    float initial_conf[]={1.55, -0.65, -1.8, -1.0, -1.55, 3.14};
```

---

### Robot Controller — Motion Gains
*File: `ros2_ws/src/ur_coppeliasim/config/ur_controllers_coppeliasim.yaml`*

You can tune the PD gains to make the robot movements stiffer or smoother:
```yaml
197    solver:
198        error_scale: 0.5
199        iterations: 1
200        publish_state_feedback: True
201
202    # Gains are w.r.t. the robot_base_link coordinates
203    pd_gains:
204        trans_x: {p: 0.5, d: 0.05}
205        trans_y: {p: 0.5, d: 0.05}
206        trans_z: {p: 0.5, d: 0.05}
207        rot_x: {p: 3.0}
208        rot_y: {p: 3.0}
209        rot_z: {p: 3.0}
```
