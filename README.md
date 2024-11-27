# CAP5415 (Fall 2024)

## Course Project: Visual Odometry

### Installation
Install CARLA 0.9.13: https://github.com/carla-simulator/carla/releases/tag/0.9.13
	- Use [Windows] version: CARLA_0.9.13.zip
	- On Windows side:
		- Run CarlaUE4.exe to start server on localhost:port 2000
		- Run using powershell: .\CarlaUE4.exe -dx11
			- CARLA works without -dx11 until you attempt to load a new world
			- It crashes saying "Out of video memory" when loading a world
	- On Linux side, i.e. WSL2:
		- ip route show | grep -i default | awk '{ print $3}'
		- This will show you the IP address, i.e. {HOST_IP} to connect WSL2 to windows localhost
		- Test connection by running ${CARLA}/PythonAPI/examples/manual_control.py --host {HOST_IP}
			- May also need to:
				- pip3 install --force-reinstall -v "carla==0.9.13"

### Troubleshooting Linux Copy of CARLA Simulator
	- When attempting to run on Linux side directly, i.e. WSL2:
		- Needed to:
			- sudo apt install libomp5
		- Still crashed each time CarlaUE4.sh was run:
			- "4.26.2-0+++UE4+Release-4.26 522 0
                           Disabling core dumps.
                           WARNING: lavapipe is not a conformant vulkan implementation, testing use only.
                           Killed"
		- Gave up and used Windows version instead

### ROS1 Setup (Optional)
Follow ROS1 installation guide: https://github.com/carla-simulator/ros-bridge/blob/master/docs/ros_installation_ros1.md
	- Follow part "B. Using the source repository"
	- If catkin_make fails:
		- Specifically for a Python AttributeError on "importlib_metadata has no entrypoints":
			- pip3 install --upgrade importlib_metadata
	- When selecting egg file use:
		- don't do this step!!! python eggs are deprecated anyway...
	- If roslaunch carla_ros_bridge carla_ros_bridge.launch fails:
		- modify line 1 in ~/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge to:
			- #!/usr/bin/env python3
			- previously shebang was using python2 which didn't have carla installed
		- also specify host since we're on Linux side, i.e. WSL2:
			- Reference above {HOST_IP}
		- also specify longer timeout, since 2 seconds is too short:
			- 10 seconds is fine
		- also set passive to True to avoid ctrl + c on Linux side, i.e. WSL2, from quitting server:
			- roslaunch carla_ros_bridge carla_ros_bridge.launch host:={HOST_IP} timeout:=10 passive:=True
	- Currently using:
		- ROS1 Noetic

### dataset_generator.py (Preferred over ROS-bridge)
Run using `dataset_generator.py --host {HOST_IP} --render`.
`--render` can be omitted if looking to run headless.
Other options available. Use `--help` for more arguments.