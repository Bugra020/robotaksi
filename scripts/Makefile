CARLA_SERVER=~/carla/Dist/CARLA_Shipping_0.9.15-262-g8f7e40f3c-dirty/LinuxNoEditor/CarlaUE4.sh
#CARLA_FLAGS=-RenderOffScreen -nosound -quality-level=Low -fps=10 -benchmark -carla-server
CARLA_FLAGS=-RenderOffScreen -nosound -fps=30 -benchmark -carla-server

ROS_BRIDGE=ros2 launch carla_ros_bridge carla_ros_bridge.launch.py host:=host.docker.internal port:=2000
LIDAR_PUBLISHER=ros2 run carla_lidar_publisher lidar_publisher
STATIC_TF_PUBLISHER=ros2 run carla_lidar_publisher static_tf_publisher
DYNAMIC_TF_PUBLISHER=ros2 run carla_lidar_publisher dynamic_tf_publisher
CLOUD_TO_GRID=ros2 run carla_lidar_publisher occupancy_grid_publisher

run: kill
	@echo "Starting CARLA Simulator in low-performance mode..."
	#@(${CARLA_SERVER} ${CARLA_FLAGS} > /dev/null 2>&1 &)
	@sleep 15

	@echo "Starting ROS2 Bridge..."
	@(${ROS_BRIDGE} > /dev/null 2>&1 &)
	@sleep 5
	
	@echo "Running the static tf publisher..."
	#@(${STATIC_TF_PUBLISHER} > /dev/null 2>&1 &)
	
	@echo "Running the dynamic tf publisher..."
	#@(${DYNAMIC_TF_PUBLISHER} > /dev/null 2>&1 &)

	@echo "Running the LIDAR Script..."
	@(${LIDAR_PUBLISHER})

	#@echo "running the occupancy grid converter..."
	#@(${CLOUD_TO_GRID} > /dev/null 2>&1 &)

kill:
	@echo "Stopping CARLA, ROS bridge, static_tf_publisher and dynamic_tf_publisher..."
	@-pkill -f CarlaUE4 > /dev/null 2>&1 || true
	@-pkill -f carla_ros_bridge > /dev/null 2>&1 || true
	@-pkill -f static_tf_publisher > /dev/null 2>&1 || true
	@-pkill -f dynamic_tf_publisher > /dev/null 2>&1 || true
	@-pkill -f lidar_publisher > /dev/null 2>&1 || true
	@-pkill -f occupancy_grid_publisher > /dev/null 2>&1 || true

restart: kill run

