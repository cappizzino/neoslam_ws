#######################################################################################
#######################################################################################
######################## SIMULATION SPECIFIC CONFIGURATION ############################
#######################################################################################
#######################################################################################
# * use this file to source any additional configuration before launch (e.g. SERVER_IP)

# clear path robotics' configs
export ENABLE_EKF=0                     # enable / disable localization EKF (keep it disabled for ranger)
export CPR_GAZEBO_PLATFORM='husky'      # keep it as 'husky' for the ranger use-case

# husky's configs (active)
export HUSKY_LASER_3D_ENABLED=1         # enable husky's 3D lidar
export HUSKY_LASER_3D_GPU=1             # enable/disable GPU processing on husky's 3D lidar
export HUSKY_LASER_3D_MIN_RANGE='0.3'
export HUSKY_LASER_3D_MAX_RANGE='150.0'
export HUSKY_LASER_3D_MIN_ANGLE='-3.14'
export HUSKY_LASER_3D_MAX_ANGLE='3.14'
# export HUSKY_LASER_3D_TOWER=1
# export HUSKY_LASER_3D_XYZ="0 0 0"
# export HUSKY_LASER_3D_RPY="0 0 0"
export HUSKY_LASER_3D_LASERS=16
export HUSKY_LASER_3D_SAMPLES=1875

# husky's configs (inactive)
# export HUSKY_REALSENSE_ENABLED=0

# simulation configs
export SIM_GAZEBO_GUI=0         # enable / disable running of gzclient upon launching the simulation
export SIM_GAZEBO_VERBOSE=1     # enable / disable output of warnings and errors in terminal
export SIM_GAZEBO_RECORDING=0   # enable / disable recording of Gazebo session for later playback

export SIM_ROBOT_X='0.0'        # robot's pose
export SIM_ROBOT_Y='0.0'
export SIM_ROBOT_Z='0.2'
export SIM_ROBOT_YAW='0.0'

export SIM_WORLD_X='0.0'        # world's pose
export SIM_WORLD_Y='0.0'
export SIM_WORLD_Z='0.0'
export SIM_WORLD_YAW='0.0'

export SIM_ROSBAG_ENABLED=1     # enable / disable bag recording (be careful to NOT run long term experiments without bags!)
export SIM_ROSBAG_SIZE='0'      # max size before splitting in Mb (i.e. 0 = infinite, 1024 = 1024Mb = 1Gb)
export SIM_ROSBAG_DURATION='8h'
export SIM_ROSBAG_PATH="$HOME/bag_files/ranger/latest/"

export SIM_ROSBAG_ARGS="
    --regex
    --split
    --size=$SIM_ROSBAG_SIZE
    --duration=$SIM_ROSBAG_DURATION
    --output-prefix=$SIM_ROSBAG_PATH
"
export SIM_ROSBAG_TOPICS="
    /imu
    /cmd_vel
    /husky(.*)
    /ground_truth/state
    /odom
    /odom_incremental
    /lio_sam/mapping/odometry
    /lio_sam/mapping/odometry_incremental
    /lio_sam/mapping/map_global/map
    /lio_sam/relocalization/fitness_score
    /lio_sam/relocalization/set_map_to_odom
    /cpu_monitor/(.*)
"

# lio_sam configs
# NOTE: Keep 'LIO_SAM_MAPPING=1'. In the ranger's use-case we have the GPS's global reference.
# Relocalization was built under the assumption that there is not any GPS factor being added.
# Follow https://redmine.ingeniarius.pt/issues/2520 for more information.
export LIO_SAM_MAPPING=1    # enable / disable LIO_SAM's mapping mode
export LIO_SAM_RVIZ=1       # enable / disable LIO_SAM's rviz

# map server
export MAP_SERVER_ENABLED=0                     # enable / disable map server
export MAP_SERVER_MAP_NAME='AgricultureWorld'   # the name of the saved map (e.g. AgricultureWorld)

# ROS specific configs
export ROSCONSOLE_FORMAT='[${severity}] [${node}] [${function}] [${line}]: ${message}'
