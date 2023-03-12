#######################################################################################
#######################################################################################
######################## SOFTWARE SPECIFIC CONFIGURATION ##############################
#######################################################################################
#######################################################################################
# * use this file to source any additional configuration before launch (e.g. SERVER_IP)

# Path CNN model
export TORCH_HOME='~/neoslam_ws/singularity/mount/model'

# enable Dimension Reduction
export DIMENSION_REDUCTION=0
export MATRIX_HOME=$HOME/neoslam_ws/singularity/mount/data 

# disable husky's EKF
export ENABLE_EKF=false

# enable husky's 3D Lidar
export HUSKY_LASER_3D_ENABLED=1

# run husky's 3D Lidar w/ GPU (enhances performance of simulation)
export HUSKY_LASER_3D_GPU=0

# husky's 3D Lidar range
export HUSKY_LASER_3D_MIN_RANGE=0.3
export HUSKY_LASER_3D_MAX_RANGE=150.0

# husky's 3D Lidar angle
export HUSKY_LASER_3D_MIN_ANGLE=-3.14
export HUSKY_LASER_3D_MAX_ANGLE=3.14

# simulation gazebo GUI
export SIM_GAZEBO_GUI=1

# export HUSKY_LASER_3D_TOWER=1
# export HUSKY_LASER_3D_XYZ="0 0 0"
# export HUSKY_LASER_3D_RPY="0 0 0"
# export HUSKY_LASER_3D_LASERS=16
# export HUSKY_LASER_3D_SAMPLES=1875

# useful for debug
# export ROSCONSOLE_FORMAT='[${severity}] [${node}] [${function}] [${line}]: ${message}'

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################