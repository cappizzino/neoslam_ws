#######################################################################################
#######################################################################################
##################### SYSTEM'S SOFTWARE SPECIFIC CONFIGURATION ########################
#######################################################################################
#######################################################################################
# * use this file to source any additional configuration before launch (e.g. SERVER_IP)

# Path CNN model
export TORCH_HOME='~/neoslam_ws/singularity/mount/model'

# Enable Dimension Reduction creation
export DIMENSION_REDUCTION=0
export MATRIX_HOME=$HOME/neoslam_ws/singularity/mount/data

# ROS bag
export SYS_ROSBAG_ENABLED=0     # enable / disable bag recording (be careful to NOT run long term experiments without bags!)
export SYS_ROSBAG_SIZE='0'      # max size before splitting in Mb (i.e. 0 = infinite, 1024 = 1024Mb = 1Gb)
export SYS_ROSBAG_DURATION='8h'
export SYS_ROSBAG_PATH="$HOME/bag_files/neoslam/latest/"

export SYS_ROSBAG_ARGS="
    --regex
    --split
    --size=$SYS_ROSBAG_SIZE
    --duration=$SYS_ROSBAG_DURATION
    --output-prefix=$SYS_ROSBAG_PATH
"
export SYS_ROSBAG_TOPICS="
    /imu
    /cmd_vel
    /husky(.*)
"

# lio_sam configs
# NOTE: Keep 'LIO_SAM_MAPPING=1'. In the ranger's use-case we have the GPS's global reference.
# Relocalization was built under the assumption that there is not any GPS factor being added.
# Follow https://redmine.ingeniarius.pt/issues/2520 for more information.
export LIO_SAM_MAPPING=1    # enable / disable LIO_SAM's mapping mode
export LIO_SAM_RVIZ=1       # enable / disable LIO_SAM's rviz

# map server
export MAP_SERVER_ENABLED=0                 # enable / disable map server
export MAP_SERVER_MAP_NAME='IngeniariusHQ'  # the name of the saved map (e.g. IngeniariusHQ)

# ROS specific configs
export ROSCONSOLE_FORMAT='[${severity}] [${node}] [${function}] [${line}]: ${message}'
