#!/bin/bash

MAP_NAME=${1:-'map'}
MAP_TOPIC=${2:-'/map'}
SAVE_DIRECTORY="$( cd "$( dirname "$0" )"/../../maps; pwd )"

rosrun map_server map_server "$SAVE_DIRECTORY/$MAP_NAME.yaml" map:="$MAP_TOPIC" map_metadata:="${MAP_TOPIC}_metadata"