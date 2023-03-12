#!/bin/bash

# Starts multiple cockroach DB nodes
# * add more if needed

# name of cockroach DB process
PNAME="cockroach"
# location of the running script
DIR_PATH=$(cd $(dirname $0); pwd)

[[ -z $(pgrep $PNAME) ]] && {
    # node 1
    cockroach start-single-node \
        --insecure \
        --store="$DIR_PATH/../../data/cdb/node1" \
        --background; \
    # check if nakama database exists, if not migrate it
    cockroach sql \
        --insecure \
        --execute="USE nakama;" &> /dev/null || {
            echo "Migrating Nakama..."; nakama migrate up;
        }
} || 
echo "Cockroach DB already running..."
