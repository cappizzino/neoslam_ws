#!/bin/bash

THIS_PATH=`dirname "$0"`
REPO_PATH=`( cd "$THIS_PATH/../.." && pwd )`
THIS_PATH=`( cd "$THIS_PATH" && pwd )`

singularity build \
    --fakeroot \
    --fix-perms \
    -F \
    $REPO_PATH/images/neoslam.sif \
    $THIS_PATH/recipe.def
