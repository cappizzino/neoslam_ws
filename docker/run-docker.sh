#!bin/bash

cd ..

docker run \
	   -it \
	   --rm \
	   -v ${PWD}/:/workspace \
	   neoslam \
	   bash
