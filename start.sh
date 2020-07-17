#!/bin/bash

IMAGE=dc-image
NAME=dc

docker run --rm --name $NAME-container \
          -p 5000:5000 \
          -v $pwd/data:/data
          dc-image
